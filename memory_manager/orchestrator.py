"""Main orchestrator for the tiered memory manager."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from .active_context import ActiveContext, ContextItem
from .backends import DocStore, FileDocStore, MemoryDocStore
from .background_worker import BackgroundWorker, CascadeDistiller, Distiller, HeuristicDistiller
from .cold_store import ColdStoreManager
from .eviction import EvictionManager
from .hot_buffer import HotBufferManager
from .models import MemorySession, RetrievalResult
from .storage import default_base_path, default_hot_path, safe_id
from .summarizer import HeuristicSummarizer, Summarizer, render_entries
from .trigger import EmbedFn, OffloadTrigger, TriggerConfig
from .warm_store import WarmStoreManager

logger = logging.getLogger(__name__)

_DISTILLER_DEFAULT = object()  # sentinel distinguishing None from "not provided"


class TieredMemoryManager:
    """Coordinates active, hot, warm, and cold memory tiers.

    Two modes share all tier logic and differ only in where documents live:

    * **Ephemeral** (the default): every tier lives in process memory. Nothing is
      written to disk, there is no autosave thread, and ``close()`` wipes it.
    * **Persistent** (opt-in): JSON files on disk, with optional HMAC/Fernet
      security, autosave, and resume across processes. Selected by passing
      ``persist=True``, ``base_paths``, or using :meth:`persistent`.

    Parameters
    ----------
    session_id:
        Unique identifier for this project/session.  In persistent mode, re-using
        the same ID across processes reloads the previously saved session state.
    project_name:
        Human-readable project label.  Overrides any loaded value when set.
    pinned_summary:
        Seed text for the pinned context.  Overrides any loaded value when set.
    base_paths:
        Optional overrides for ``"hot"``, ``"warm"``, and ``"cold"`` directories.
    summarizer:
        Batch compaction strategy used by ``compact_hot_to_warm()``.
        Defaults to :class:`HeuristicSummarizer`.
    trigger_config:
        Thresholds for the composite offload trigger.
    embed_fn:
        Optional embedding callback enabling semantic-drift scoring.
    encrypt:
        Enable Fernet encryption on top of HMAC signing.  Requires the
        ``cryptography`` package.
    autosave_interval:
        Seconds between background session flushes.  Set to ``0`` to disable.
    distiller:
        :class:`~memory_manager.background_worker.Distiller` used by the
        continuous background worker.  Defaults to
        ``CascadeDistiller([HeuristicDistiller()])`` (zero-AI, always available).
        Pass a :class:`~memory_manager.background_worker.LLMDistiller` or
        :class:`~memory_manager.background_worker.SpeculativeDistiller` for
        LLM-backed distillation.  Set to ``None`` to disable the worker entirely.
    secure:
        Enable HMAC-SHA256 tamper protection on all stored files.  Off by default
        for local-only setups.  Useful for shared or networked environments.
        Ignored in ephemeral mode, where nothing is stored outside the process.
    persist:
        ``True`` for file-backed storage, ``False`` for in-memory only.  ``None``
        (default) means persistent when ``base_paths`` is given, else ephemeral.
    """

    def __init__(
        self,
        session_id: str,
        project_name: str = "",
        pinned_summary: str = "",
        base_paths: dict[str, str | Path] | None = None,
        summarizer: Summarizer | None = None,
        trigger_config: TriggerConfig | None = None,
        embed_fn: EmbedFn | None = None,
        encrypt: bool = False,
        autosave_interval: float = 120.0,
        distiller: Distiller | None = _DISTILLER_DEFAULT,  # type: ignore[assignment]
        secure: bool = False,
        persist: bool | None = None,
    ):
        normalized_session_id = safe_id(session_id)
        paths = base_paths or {}
        if persist is None:
            persist = bool(base_paths)
        self._closed = False

        # Security layer — opt-in, and only meaningful when data reaches disk.
        self.security = None
        if persist and (secure or encrypt):
            from .security import get_security_manager
            self.security = get_security_manager(encrypt=encrypt)

        self.store: DocStore = FileDocStore(self.security) if persist else MemoryDocStore()
        if not persist:
            autosave_interval = 0  # nothing to flush to

        self.warm_store = WarmStoreManager(paths.get("warm"), security=self.security, store=self.store)

        # Resolve session file path via the warm store's canonical session dir
        # to avoid duplicating path logic.
        session_dir = self.warm_store.session_dir(normalized_session_id)
        self.store.ensure_dir(session_dir)
        self._session_path = session_dir / "session.json"

        # Load persisted session if available; fall back to a fresh one.
        loaded = self._load_session(normalized_session_id)
        if loaded is not None:
            self.session = loaded
            if project_name:
                self.session.project_name = project_name
            if pinned_summary:
                self.session.pinned_summary = pinned_summary
        else:
            self.session = MemorySession(
                session_id=normalized_session_id,
                project_name=project_name,
                pinned_summary=pinned_summary,
            )

        self.active_context = ActiveContext(self.session)
        self.hot_buffer = HotBufferManager(
            normalized_session_id, paths.get("hot"), security=self.security, store=self.store
        )
        self.cold_store = ColdStoreManager(
            paths.get("cold"), security=self.security, embed_fn=embed_fn, store=self.store,
        )
        self.eviction_manager = EvictionManager(self.hot_buffer)
        self.summarizer = summarizer or HeuristicSummarizer()
        self.offload_trigger = OffloadTrigger(trigger_config)
        self.embed_fn = embed_fn
        self.pinned_embedding = (
            embed_fn(self.session.pinned_summary)
            if embed_fn and self.session.pinned_summary
            else None
        )
        self.last_offload_result: dict[str, Any] | None = None

        # Background distillation worker — or inline for instant distillers.
        if distiller is _DISTILLER_DEFAULT:
            distiller = CascadeDistiller([HeuristicDistiller()])

        if distiller is None:
            self.background_worker: BackgroundWorker | None = None
            self._inline_distiller: Distiller | None = None
        elif getattr(distiller, "is_instant", False):
            # Instant distillers (e.g. HeuristicDistiller) have microsecond cost.
            # Running them on a daemon thread adds queue/locking overhead for no
            # benefit — call them inline in add_assistant_message instead.
            self.background_worker = None
            self._inline_distiller = distiller
        else:
            self.background_worker = BackgroundWorker(
                distiller=distiller,
                on_result=self.apply_distillation,
                get_summary=self.get_pinned_summary,
            )
            self._inline_distiller = None

        # Autosave — single daemon thread with Event-based loop.
        # Unlike recursive threading.Timer, this cannot drift or leak threads.
        self._autosave_interval = autosave_interval
        self._autosave_stop = threading.Event()
        self._autosave_thread: threading.Thread | None = None
        if autosave_interval > 0:
            self._autosave_thread = threading.Thread(
                target=self._autosave_loop,
                name=f"memory-autosave-{safe_id(session_id)}",
                daemon=True,
            )
            self._autosave_thread.start()

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def _load_session(self, session_id: str) -> MemorySession | None:
        if not self.store.exists(self._session_path):
            return None
        try:
            data = self.store.read(self._session_path) or {}
            session = MemorySession.from_json(data)
            logger.info("Resumed session '%s' from disk.", session_id)
            return session
        except Exception as exc:
            # If security is enabled and a tamper is detected, the
            # security.verify call already raises TamperDetectedError
            # which propagates uncaught — correct for a poisoned session.
            logger.warning(
                "Could not load session '%s': %s. Starting fresh.", session_id, exc
            )
            return None

    def _save_session(self) -> None:
        if getattr(self, "_discarded", False):
            return
        try:
            self.store.write(self._session_path, self.session.to_json())
        except Exception as exc:
            logger.warning("Failed to save session state: %s", exc)

    def _autosave_loop(self) -> None:
        """Daemon thread that periodically saves session state.

        Uses ``Event.wait(timeout)`` instead of recursive ``Timer`` objects.
        The event is set by :meth:`close` to wake the thread for immediate
        shutdown — no orphan timers, no drift.
        """
        while not self._autosave_stop.wait(timeout=self._autosave_interval):
            self._save_session()

    def close(self) -> None:
        """End the session.

        Stops the background worker and autosave. Persistent sessions are flushed
        to disk; ephemeral sessions are wiped, leaving no trace.
        """
        if self._closed:
            return
        self._autosave_stop.set()  # wake the autosave thread immediately
        if self._autosave_thread is not None:
            self._autosave_thread.join(timeout=2.0)
            self._autosave_thread = None
        if self.background_worker is not None:
            self.background_worker.stop()
        if self.is_persistent:
            self._save_session()
        else:
            self.wipe()
        self._closed = True

    def __enter__(self) -> "TieredMemoryManager":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API — session state accessors
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self.session.session_id

    # ------------------------------------------------------------------
    # Modes
    # ------------------------------------------------------------------

    @classmethod
    def ephemeral(cls, session_id: str = "session", **kwargs: Any) -> "TieredMemoryManager":
        """In-process memory only; leaves no trace after ``close()`` or process exit."""
        kwargs.pop("base_paths", None)
        return cls(session_id, persist=False, **kwargs)

    @classmethod
    def persistent(
        cls,
        base_path: str | Path | None = None,
        session_id: str = "session",
        **kwargs: Any,
    ) -> "TieredMemoryManager":
        """File-backed memory.

        With *base_path*, tiers live in ``hot/``, ``warm/`` and ``cold/`` under it
        (a tmpfs path gives RAM speed that lasts until reboot). Without it, the
        standard locations apply (``MEMORY_MANAGER_HOME``, tmpfs hot buffer).
        """
        if base_path is not None:
            root = Path(base_path).expanduser()
            kwargs["base_paths"] = {"hot": root / "hot", "warm": root / "warm", "cold": root / "cold"}
        return cls(session_id, persist=True, **kwargs)

    @property
    def is_persistent(self) -> bool:
        return bool(self.store.persistent)

    @property
    def is_ephemeral(self) -> bool:
        return not self.is_persistent

    @property
    def closed(self) -> bool:
        return self._closed

    def wipe(self) -> None:
        """Delete this session's memory from every tier, whatever the backend.

        Session state (pinned summary, decisions) is reset too, and nothing is
        flushed back to disk by autosave or ``close()`` unless the session is used
        again afterwards.
        """
        self._discarded = True
        self.session = MemorySession(session_id=self.session_id)
        self.active_context.session = self.session
        self.hot_buffer.clear()
        self.warm_store.clear(self.session_id)
        self.cold_store.clear(self.session_id)
        self.eviction_manager.access_history.clear()
        self.eviction_manager.reference_count.clear()

    def promote(self, base_path: str | Path | None = None) -> None:
        """Move an ephemeral session to file storage and continue persistently.

        For "actually, keep this one". Documents are copied to *base_path* (or the
        standard locations), the in-memory copies are wiped, and autosave starts.
        """
        if self.is_persistent:
            return
        sid = self.session_id
        old_store = self.store
        hot_entries = self.hot_buffer.get_entries(limit=None, include_compacted=True)
        sections = self.warm_store.list_sections(sid)
        chunks = self.cold_store.list_chunks(sid)

        new_store = FileDocStore(self.security)
        root = Path(base_path).expanduser() if base_path is not None else None
        self.hot_buffer.store = new_store
        self.hot_buffer.base_path = (root / "hot") if root else default_hot_path()
        self.warm_store.store = new_store
        self.warm_store.base_path = (root / "warm") if root else default_base_path("llm_warm")
        self.cold_store.store = new_store
        self.cold_store.base_path = (root / "cold") if root else default_base_path("llm_cold")
        self.store = new_store
        self._session_path = self.warm_store.session_dir(sid) / "session.json"

        self.hot_buffer._init_file()
        self.hot_buffer._write_state({"entries": hot_entries})
        for section in sections:
            self.warm_store.put_section(sid, section)
        for chunk in chunks:
            self.cold_store.put_chunk(sid, chunk)
        self._save_session()
        if isinstance(old_store, MemoryDocStore):
            old_store.clear()

        if self._autosave_interval <= 0:
            self._autosave_interval = 120.0
        self._autosave_stop.clear()
        if self._autosave_thread is None:
            self._autosave_thread = threading.Thread(
                target=self._autosave_loop, name=f"memory-autosave-{sid}", daemon=True
            )
            self._autosave_thread.start()

    def get_pinned_summary(self) -> str:
        """Return the current pinned summary string.

        Used by the :class:`~memory_manager.background_worker.BackgroundWorker`
        as the ``get_summary`` callback for incremental distillation.
        """
        return self.session.pinned_summary

    # ------------------------------------------------------------------
    # Public API — session export / import
    # ------------------------------------------------------------------

    def export_session(self, dest: str | Path) -> Path:
        """Export the current session to a portable tar.gz archive.

        Includes:
        - ``session.json`` — session metadata and pinned summary
        - ``warm/`` — all warm section files
        - ``cold/`` — all cold chunk files
        - ``hot/`` — hot buffer snapshot

        Parameters
        ----------
        dest:
            Output file path.  ``.tar.gz`` is appended if not present.

        Returns
        -------
        Path
            Absolute path to the created archive.
        """
        import tarfile

        self._save_session()

        dest = Path(dest)
        if not dest.name.endswith(".tar.gz"):
            dest = dest.with_suffix(".tar.gz")
        dest.parent.mkdir(parents=True, exist_ok=True)

        sid = self.session_id

        if self.is_ephemeral:
            # In-memory documents: serialize each one under the same archive layout.
            import io
            import json as _json

            def add_doc(tar: tarfile.TarFile, arcname: str, doc: dict[str, Any] | None) -> None:
                if doc is None:
                    return
                raw = (_json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
                info = tarfile.TarInfo(arcname)
                info.size = len(raw)
                tar.addfile(info, io.BytesIO(raw))

            with tarfile.open(dest, "w:gz") as tar:
                add_doc(tar, "session.json", self.store.read(self._session_path))
                for path in self.store.list(self.warm_store._section_dir(sid)):
                    add_doc(tar, f"warm/{path.name}", self.store.read(path))
                for path in self.store.list(self.cold_store._index_dir(sid)):
                    add_doc(tar, f"cold/{path.name}", self.store.read(path))
                add_doc(tar, "hot/hot_buffer.json", self.store.read(self.hot_buffer.file_path))
            logger.info("Exported ephemeral session '%s' to %s", sid, dest)
            return dest.resolve()

        with tarfile.open(dest, "w:gz") as tar:
            # Session state.
            if self._session_path.exists():
                tar.add(self._session_path, arcname="session.json")

            # Warm sections.
            warm_dir = self.warm_store._section_dir(sid)
            if warm_dir.exists():
                for f in warm_dir.glob("*.json"):
                    tar.add(f, arcname=f"warm/{f.name}")

            # Cold chunks.
            cold_dir = self.cold_store._index_dir(sid)
            if cold_dir.exists():
                for f in cold_dir.glob("*.json"):
                    tar.add(f, arcname=f"cold/{f.name}")

            # Hot buffer.
            hot_path = self.hot_buffer.file_path
            if hot_path.exists():
                tar.add(hot_path, arcname="hot/hot_buffer.json")

        logger.info("Exported session '%s' to %s", sid, dest)
        return dest.resolve()

    def import_session(self, archive: str | Path) -> None:
        """Import a session from a tar.gz archive created by :meth:`export_session`.

        Replaces the current session state with the archived one.  The hot
        buffer index is rebuilt automatically on the next access.

        Parameters
        ----------
        archive:
            Path to the ``.tar.gz`` archive.
        """
        import tarfile

        archive = Path(archive)
        if not archive.exists():
            raise FileNotFoundError(f"Archive not found: {archive}")

        sid = self.session_id

        if self.is_ephemeral:
            import json as _json

            targets = {
                "session.json": lambda name: self._session_path,
                "warm/": lambda name: self.warm_store._section_dir(sid) / Path(name).name,
                "cold/": lambda name: self.cold_store._index_dir(sid) / Path(name).name,
                "hot/": lambda name: self.hot_buffer.file_path,
            }
            with tarfile.open(archive, "r:gz") as tar:
                for member in tar.getmembers():
                    f = tar.extractfile(member) if member.isfile() else None
                    if f is None:
                        continue
                    doc = _json.loads(f.read())
                    if isinstance(doc, dict) and doc.get("env") == 1 and "sig" in doc:
                        raise ValueError(
                            "Archive contains signed/encrypted documents; import it into a "
                            "persistent manager with the same security settings."
                        )
                    for prefix, target in targets.items():
                        if member.name == prefix or (prefix.endswith("/") and member.name.startswith(prefix)):
                            self.store.write(target(member.name), doc)
                            break
            loaded = self._load_session(sid)
            if loaded is not None:
                self.session = loaded
                self.active_context.session = loaded
            self.hot_buffer._rebuild_index()
            logger.info("Imported session '%s' from %s into memory", sid, archive)
            return

        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = member.name
                f = tar.extractfile(member)
                if f is None:
                    continue
                data = f.read()

                if name == "session.json":
                    self._session_path.parent.mkdir(parents=True, exist_ok=True)
                    self._session_path.write_bytes(data)
                elif name.startswith("warm/"):
                    dest = self.warm_store._section_dir(sid) / Path(name).name
                    dest.write_bytes(data)
                elif name.startswith("cold/"):
                    dest = self.cold_store._index_dir(sid) / Path(name).name
                    dest.write_bytes(data)
                elif name.startswith("hot/"):
                    self.hot_buffer.file_path.parent.mkdir(parents=True, exist_ok=True)
                    self.hot_buffer.file_path.write_bytes(data)

        # Reload session state and rebuild hot buffer index.
        loaded = self._load_session(sid)
        if loaded is not None:
            self.session = loaded
            self.active_context.session = loaded
        self.hot_buffer._rebuild_index()

        logger.info("Imported session '%s' from %s", sid, archive)

    # ------------------------------------------------------------------
    # Public API — message ingestion
    # ------------------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
        context_ref: str | None = None,
        tags: list[str] | None = None,
        importance: str = "medium",
        phase: str = "",
        auto_offload: bool = True,
    ) -> str:
        """Append a message of any role.

        Assistant messages feed the distiller (background or inline). When
        *auto_offload* is true they also run the composite offload trigger, as
        :meth:`add_assistant_message` always has. :class:`ContextVirtualizer`
        passes ``auto_offload=False`` because it makes turn-aligned offload
        decisions itself.
        """
        self._discarded = False
        entry_id = self.hot_buffer.append_entry(
            role, content, context_ref=context_ref, tags=tags,
            importance=importance, phase=phase,
        )
        self.eviction_manager.record_reference(entry_id)
        if role == "assistant":
            self._distill_latest()
            if auto_offload:
                self.last_offload_result = self.evaluate_offload()
        return entry_id

    def add_user_message(
        self,
        content: str,
        context_ref: str | None = None,
        tags: list[str] | None = None,
        importance: str = "medium",
        phase: str = "",
    ) -> str:
        return self.add_message(
            "user", content, context_ref=context_ref, tags=tags, importance=importance, phase=phase,
        )

    def add_assistant_message(
        self,
        content: str,
        tags: list[str] | None = None,
        context_ref: str | None = None,
        importance: str = "medium",
        phase: str = "",
    ) -> str:
        return self.add_message(
            "assistant", content, context_ref=context_ref, tags=tags, importance=importance, phase=phase,
        )

    def _distill_latest(self) -> None:
        """Push the latest exchange pair to the background worker (slow distillers)
        or distill inline (instant distillers, e.g. HeuristicDistiller)."""
        if self.background_worker is not None:
            recent = self.hot_buffer.get_entries(limit=2)
            delta = [
                {"role": e.get("entry_type", "unknown"), "content": e.get("content", "")}
                for e in recent
            ]
            self.background_worker.push(delta)
        elif self._inline_distiller is not None:
            recent = self.hot_buffer.get_entries(limit=2)
            delta = [
                {"role": e.get("entry_type", "unknown"), "content": e.get("content", "")}
                for e in recent
            ]
            result = self._inline_distiller.distill(delta, self.get_pinned_summary())
            if result.get("summary"):
                self.apply_distillation(result)

    def add_decision(self, description: str, decision_type: str = "general") -> None:
        self.session.active_decisions.append(
            {"type": decision_type, "description": description}
        )

    # ------------------------------------------------------------------
    # Public API — compaction and retrieval
    # ------------------------------------------------------------------

    def compact_hot_to_warm(
        self, section_name: str = "session_summary", limit: int = 40
    ) -> str | None:
        """Summarize recent un-compacted hot entries into a warm section."""
        entries = self.hot_buffer.get_entries(limit=limit)
        if not entries:
            return None
        summary = self.summarizer.summarize_entries(entries, self._summary_context())
        content = summary.get("summary", "")
        if not content:
            return None
        source_ids = [e.get("entry_id") for e in entries if e.get("entry_id")]
        section_id = self.warm_store.store_section(
            self.session_id,
            section_name,
            content,
            metadata={
                "summary_confidence": summary.get("confidence"),
                "decisions": summary.get("decisions", []),
                "facts": summary.get("facts", {}),
                "next_steps": summary.get("next_steps", []),
                "tier": summary.get("tier", "extend"),
                "notes_for_core": summary.get("notes_for_core", ""),
            },
            source_entry_ids=source_ids,
            raw_content=render_entries(entries),
        )
        self.hot_buffer.mark_compacted(source_ids)
        self.apply_distillation(summary)

        # Auto-cascade: index the new section into cold immediately so it is
        # searchable without a separate index_warm_to_cold() call.
        try:
            section_data = self.warm_store.get_section(self.session_id, section_id)
            if section_data:
                self.cold_store.index_warm_section(section_data)
                self.warm_store.mark_indexed(self.session_id, section_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auto-cascade warm→cold failed for %s: %s", section_id, exc)

        return section_id

    def offload_entries(
        self,
        entry_ids: list[str],
        section_name: str = "offloaded_turns",
        chunk_size: int = 80,
        chunk_overlap: int = 20,
    ) -> str | None:
        """Distill specific hot entries into one warm section and index it into cold.

        Used by :class:`ContextVirtualizer` for turn-aligned offloads. Unlike
        :meth:`compact_hot_to_warm` it does not overwrite the pinned summary with
        the compaction summary (the distiller keeps that current per turn), but it
        does carry decisions and next steps into session state. Conversation
        excerpts are retrieved into a tight token budget, so they are chunked
        smaller than the cold store's document default. Returns the section ID,
        or ``None`` when nothing was left to offload.
        """
        entries = [e for e in self.hot_buffer.get_entries_by_id(entry_ids) if not e.get("compacted_at")]
        if not entries:
            return None
        summary = self.summarizer.summarize_entries(entries, self._summary_context())
        source_ids = [e["entry_id"] for e in entries]
        section_id = self.warm_store.store_section(
            self.session_id,
            section_name,
            summary.get("summary", "") or render_entries(entries),
            metadata={
                "summary_confidence": summary.get("confidence"),
                "decisions": summary.get("decisions", []),
                "facts": summary.get("facts", {}),
                "next_steps": summary.get("next_steps", []),
                "tier": summary.get("tier", "extend"),
                "chunking": {"size": chunk_size, "overlap": chunk_overlap},
            },
            source_entry_ids=source_ids,
            raw_content=render_entries(entries),
        )
        self.hot_buffer.mark_compacted(source_ids)
        self.apply_distillation({
            "decisions": summary.get("decisions", []),
            "next_steps": summary.get("next_steps", []),
            "notes_for_core": self.session.memory_notes,
        })
        try:
            section = self.warm_store.get_section(self.session_id, section_id)
            if section and self.cold_store.index_warm_section(section):
                self.warm_store.mark_indexed(self.session_id, section_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Offload warm→cold indexing failed for %s: %s", section_id, exc)
        return section_id

    def evaluate_offload(self) -> dict[str, Any]:
        """Evaluate the composite trigger and compact automatically when it fires."""
        entries = self.hot_buffer.get_entries(limit=None)
        result = self.offload_trigger.evaluate(entries, self.pinned_embedding, self.embed_fn)
        if result["should_offload"]:
            # Drain the background worker first so its distillation of the most
            # recent exchange is applied to session state before compaction.
            if self.background_worker is not None:
                self.background_worker.flush()
            result["warm_section_id"] = self.compact_hot_to_warm()
        else:
            result["warm_section_id"] = None
        return result

    def index_warm_to_cold(self, include_indexed: bool = False) -> list[str]:
        """Index warm sections into the cold tier."""
        created: list[str] = []
        for section in self.warm_store.list_sections(
            self.session_id, include_indexed=include_indexed
        ):
            if include_indexed and section.get("indexed_at"):
                self.cold_store.delete_chunks_for_source_section(
                    self.session_id, section.get("section_id")
                )
            chunk_ids = self.cold_store.index_warm_section(section)
            if chunk_ids:
                created.extend(chunk_ids)
                self.warm_store.mark_indexed(self.session_id, section["section_id"])
        return created

    def refresh_context(self, query: str, k: int = 8) -> list[RetrievalResult]:
        """Retrieve relevant warm and cold memories for a query."""
        warm = self.warm_store.retrieve_sections(
            self.session_id, query=query, k=max(1, k // 2)
        )
        cold = self.cold_store.retrieve(self.session_id, query=query, k=k)
        return sorted([*warm, *cold], key=lambda r: r.relevance, reverse=True)[:k]

    def build_context_for_llm(self, query: str = "", max_tokens: int = 32_000) -> str:
        """Build a prompt-ready context block under a token budget.

        Hot buffer entries are ranked by composite relevance score so that
        the most important and recent information appears first in the
        prompt.  Retrieved warm/cold results are interleaved by their
        own relevance scores.
        """
        items: list[ContextItem] = []
        for entry in self.hot_buffer.get_entries_ranked(limit=20):
            # Use the index score (0–1) scaled to 1–10 for priority.
            score = self.hot_buffer.get_entry_score(entry.get("entry_id", ""))
            priority = max(1, int(score * 10))
            items.append(
                ContextItem(content=self._format_hot_entry(entry), priority=priority)
            )
        if query:
            for result in self.refresh_context(query, k=6):
                items.append(
                    ContextItem(
                        content=self._format_retrieval_result(result),
                        priority=int(result.relevance * 10),
                    )
                )
        return self.active_context.build_context(items, max_tokens=max_tokens)

    def maybe_offload(self, current_task: str = "") -> dict[str, Any]:
        """Evaluate eviction and compact selected entries when needed.

        Routes evicted entries through :meth:`compact_hot_to_warm` so they
        benefit from auto-cascade to cold storage and distillation.
        """
        offload_ids, pin_ids = self.eviction_manager.evaluate_for_eviction(
            current_task=current_task
        )
        for entry_id in pin_ids:
            self.hot_buffer.tag_entry(entry_id, "pin")
        section_id = None
        if offload_ids:
            # Summarize the eviction candidates and route through the full
            # compaction pipeline (auto-cascade + distillation).
            entries = [
                e for e in self.hot_buffer.get_entries(limit=None)
                if e.get("entry_id") in set(offload_ids)
            ]
            if entries:
                summary = self.summarizer.summarize_entries(entries, self._summary_context())
                content = summary.get("summary", "")
                if content:
                    source_ids = [e.get("entry_id") for e in entries if e.get("entry_id")]
                    section_id = self.warm_store.store_section(
                        self.session_id,
                        "evicted_hot_entries",
                        content,
                        metadata={
                            "summary_confidence": summary.get("confidence"),
                            "evicted_entry_count": len(entries),
                            "tier": summary.get("tier", "extend"),
                        },
                        source_entry_ids=source_ids,
                        raw_content=render_entries(entries),
                    )
                    self.hot_buffer.mark_compacted(source_ids)
                    self.apply_distillation(summary)
                    # Auto-cascade to cold.
                    try:
                        section_data = self.warm_store.get_section(
                            self.session_id, section_id
                        )
                        if section_data:
                            self.cold_store.index_warm_section(section_data)
                            self.warm_store.mark_indexed(
                                self.session_id, section_id
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Eviction auto-cascade warm→cold failed for %s: %s",
                            section_id, exc,
                        )
        return {"offloaded": offload_ids, "pinned": pin_ids, "warm_section_id": section_id}

    # ------------------------------------------------------------------
    # Public API — distillation (called by BackgroundWorker)
    # ------------------------------------------------------------------

    def apply_distillation(self, summary: dict[str, Any]) -> None:
        """Apply a distillation result to the live session state.

        This is the callback registered with the background worker and the
        speculative distiller's refinement thread.  It is intentionally public
        so that callers can also apply distillation results from external sources.
        """
        distilled = str(summary.get("summary", "")).strip()
        if distilled:
            self.session.pinned_summary = distilled
            if self.embed_fn:
                self.pinned_embedding = self.embed_fn(distilled)
        for decision in summary.get("decisions", []):
            if decision and not any(
                item.get("description") == decision
                for item in self.session.active_decisions
            ):
                self.add_decision(str(decision), "distilled")
        self.session.memory_notes = str(summary.get("notes_for_core", "")).strip()
        for next_step in summary.get("next_steps", []):
            step = str(next_step).strip()
            if step and step not in self.session.open_questions:
                self.session.open_questions.append(step)

        # Act on the tier recommendation from the distiller.
        tier = str(summary.get("tier", "extend")).strip().lower()
        if tier == "pin":
            # Tag the most recent un-compacted hot entries so eviction scoring
            # keeps them in the buffer longer.
            for entry in self.hot_buffer.get_entries(limit=5):
                entry_id = entry.get("entry_id")
                if entry_id and "pin" not in entry.get("tags", []):
                    self.hot_buffer.tag_entry(entry_id, "pin")
        elif tier == "archive":
            # Resolved content — cascade un-indexed warm sections to cold so
            # the warm tier stays lean.
            try:
                self.index_warm_to_cold()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tier=archive cold cascade failed: %s", exc)

    # Keep the private alias for any code that calls _apply_distillation directly.
    _apply_distillation = apply_distillation

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _summary_context(self) -> dict[str, Any]:
        return {
            "current_task": self.session.current_task,
            "pinned_summary": self.session.pinned_summary,
        }

    def _format_hot_entry(self, entry: dict[str, Any]) -> str:
        metadata = [
            "tier=hot",
            f"id={entry.get('entry_id', '')}",
            f"type={entry.get('entry_type', '')}",
        ]
        if entry.get("tags"):
            metadata.append(f"tags={','.join(entry.get('tags', []))}")
        if entry.get("created_at"):
            metadata.append(f"created_at={entry.get('created_at')}")
        return f"[Memory {' '.join(metadata)}]\n{entry.get('content', '')}"

    def _format_retrieval_result(self, result: RetrievalResult) -> str:
        metadata = [
            f"source={result.source}",
            f"id={result.identifier or ''}",
            f"relevance={result.relevance:.3f}",
        ]
        if result.metadata.get("section_name"):
            metadata.append(f"section={result.metadata['section_name']}")
        tags = result.metadata.get("tags")
        if tags:
            metadata.append(f"tags={','.join(tags)}")
        return f"[Memory {' '.join(metadata)}]\n{result.content}"
