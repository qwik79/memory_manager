"""Main orchestrator for the tiered memory manager."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from .active_context import ActiveContext, ContextItem
from .background_worker import BackgroundWorker, CascadeDistiller, Distiller, HeuristicDistiller
from .cold_store import ColdStoreManager
from .eviction import EvictionManager
from .hot_buffer import HotBufferManager
from .models import MemorySession, RetrievalResult
from .storage import atomic_json_write, read_json, safe_id
from .summarizer import HeuristicSummarizer, Summarizer
from .trigger import EmbedFn, OffloadTrigger, TriggerConfig
from .warm_store import WarmStoreManager

logger = logging.getLogger(__name__)

_DISTILLER_DEFAULT = object()  # sentinel distinguishing None from "not provided"


class TieredMemoryManager:
    """Coordinates active, hot, warm, and cold memory tiers.

    Parameters
    ----------
    session_id:
        Unique identifier for this project/session.  Re-using the same ID
        across processes reloads the previously saved session state.
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
    ):
        normalized_session_id = safe_id(session_id)
        paths = base_paths or {}

        # Security layer — opt-in.  Off by default for local-only setups.
        self.security = None
        if secure or encrypt:
            from .security import get_security_manager
            self.security = get_security_manager(encrypt=encrypt)

        self.warm_store = WarmStoreManager(paths.get("warm"), security=self.security)

        # Resolve session file path via the warm store's canonical session dir
        # to avoid duplicating path logic.
        session_dir = self.warm_store.session_dir(normalized_session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
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
            normalized_session_id, paths.get("hot"), security=self.security
        )
        self.cold_store = ColdStoreManager(
            paths.get("cold"), security=self.security, embed_fn=embed_fn,
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
        if not self._session_path.exists():
            return None
        try:
            data = read_json(self._session_path, security=self.security)
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
        try:
            atomic_json_write(
                self._session_path, self.session.to_json(), security=self.security
            )
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
        """Stop the background worker, cancel autosave, and flush session to disk."""
        self._autosave_stop.set()  # wake the autosave thread immediately
        if self._autosave_thread is not None:
            self._autosave_thread.join(timeout=2.0)
            self._autosave_thread = None
        if self.background_worker is not None:
            self.background_worker.stop()
        self._save_session()

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
        self.hot_buffer._rebuild_index()

        logger.info("Imported session '%s' from %s", sid, archive)

    # ------------------------------------------------------------------
    # Public API — message ingestion
    # ------------------------------------------------------------------

    def add_user_message(
        self,
        content: str,
        context_ref: str | None = None,
        tags: list[str] | None = None,
        importance: str = "medium",
        phase: str = "",
    ) -> str:
        entry_id = self.hot_buffer.append_entry(
            "user", content, context_ref=context_ref, tags=tags,
            importance=importance, phase=phase,
        )
        self.eviction_manager.record_reference(entry_id)
        return entry_id

    def add_assistant_message(
        self,
        content: str,
        tags: list[str] | None = None,
        context_ref: str | None = None,
        importance: str = "medium",
        phase: str = "",
    ) -> str:
        entry_id = self.hot_buffer.append_entry(
            "assistant", content, context_ref=context_ref, tags=tags,
            importance=importance, phase=phase,
        )
        self.eviction_manager.record_reference(entry_id)

        # Push the latest exchange pair to the background worker (slow distillers)
        # or call inline (instant distillers, e.g. HeuristicDistiller).
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

        self.last_offload_result = self.evaluate_offload()
        return entry_id

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
