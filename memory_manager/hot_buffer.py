"""Tier 1: hot buffer for recent raw memory entries."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from .models import MemoryEntry
from .scoring import EntryIndex, auto_classify_importance, compute_relevance_score
from .storage import atomic_json_write, default_hot_path, estimate_tokens, read_json, safe_id


class HotBufferManager:
    """Structured file-backed hot buffer for recent session entries."""

    def __init__(
        self,
        session_id: str,
        base_path: str | Path | None = None,
        security: Any = None,
    ):
        self.session_id = safe_id(session_id)
        self.base_path = Path(base_path) if base_path else default_hot_path()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._security = security
        self._index = EntryIndex()
        self._init_file()
        # Populate the index from any persisted entries.
        self._rebuild_index()

    @property
    def file_path(self) -> Path:
        return self.base_path / f"{self.session_id}_hot.json"

    def _init_file(self) -> None:
        if self.file_path.exists():
            return
        atomic_json_write(
            self.file_path,
            {
                "schema_version": 1,
                "session_id": self.session_id,
                "type": "hot_buffer",
                "entries": [],
                "metadata": {"created_at": time.time(), "last_cleanup": time.time()},
            },
            security=self._security,
        )

    def _read_state(self) -> dict[str, Any]:
        return read_json(self.file_path, default={"entries": []}, security=self._security)

    def _write_state(self, data: dict[str, Any]) -> None:
        data.setdefault("schema_version", 1)
        data.setdefault("session_id", self.session_id)
        data.setdefault("type", "hot_buffer")
        data.setdefault("metadata", {})["updated_at"] = time.time()
        atomic_json_write(self.file_path, data, security=self._security)

    def append_entry(
        self,
        entry_type: str,
        content: str,
        context_ref: str | None = None,
        tags: list[str] | None = None,
        relevance_hint: float = 1.0,
        importance: str = "medium",
        phase: str = "",
    ) -> str:
        """Append an entry and return its ID.

        When *importance* is left at the default ``"medium"``, the content
        is auto-classified by regex analysis and may be promoted to
        ``"high"`` or ``"critical"`` (or demoted to ``"low"``).
        """
        # Auto-classify importance when the caller hasn't set an explicit value.
        if importance == "medium":
            importance = auto_classify_importance(content)

        entry = MemoryEntry.create(
            self.session_id,
            entry_type,
            content,
            context_ref=context_ref,
            tags=tags,
            relevance_hint=relevance_hint,
            importance=importance,
            phase=phase,
        )
        entry_dict = entry.to_json()
        with self._lock:
            data = self._read_state()
            data.setdefault("entries", []).append(entry_dict)
            data.setdefault("metadata", {})["entry_count"] = len(data["entries"])
            self._write_state(data)
            # Update the in-memory index.
            score = compute_relevance_score(entry_dict)
            self._index.add(
                entry_id=entry.entry_id,
                tags=entry.tags,
                importance=entry.importance,
                phase=entry.phase,
                score=score,
            )
        return entry.entry_id

    def get_entries(
        self,
        limit: int | None = 50,
        tags_filter: list[str] | None = None,
        context_ref: str | None = None,
        relevance_min: float = 0.0,
        include_compacted: bool = False,
    ) -> list[dict[str, Any]]:
        """Query recent entries from the hot buffer."""
        with self._lock:
            entries = list(self._read_state().get("entries", []))
        if not include_compacted:
            entries = [e for e in entries if not e.get("compacted_at")]
        if tags_filter:
            entries = [e for e in entries if any(t in e.get("tags", []) for t in tags_filter)]
        if context_ref:
            entries = [e for e in entries if e.get("context_ref") == context_ref]
        if relevance_min:
            entries = [e for e in entries if e.get("relevance_hint", 1.0) >= relevance_min]
        return entries[-limit:] if limit else entries

    def mark_compacted(self, entry_ids: list[str]) -> int:
        """Mark hot entries as compacted/offloaded."""
        ids = set(entry_ids)
        now = time.time()
        changed = 0
        with self._lock:
            data = self._read_state()
            for entry in data.get("entries", []):
                if entry.get("entry_id") in ids and not entry.get("compacted_at"):
                    entry["compacted_at"] = now
                    changed += 1
            if changed:
                self._write_state(data)
        return changed

    def tag_entry(self, entry_id: str, tag: str) -> bool:
        """Attach a tag to an entry and update the index."""
        with self._lock:
            data = self._read_state()
            for entry in data.get("entries", []):
                if entry.get("entry_id") == entry_id:
                    entry.setdefault("tags", [])
                    if tag not in entry["tags"]:
                        entry["tags"].append(tag)
                    self._write_state(data)
                    # Update the index: add the tag and recompute score.
                    self._index.add_tag(entry_id, tag)
                    self._index.update_score(
                        entry_id, compute_relevance_score(entry)
                    )
                    return True
        return False

    def get_entries_ranked(
        self,
        limit: int = 20,
        tags_filter: list[str] | None = None,
        importance_min: str | None = None,
        phase: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return entries ordered by composite relevance score, highest first.

        Uses the in-memory :class:`~scoring.EntryIndex` for fast filtering
        and ranking.  Falls back to ``get_entries()`` if the index is empty.
        """
        ranked_ids = self._index.top_k(
            k=limit,
            tags_filter=tags_filter,
            importance_min=importance_min,
            phase=phase,
        )
        if not ranked_ids:
            return self.get_entries(limit=limit, tags_filter=tags_filter)

        # Fetch the actual entry dicts and preserve the ranked order.
        with self._lock:
            entries_by_id = {
                e.get("entry_id"): e
                for e in self._read_state().get("entries", [])
                if not e.get("compacted_at")
            }
        return [
            entries_by_id[eid]
            for eid in ranked_ids
            if eid in entries_by_id
        ]

    def _rebuild_index(self) -> None:
        """Populate the index from persisted entries on startup."""
        entries = self.get_entries(limit=None, include_compacted=False)
        if entries:
            self._index.rebuild(entries)

    def get_entry_score(self, entry_id: str) -> float:
        """Return the composite relevance score for an entry, or 0.0 if unknown.

        Public accessor so callers don't need to reach into ``_index`` directly.
        """
        return self._index.get_score(entry_id)

    def delete_entry(self, entry_id: str) -> bool:
        """Permanently remove an entry from the hot buffer and the index.

        Returns ``True`` if the entry was found and removed, ``False`` otherwise.
        Use :meth:`tag_entry` with ``"discard"`` if you want a soft delete that
        still lets the eviction scorer handle removal gracefully.
        """
        with self._lock:
            data = self._read_state()
            entries = data.get("entries", [])
            original_len = len(entries)
            data["entries"] = [e for e in entries if e.get("entry_id") != entry_id]
            if len(data["entries"]) == original_len:
                return False
            data.setdefault("metadata", {})["entry_count"] = len(data["entries"])
            self._write_state(data)
            self._index.remove(entry_id)
        return True

    def get_size_estimate(self) -> tuple[int, int]:
        """Return estimated tokens and bytes for the hot file."""
        entries = self.get_entries(limit=None, include_compacted=True)
        content = "\n".join(e.get("content", "") for e in entries)
        size = self.file_path.stat().st_size if self.file_path.exists() else 0
        return estimate_tokens(content, overhead=1000), size
