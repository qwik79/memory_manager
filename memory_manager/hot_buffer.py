"""Tier 1: hot buffer for recent raw memory entries."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from .models import MemoryEntry
from .storage import atomic_json_write, default_base_path, estimate_tokens, read_json, safe_id


class HotBufferManager:
    """Structured file-backed hot buffer for recent session entries."""

    def __init__(self, session_id: str, base_path: str | Path | None = None):
        self.session_id = safe_id(session_id)
        self.base_path = Path(base_path) if base_path else default_base_path("llm_hot_buffer")
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_file()

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
        )

    def _read_state(self) -> dict[str, Any]:
        return read_json(self.file_path, default={"entries": []})

    def _write_state(self, data: dict[str, Any]) -> None:
        data.setdefault("schema_version", 1)
        data.setdefault("session_id", self.session_id)
        data.setdefault("type", "hot_buffer")
        data.setdefault("metadata", {})["updated_at"] = time.time()
        atomic_json_write(self.file_path, data)

    def append_entry(
        self,
        entry_type: str,
        content: str,
        context_ref: str | None = None,
        tags: list[str] | None = None,
        relevance_hint: float = 1.0,
    ) -> str:
        """Append an entry and return its ID."""
        entry = MemoryEntry.create(
            self.session_id,
            entry_type,
            content,
            context_ref=context_ref,
            tags=tags,
            relevance_hint=relevance_hint,
        )
        with self._lock:
            data = self._read_state()
            data.setdefault("entries", []).append(entry.to_json())
            data.setdefault("metadata", {})["entry_count"] = len(data["entries"])
            self._write_state(data)
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
        """Attach a tag to an entry."""
        with self._lock:
            data = self._read_state()
            for entry in data.get("entries", []):
                if entry.get("entry_id") == entry_id:
                    entry.setdefault("tags", [])
                    if tag not in entry["tags"]:
                        entry["tags"].append(tag)
                    self._write_state(data)
                    return True
        return False

    def get_size_estimate(self) -> tuple[int, int]:
        """Return estimated tokens and bytes for the hot file."""
        entries = self.get_entries(limit=None, include_compacted=True)
        content = "\n".join(e.get("content", "") for e in entries)
        size = self.file_path.stat().st_size if self.file_path.exists() else 0
        return estimate_tokens(content, overhead=1000), size
