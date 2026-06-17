"""Tier 2: durable warm store for compacted memory sections."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from .models import RetrievalResult, WarmSection, new_id
from .storage import atomic_json_write, default_base_path, read_json, safe_id


class WarmStoreManager:
    """Disk-backed canonical store for summarized memory sections."""

    def __init__(self, base_path: str | Path | None = None):
        self.base_path = Path(base_path) if base_path else default_base_path("llm_warm")
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _session_dir(self, session_id: str) -> Path:
        return self.base_path / "sessions" / safe_id(session_id)

    def _section_dir(self, session_id: str) -> Path:
        path = self._session_dir(session_id) / "sections"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def store_section(
        self,
        session_id: str,
        section_name: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        source_entry_ids: list[str] | None = None,
    ) -> str:
        """Store a compacted section and return its section ID."""
        section = WarmSection(
            section_id=new_id("warm"),
            session_id=safe_id(session_id),
            section_name=section_name,
            content=content,
            metadata=metadata or {},
            source_entry_ids=source_entry_ids or [],
        )
        path = self._section_dir(session_id) / f"{section.section_id}.json"
        with self._lock:
            atomic_json_write(path, section.to_json())
        return section.section_id

    def get_section(self, session_id: str, section_id: str) -> dict[str, Any] | None:
        """Load one warm section."""
        path = self._section_dir(session_id) / f"{safe_id(section_id)}.json"
        if not path.exists():
            return None
        return read_json(path)

    def list_sections(self, session_id: str, include_indexed: bool = True) -> list[dict[str, Any]]:
        """List warm sections for a session."""
        section_dir = self._section_dir(session_id)
        sections = [read_json(path) for path in sorted(section_dir.glob("*.json"))]
        if not include_indexed:
            sections = [s for s in sections if not s.get("indexed_at")]
        return sections

    def mark_indexed(self, session_id: str, section_id: str) -> bool:
        """Mark a warm section as indexed by cold storage."""
        path = self._section_dir(session_id) / f"{safe_id(section_id)}.json"
        if not path.exists():
            return False
        with self._lock:
            data = read_json(path)
            data["indexed_at"] = time.time()
            atomic_json_write(path, data)
        return True

    def retrieve_sections(
        self,
        session_id: str,
        query: str = "",
        k: int = 5,
        relevance_threshold: float = 0.3,
    ) -> list[RetrievalResult]:
        """Retrieve warm sections by recency plus lexical overlap."""
        results: list[RetrievalResult] = []
        for section in self.list_sections(session_id):
            score = self._score(query, section)
            if score >= relevance_threshold:
                results.append(
                    RetrievalResult(
                        source=f"warm://{section.get('session_id')}/{section.get('section_id')}",
                        identifier=section.get("section_id"),
                        content=section.get("content", ""),
                        metadata=section.get("metadata", {}),
                        relevance=score,
                    )
                )
        return sorted(results, key=lambda r: r.relevance, reverse=True)[:k]

    def _score(self, query: str, section: dict[str, Any]) -> float:
        age_seconds = max(0.0, time.time() - section.get("created_at", time.time()))
        recency = max(0.0, min(1.0, 1.0 / (1.0 + age_seconds / 86_400)))
        if not query:
            return recency * 0.7
        q_words = set(query.lower().split())
        content = " ".join(
            [section.get("section_name", ""), section.get("content", ""), str(section.get("metadata", {}))]
        )
        c_words = set(content.lower().split())
        lexical = len(q_words & c_words) / max(len(q_words), 1)
        return min(1.0, 0.4 * recency + 0.6 * lexical)
