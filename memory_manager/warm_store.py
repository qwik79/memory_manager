"""Tier 2: durable warm store for compacted memory sections."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .backends import MEMORY_ROOT, DocStore, FileDocStore
from .models import RetrievalResult, WarmSection, new_id
from .storage import default_base_path, safe_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BM25-inspired scoring parameters
# ---------------------------------------------------------------------------
_BM25_K1 = 1.5   # term frequency saturation constant
_BM25_B = 0.75   # document length normalization (0 = none, 1 = full)
_RECENCY_HALF_LIFE_HOURS = 48.0  # warm tier half-life (exponential decay)
_RECENCY_HALF_LIFE_SECONDS = _RECENCY_HALF_LIFE_HOURS * 3600


class WarmStoreManager:
    """Disk-backed canonical store for summarized memory sections.

    Retrieval uses a BM25-inspired scorer with recency decay and metadata
    bonuses.  The warm store is the *canonical* representation of compacted
    memory — cold storage is a rebuildable index derived from warm.
    """

    def __init__(
        self,
        base_path: str | Path | None = None,
        security: Any = None,
        store: DocStore | None = None,
    ):
        self.store: DocStore = store if store is not None else FileDocStore(security)
        if base_path:
            self.base_path = Path(base_path)
        else:
            self.base_path = default_base_path("llm_warm") if self.store.persistent else MEMORY_ROOT / "warm"
        self.store.ensure_dir(self.base_path)
        self._lock = threading.RLock()
        self._security = security

    @property
    def persistent(self) -> bool:
        return bool(self.store.persistent)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def session_dir(self, session_id: str) -> Path:
        """Return the root directory for a session's warm data.

        Public so the orchestrator can resolve session file paths without
        duplicating path logic.
        """
        return self.base_path / "sessions" / safe_id(session_id)

    def _section_dir(self, session_id: str) -> Path:
        path = self.session_dir(session_id) / "sections"
        self.store.ensure_dir(path)
        return path

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store_section(
        self,
        session_id: str,
        section_name: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        source_entry_ids: list[str] | None = None,
        raw_content: str = "",
    ) -> str:
        """Store a compacted section and return its section ID.

        *content* is the distilled text shown in prompts; *raw_content* keeps the
        full source text so the cold index can be rebuilt without loss.
        """
        section = WarmSection(
            section_id=new_id("warm"),
            session_id=safe_id(session_id),
            section_name=section_name,
            content=content,
            metadata=metadata or {},
            source_entry_ids=source_entry_ids or [],
            raw_content=raw_content,
        )
        path = self._section_dir(session_id) / f"{section.section_id}.json"
        with self._lock:
            self.store.write(path, section.to_json())
        return section.section_id

    def put_section(self, session_id: str, section: dict[str, Any]) -> None:
        """Write an existing section record as-is (used when promoting a session)."""
        with self._lock:
            self.store.write(self._section_dir(session_id) / f"{safe_id(section['section_id'])}.json", section)

    def clear(self, session_id: str) -> int:
        """Delete every document (sections and session state) for a session."""
        deleted = 0
        with self._lock:
            for path in self.store.walk(self.session_dir(session_id)):
                deleted += int(self.store.delete(path))
        return deleted

    def get_section(self, session_id: str, section_id: str) -> dict[str, Any] | None:
        """Load one warm section."""
        path = self._section_dir(session_id) / f"{safe_id(section_id)}.json"
        return self.store.read(path)

    def list_sections(self, session_id: str, include_indexed: bool = True) -> list[dict[str, Any]]:
        """List warm sections for a session."""
        section_dir = self._section_dir(session_id)
        sections = []
        for path in self.store.list(section_dir):
            try:
                sections.append(self.store.read(path) or {})
            except Exception:
                logger.warning(
                    "Skipping warm section %s — failed verification or read.", path.name
                )
        if not include_indexed:
            sections = [s for s in sections if not s.get("indexed_at")]
        return sections

    def mark_indexed(self, session_id: str, section_id: str) -> bool:
        """Mark a warm section as indexed by cold storage."""
        path = self._section_dir(session_id) / f"{safe_id(section_id)}.json"
        with self._lock:
            data = self.store.read(path)
            if data is None:
                return False
            data["indexed_at"] = time.time()
            self.store.write(path, data)
        return True

    def gc(
        self,
        session_id: str,
        max_age_days: float = 30.0,
    ) -> int:
        """Purge warm sections older than *max_age_days* that have been indexed.

        Only sections with an ``indexed_at`` timestamp are eligible — these
        have already been cascaded to cold storage, so the warm copy is
        redundant.  Returns the number of sections deleted.
        """
        cutoff = time.time() - max_age_days * 86_400
        deleted = 0
        for path in self.store.list(self._section_dir(session_id)):
            try:
                data = self.store.read(path) or {}
            except Exception:
                continue
            indexed_at = data.get("indexed_at")
            created_at = data.get("created_at", time.time())
            if indexed_at and created_at < cutoff:
                self.store.delete(path)
                deleted += 1
                logger.debug(
                    "GC: purged warm section %s (age=%.1f days).",
                    data.get("section_id", path.stem),
                    (time.time() - created_at) / 86_400,
                )
        if deleted:
            logger.info("Warm GC: purged %d old indexed section(s) for session '%s'.", deleted, session_id)
        return deleted

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_sections(
        self,
        session_id: str,
        query: str = "",
        k: int = 5,
        relevance_threshold: float = 0.15,
    ) -> list[RetrievalResult]:
        """Retrieve warm sections using BM25-inspired scoring with recency decay."""
        sections = self.list_sections(session_id)
        if not sections:
            return []

        # Pre-compute corpus statistics for BM25 IDF.
        all_contents = [self._section_text(s) for s in sections]
        avg_doc_len = sum(len(c.split()) for c in all_contents) / max(len(all_contents), 1)
        # IDF: count how many sections contain each query term.
        q_tokens = self._tokenize(query)
        idf: dict[str, float] = {}
        if q_tokens:
            n = len(sections)
            for term in set(q_tokens):
                df = sum(1 for c in all_contents if term in c.lower())
                idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

        results: list[RetrievalResult] = []
        for section, content in zip(sections, all_contents):
            score = self._score(query, section, content, q_tokens, idf, avg_doc_len)
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

    # ------------------------------------------------------------------
    # BM25-inspired scoring
    # ------------------------------------------------------------------

    def _score(
        self,
        query: str,
        section: dict[str, Any],
        content: str,
        q_tokens: list[str],
        idf: dict[str, float],
        avg_doc_len: float,
    ) -> float:
        """Compute a composite relevance score in [0.0, 1.0].

        Signals:
        1. **BM25 term relevance** (0.50 weight) — IDF-weighted term frequency
           with saturation and document length normalization.
        2. **Recency** (0.30 weight) — exponential decay with 48-hour half-life.
        3. **Metadata bonus** (0.20 weight) — decisions, facts, pin tier,
           source density.
        """
        # --- Recency (exponential decay) ---------------------------------
        age_seconds = max(0.0, time.time() - section.get("created_at", time.time()))
        recency = math.exp(-0.693 * age_seconds / _RECENCY_HALF_LIFE_SECONDS)

        # --- BM25 term relevance -----------------------------------------
        bm25 = 0.0
        if q_tokens and idf:
            doc_tokens = self._tokenize(content)
            doc_len = len(doc_tokens)
            tf_counter = Counter(doc_tokens)
            norm_factor = 1.0 - _BM25_B + _BM25_B * (doc_len / max(avg_doc_len, 1.0))
            raw = 0.0
            for term in set(q_tokens):
                tf = tf_counter.get(term, 0)
                term_score = idf.get(term, 0.0) * (tf * (_BM25_K1 + 1.0)) / (tf + _BM25_K1 * norm_factor)
                raw += term_score
            # Normalize to [0, 1] via sigmoid squashing (prevents unbounded growth).
            bm25 = raw / (raw + 3.0) if raw > 0 else 0.0

        # --- Metadata bonus -----------------------------------------------
        meta = section.get("metadata", {})
        bonus = 0.0
        # Sections distilled from decisions are higher value.
        if meta.get("decisions"):
            bonus += 0.35
        # Sections with extracted facts are information-dense.
        if meta.get("facts"):
            bonus += 0.20
        # Pin-tier sections were explicitly marked important by the distiller.
        if str(meta.get("tier", "")).lower() == "pin":
            bonus += 0.30
        # Source density: sections compacted from more entries are richer.
        source_count = len(section.get("source_entry_ids", []))
        if source_count >= 5:
            bonus += 0.15
        elif source_count >= 2:
            bonus += 0.05
        bonus = min(bonus, 1.0)

        # --- No-query mode: recency + metadata only ----------------------
        if not query:
            return min(1.0, 0.6 * recency + 0.4 * bonus)

        # --- Composite ----------------------------------------------------
        return min(1.0, 0.50 * bm25 + 0.30 * recency + 0.20 * bonus)

    # ------------------------------------------------------------------
    # Text processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _section_text(section: dict[str, Any]) -> str:
        """Concatenate section name, content, and metadata for scoring."""
        parts = [
            section.get("section_name", ""),
            section.get("content", ""),
        ]
        meta = section.get("metadata", {})
        if meta.get("decisions"):
            parts.extend(str(d) for d in meta["decisions"])
        return " ".join(parts)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase word tokenization with punctuation stripping."""
        if not text:
            return []
        return [
            t.strip(".,!?;:()[]{}\"'`).").lower()
            for t in text.split()
            if t.strip()
        ]
