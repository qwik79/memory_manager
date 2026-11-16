"""Tier 3: rebuildable cold index for long-term retrieval.

Supports two retrieval modes:

* **Lexical** (always available, zero dependencies) — cosine similarity over
  term-frequency vectors.  Good for keyword-heavy queries.
* **Semantic** (opt-in via ``embed_fn``) — cosine similarity over dense
  embedding vectors from any embedding model.  When an embedding function is
  provided, retrieval uses a blended score: ``0.7 × semantic + 0.3 × lexical``.
  Chunks indexed before the embedding function was configured are scored with
  pure lexical similarity — backward compatible.
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from .models import RetrievalResult, new_id
from .storage import atomic_json_write, default_base_path, read_json, safe_id

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], Sequence[float]]

# ---------------------------------------------------------------------------
# Cosine helper — works for both sparse (Counter) and dense (list) vectors.
# ---------------------------------------------------------------------------

def _cosine_sparse(left: Counter[str], right: Counter[str]) -> float:
    """Cosine similarity over sparse term-frequency vectors."""
    if not left or not right:
        return 0.0
    dot = sum(left[t] * right.get(t, 0) for t in left)
    left_norm = math.sqrt(sum(v * v for v in left.values()))
    right_norm = math.sqrt(sum(v * v for v in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _cosine_dense(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity over dense embedding vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


class ColdStoreManager:
    """File-backed cold index with hybrid lexical + semantic retrieval.

    When constructed with an ``embed_fn``, new chunks are indexed with both
    term-frequency bags and dense embedding vectors.  Retrieval blends the
    two signals (``0.7 × semantic + 0.3 × lexical``) for significantly better
    recall on paraphrased or conceptually related queries.

    Without ``embed_fn``, behaviour is identical to the pure-lexical scorer
    that shipped in v0.1.0 — zero dependencies, always available.
    """

    def __init__(
        self,
        base_path: str | Path | None = None,
        security: Any = None,
        embed_fn: EmbedFn | None = None,
    ):
        self.base_path = Path(base_path) if base_path else default_base_path("llm_cold")
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._security = security
        self._embed_fn = embed_fn

    def _index_dir(self, session_id: str) -> Path:
        path = self.base_path / "sessions" / safe_id(session_id) / "chunks"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def index_section(
        self,
        session_id: str,
        section_name: str,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        source_section_id: str | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> list[str]:
        """Chunk and index a section; return created chunk IDs."""
        chunks = self._split_words(content, chunk_size, chunk_overlap)
        chunk_ids: list[str] = []
        for index, chunk in enumerate(chunks):
            chunk_id = new_id("cold")
            data: dict[str, Any] = {
                "schema_version": 2,
                "chunk_id": chunk_id,
                "session_id": safe_id(session_id),
                "section_name": section_name,
                "source_section_id": source_section_id,
                "content": chunk,
                "tags": tags or [],
                "metadata": metadata or {},
                "created_at": time.time(),
                "chunk_index": index,
                "term_frequencies": dict(Counter(self._tokens(chunk))),
            }
            # Dense embedding — opt-in.
            if self._embed_fn is not None:
                try:
                    data["embedding"] = list(self._embed_fn(chunk))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Embedding failed for chunk %s: %s", chunk_id, exc)
            atomic_json_write(
                self._index_dir(session_id) / f"{chunk_id}.json",
                data,
                security=self._security,
            )
            chunk_ids.append(chunk_id)
        return chunk_ids

    def index_warm_section(self, section: dict[str, Any], tags: list[str] | None = None) -> list[str]:
        """Index a warm section record into cold storage."""
        metadata = dict(section.get("metadata", {}))
        metadata["source_entry_ids"] = section.get("source_entry_ids", [])
        return self.index_section(
            session_id=section.get("session_id", "default"),
            section_name=section.get("section_name", "warm_section"),
            content=section.get("content", ""),
            tags=tags or metadata.get("tags", []),
            metadata=metadata,
            source_section_id=section.get("section_id"),
        )

    def delete_chunks_for_source_section(self, session_id: str, source_section_id: str | None) -> int:
        """Delete indexed chunks that came from a specific warm section."""
        if not source_section_id:
            return 0
        deleted = 0
        for path in self._index_dir(session_id).glob("*.json"):
            try:
                data = read_json(path, security=self._security)
            except Exception:
                logger.warning("Skipping unreadable/tampered chunk %s during delete.", path.name)
                continue
            if data.get("source_section_id") == source_section_id:
                path.unlink()
                deleted += 1
        return deleted

    def retrieve(
        self,
        session_id: str,
        query: str,
        k: int = 10,
        tags_filter: list[str] | None = None,
        min_relevance: float = 0.1,
    ) -> list[RetrievalResult]:
        """Retrieve cold chunks using hybrid semantic + lexical scoring.

        When an ``embed_fn`` is configured, the query is embedded and scored
        against chunk embeddings.  Chunks without embeddings (legacy or
        embedding-failed) fall back to pure lexical scoring.

        Chunks that fail signature verification are silently skipped rather than
        crashing retrieval — a tampered chunk should not surface to the LLM.
        """
        query_terms = Counter(self._tokens(query))
        if not query_terms and self._embed_fn is None:
            return []

        # Compute query embedding once for the entire retrieval pass.
        query_embedding: Sequence[float] | None = None
        if self._embed_fn is not None:
            try:
                query_embedding = self._embed_fn(query)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Query embedding failed, falling back to lexical: %s", exc)

        results: list[RetrievalResult] = []
        for path in self._index_dir(session_id).glob("*.json"):
            try:
                data = read_json(path, security=self._security)
            except Exception:
                logger.warning(
                    "Cold chunk %s failed verification — skipping to prevent prompt injection.",
                    path.name,
                )
                continue
            tags = data.get("tags", [])
            if tags_filter and tags_filter != ["all"] and not any(tag in tags for tag in tags_filter):
                continue

            score = self._score_chunk(query_terms, query_embedding, data)
            if score >= min_relevance:
                results.append(
                    RetrievalResult(
                        source=f"cold://{data.get('session_id')}/{data.get('chunk_id')}",
                        identifier=data.get("chunk_id"),
                        content=data.get("content", ""),
                        metadata={
                            **data.get("metadata", {}),
                            "tags": tags,
                            "section_name": data.get("section_name"),
                            "source_section_id": data.get("source_section_id"),
                        },
                        relevance=score,
                    )
                )
        return sorted(results, key=lambda r: r.relevance, reverse=True)[:k]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_chunk(
        self,
        query_terms: Counter[str],
        query_embedding: Sequence[float] | None,
        chunk: dict[str, Any],
    ) -> float:
        """Compute a relevance score for a single chunk.

        Returns a blended score when both lexical and semantic signals are
        available, or a single-signal score when only one is.
        """
        lexical = _cosine_sparse(query_terms, Counter(chunk.get("term_frequencies", {})))
        chunk_embedding = chunk.get("embedding")

        if query_embedding is not None and chunk_embedding:
            semantic = _cosine_dense(query_embedding, chunk_embedding)
            # Blend: semantic is the primary signal, lexical provides keyword
            # precision for exact matches.
            return 0.7 * semantic + 0.3 * lexical

        return lexical

    # ------------------------------------------------------------------
    # Text processing
    # ------------------------------------------------------------------

    def _split_words(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        words = text.split()
        if not words:
            return []
        chunk_size = max(1, chunk_size)
        overlap = min(max(0, overlap), chunk_size - 1)
        step = chunk_size - overlap
        return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]

    def _tokens(self, text: str) -> list[str]:
        return [token.strip(".,!?;:()[]{}\"'`).").lower() for token in text.split() if token.strip()]
