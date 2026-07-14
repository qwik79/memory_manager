"""Tier 3: rebuildable cold index for long-term retrieval."""

from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .models import RetrievalResult, new_id
from .storage import atomic_json_write, default_base_path, read_json, safe_id


class ColdStoreManager:
    """File-backed cold index with deterministic lexical vector scoring.

    This keeps the cold tier dependency-free. A future Chroma/Qdrant adapter can
    implement the same public methods while using warm sections as source truth.
    """

    def __init__(self, base_path: str | Path | None = None):
        self.base_path = Path(base_path) if base_path else default_base_path("llm_cold")
        self.base_path.mkdir(parents=True, exist_ok=True)

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
            data = {
                "schema_version": 1,
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
            atomic_json_write(self._index_dir(session_id) / f"{chunk_id}.json", data)
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
            data = read_json(path)
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
        """Retrieve cold chunks by cosine similarity over lexical vectors."""
        query_terms = Counter(self._tokens(query))
        if not query_terms:
            return []
        results: list[RetrievalResult] = []
        for path in self._index_dir(session_id).glob("*.json"):
            data = read_json(path)
            tags = data.get("tags", [])
            if tags_filter and tags_filter != ["all"] and not any(tag in tags for tag in tags_filter):
                continue
            score = self._cosine(query_terms, Counter(data.get("term_frequencies", {})))
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

    def _cosine(self, left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0.0
        dot = sum(left[t] * right.get(t, 0) for t in left)
        left_norm = math.sqrt(sum(v * v for v in left.values()))
        right_norm = math.sqrt(sum(v * v for v in right.values()))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
