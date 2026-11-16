"""ChromaDB-backed cold store — drop-in replacement for ColdStoreManager.

Implements the identical public interface as
:class:`~memory_manager.cold_store.ColdStoreManager` so it can be swapped in
at runtime::

    from memory_manager.cold_store_chroma import ChromaColdStore
    from memory_manager import TieredMemoryManager, make_ollama_embed_fn

    chroma = ChromaColdStore(embed_fn=make_ollama_embed_fn())
    mgr = TieredMemoryManager("my_session")
    mgr.cold_store = chroma  # swap after init

Install dependency::

    pip install "memory_manager[chroma]"
    # or: pip install chromadb>=0.5

The store uses ChromaDB's persistent client, writing to disk under
*base_path* / "vectordb".  When ``embed_fn`` is provided, vectors are
generated client-side (e.g. via Ollama) and stored in the collection.
When ``embed_fn`` is ``None``, ChromaDB uses its built-in sentence-transformer
model for embedding.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

from .models import RetrievalResult, new_id
from .storage import default_base_path, safe_id

logger = logging.getLogger(__name__)


class ChromaColdStore:
    """Semantic cold store backed by ChromaDB.

    Parameters
    ----------
    base_path:
        Directory for the persistent ChromaDB database.  Defaults to
        ``~/.local/share/memory_manager/llm_cold_chroma``.
    embed_fn:
        Optional single-string embedding function.  When provided, embeddings
        are generated client-side (e.g. via :func:`~memory_manager.make_ollama_embed_fn`)
        and passed to ChromaDB directly, giving full control over the model.
        When ``None``, ChromaDB's built-in embedding function is used.
    security:
        Accepted for interface compatibility with ``ColdStoreManager``.
        ChromaDB manages its own storage — this parameter is unused.
    """

    def __init__(
        self,
        base_path: str | Path | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
        security: Any = None,  # accepted for interface compat, not used
    ) -> None:
        import chromadb  # noqa: PLC0415 — optional dependency

        path = Path(base_path) if base_path else default_base_path("llm_cold_chroma")
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path / "vectordb"))
        self._embed_fn = embed_fn
        self._collections: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_collection(self, session_id: str) -> Any:
        sid = safe_id(session_id)
        if sid not in self._collections:
            self._collections[sid] = self._client.get_or_create_collection(
                name=sid,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[sid]

    def _split_words(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        words = text.split()
        if not words:
            return []
        chunk_size = max(1, chunk_size)
        overlap = min(max(0, overlap), chunk_size - 1)
        step = chunk_size - overlap
        return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]

    @staticmethod
    def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
        """ChromaDB metadata values must be str | int | float | bool."""
        clean: dict[str, Any] = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, (list, dict)):
                clean[k] = str(v)
        return clean

    # ------------------------------------------------------------------
    # Public API — matches ColdStoreManager exactly
    # ------------------------------------------------------------------

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
        """Chunk, embed, and index a text section. Returns created chunk IDs."""
        chunks = self._split_words(content, chunk_size, chunk_overlap)
        if not chunks:
            return []

        col = self._get_collection(session_id)
        chunk_ids: list[str] = []

        for i, chunk in enumerate(chunks):
            cid = new_id("cold")
            meta = self._sanitize_metadata({
                **(metadata or {}),
                "section_name": section_name,
                "source_section_id": source_section_id or "",
                "chunk_index": i,
                "tags": ",".join(tags or []),
                "session_id": safe_id(session_id),
                "created_at": time.time(),
            })

            upsert_kwargs: dict[str, Any] = {
                "ids": [cid],
                "documents": [chunk],
                "metadatas": [meta],
            }
            if self._embed_fn is not None:
                try:
                    upsert_kwargs["embeddings"] = [self._embed_fn(chunk)]
                except Exception as exc:
                    logger.warning("Embedding failed for chunk %s: %s — skipping.", cid, exc)
                    continue

            try:
                col.upsert(**upsert_kwargs)
                chunk_ids.append(cid)
            except Exception as exc:
                logger.warning("ChromaDB upsert failed for chunk %s: %s", cid, exc)

        return chunk_ids

    def index_warm_section(
        self,
        section: dict[str, Any],
        tags: list[str] | None = None,
    ) -> list[str]:
        """Index a warm section dict into cold storage."""
        metadata = dict(section.get("metadata", {}))
        metadata["source_entry_ids"] = str(section.get("source_entry_ids", []))
        return self.index_section(
            session_id=section.get("session_id", "default"),
            section_name=section.get("section_name", "warm_section"),
            content=section.get("content", ""),
            tags=tags or [],
            metadata=metadata,
            source_section_id=section.get("section_id"),
        )

    def delete_chunks_for_source_section(
        self,
        session_id: str,
        source_section_id: str | None,
    ) -> int:
        """Delete all chunks derived from the given warm section. Returns count deleted."""
        if not source_section_id:
            return 0
        col = self._get_collection(session_id)
        try:
            results = col.get(where={"source_section_id": source_section_id})
            ids: list[str] = results.get("ids", [])
            if ids:
                col.delete(ids=ids)
            return len(ids)
        except Exception as exc:
            logger.warning("ChromaDB delete failed for section %s: %s", source_section_id, exc)
            return 0

    def retrieve(
        self,
        session_id: str,
        query: str,
        k: int = 10,
        tags_filter: list[str] | None = None,
        min_relevance: float = 0.1,
    ) -> list[RetrievalResult]:
        """Retrieve semantically similar chunks for a query string."""
        col = self._get_collection(session_id)

        query_kwargs: dict[str, Any] = {"n_results": min(k, max(1, col.count()))}
        if self._embed_fn is not None:
            try:
                query_kwargs["query_embeddings"] = [self._embed_fn(query)]
            except Exception as exc:
                logger.warning("Query embedding failed: %s — returning empty.", exc)
                return []
        else:
            query_kwargs["query_texts"] = [query]

        if tags_filter and tags_filter != ["all"]:
            # ChromaDB $in requires at least one value
            query_kwargs["where"] = {"tags": {"$in": [f",{t}," for t in tags_filter]}}

        try:
            raw = col.query(**query_kwargs)
        except Exception as exc:
            logger.warning("ChromaDB query failed: %s", exc)
            return []

        docs: list[str] = (raw.get("documents") or [[]])[0]
        metas: list[dict] = (raw.get("metadatas") or [[]])[0]
        # ChromaDB returns cosine distances [0, 2]; convert to similarity [0, 1].
        dists: list[float] = (raw.get("distances") or [[]])[0]

        results: list[RetrievalResult] = []
        for doc, meta, dist in zip(docs, metas, dists):
            relevance = max(0.0, 1.0 - dist / 2.0)
            if relevance >= min_relevance:
                results.append(
                    RetrievalResult(
                        source=f"cold://{safe_id(session_id)}/chroma",
                        content=doc,
                        relevance=relevance,
                        metadata=meta or {},
                    )
                )

        return results
