"""Main orchestrator for the tiered memory manager."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .active_context import ActiveContext, ContextItem
from .cold_store import ColdStoreManager
from .eviction import EvictionManager
from .hot_buffer import HotBufferManager
from .models import MemorySession, RetrievalResult
from .summarizer import HeuristicSummarizer, Summarizer
from .warm_store import WarmStoreManager


class TieredMemoryManager:
    """Coordinates active, hot, warm, and cold memory tiers."""

    def __init__(
        self,
        session_id: str,
        project_name: str = "",
        pinned_summary: str = "",
        base_paths: dict[str, str | Path] | None = None,
        summarizer: Summarizer | None = None,
    ):
        self.session = MemorySession(session_id=session_id, project_name=project_name, pinned_summary=pinned_summary)
        paths = base_paths or {}
        self.active_context = ActiveContext(self.session)
        self.hot_buffer = HotBufferManager(session_id, paths.get("hot"))
        self.warm_store = WarmStoreManager(paths.get("warm"))
        self.cold_store = ColdStoreManager(paths.get("cold"))
        self.eviction_manager = EvictionManager(self.hot_buffer)
        self.summarizer = summarizer or HeuristicSummarizer()

    @property
    def session_id(self) -> str:
        return self.session.session_id

    def add_user_message(self, content: str, context_ref: str | None = None, tags: list[str] | None = None) -> str:
        entry_id = self.hot_buffer.append_entry("user", content, context_ref=context_ref, tags=tags)
        self.eviction_manager.record_reference(entry_id)
        return entry_id

    def add_assistant_message(self, content: str, tags: list[str] | None = None, context_ref: str | None = None) -> str:
        entry_id = self.hot_buffer.append_entry("assistant", content, context_ref=context_ref, tags=tags)
        self.eviction_manager.record_reference(entry_id)
        return entry_id

    def add_decision(self, description: str, decision_type: str = "general") -> None:
        self.session.active_decisions.append({"type": decision_type, "description": description})

    def compact_hot_to_warm(self, section_name: str = "session_summary", limit: int = 40) -> str | None:
        """Summarize recent un-compacted hot entries into a warm section."""
        entries = self.hot_buffer.get_entries(limit=limit)
        if not entries:
            return None
        summary = self.summarizer.summarize_entries(entries, {"current_task": self.session.current_task})
        content = summary.get("summary", "")
        if not content:
            return None
        source_ids = [entry.get("entry_id") for entry in entries if entry.get("entry_id")]
        section_id = self.warm_store.store_section(
            self.session_id,
            section_name,
            content,
            metadata={
                "summary_confidence": summary.get("confidence"),
                "decisions": summary.get("decisions", []),
                "facts": summary.get("facts", {}),
            },
            source_entry_ids=source_ids,
        )
        self.hot_buffer.mark_compacted(source_ids)
        return section_id

    def index_warm_to_cold(self, include_indexed: bool = False) -> list[str]:
        """Index warm sections into the cold tier."""
        created: list[str] = []
        for section in self.warm_store.list_sections(self.session_id, include_indexed=include_indexed):
            chunk_ids = self.cold_store.index_warm_section(section)
            if chunk_ids:
                created.extend(chunk_ids)
                self.warm_store.mark_indexed(self.session_id, section["section_id"])
        return created

    def refresh_context(self, query: str, k: int = 8) -> list[RetrievalResult]:
        """Retrieve relevant warm and cold memories for a query."""
        warm = self.warm_store.retrieve_sections(self.session_id, query=query, k=max(1, k // 2))
        cold = self.cold_store.retrieve(self.session_id, query=query, k=k)
        return sorted([*warm, *cold], key=lambda r: r.relevance, reverse=True)[:k]

    def build_context_for_llm(self, query: str = "", max_tokens: int = 32_000) -> str:
        """Build a prompt-ready context block under a token budget."""
        items: list[ContextItem] = []
        for entry in self.hot_buffer.get_entries(limit=20, relevance_min=0.0):
            priority = 5 + (5 if "high-priority" in entry.get("tags", []) else 0)
            items.append(ContextItem(content=entry.get("content", ""), priority=priority))
        if query:
            for result in self.refresh_context(query, k=6):
                items.append(ContextItem(content=result.content, priority=int(result.relevance * 10)))
        return self.active_context.build_context(items, max_tokens=max_tokens)

    def maybe_offload(self, current_task: str = "") -> dict[str, Any]:
        """Evaluate eviction and compact selected entries when needed."""
        offload_ids, pin_ids = self.eviction_manager.evaluate_for_eviction(current_task=current_task)
        for entry_id in pin_ids:
            self.hot_buffer.tag_entry(entry_id, "pin")
        section_id = None
        if offload_ids:
            entries = [e for e in self.hot_buffer.get_entries(limit=None) if e.get("entry_id") in set(offload_ids)]
            summary = self.summarizer.summarize_entries(entries, {"current_task": current_task})
            section_id = self.warm_store.store_section(
                self.session_id,
                "evicted_hot_entries",
                summary.get("summary", ""),
                metadata={"evicted_entry_count": len(entries)},
                source_entry_ids=offload_ids,
            )
            self.hot_buffer.mark_compacted(offload_ids)
        return {"offloaded": offload_ids, "pinned": pin_ids, "warm_section_id": section_id}
