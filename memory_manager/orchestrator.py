"""Main orchestrator for the tiered memory manager."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .active_context import ActiveContext, ContextItem
from .cold_store import ColdStoreManager
from .eviction import EvictionManager
from .hot_buffer import HotBufferManager
from .models import MemorySession, RetrievalResult
from .storage import safe_id
from .summarizer import HeuristicSummarizer, Summarizer
from .trigger import EmbedFn, OffloadTrigger, TriggerConfig
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
        trigger_config: TriggerConfig | None = None,
        embed_fn: EmbedFn | None = None,
    ):
        normalized_session_id = safe_id(session_id)
        self.session = MemorySession(
            session_id=normalized_session_id,
            project_name=project_name,
            pinned_summary=pinned_summary,
        )
        paths = base_paths or {}
        self.active_context = ActiveContext(self.session)
        self.hot_buffer = HotBufferManager(normalized_session_id, paths.get("hot"))
        self.warm_store = WarmStoreManager(paths.get("warm"))
        self.cold_store = ColdStoreManager(paths.get("cold"))
        self.eviction_manager = EvictionManager(self.hot_buffer)
        self.summarizer = summarizer or HeuristicSummarizer()
        self.offload_trigger = OffloadTrigger(trigger_config)
        self.embed_fn = embed_fn
        self.pinned_embedding = embed_fn(pinned_summary) if embed_fn and pinned_summary else None
        self.last_offload_result: dict[str, Any] | None = None

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
        self.last_offload_result = self.evaluate_offload()
        return entry_id

    def add_decision(self, description: str, decision_type: str = "general") -> None:
        self.session.active_decisions.append({"type": decision_type, "description": description})

    def compact_hot_to_warm(self, section_name: str = "session_summary", limit: int = 40) -> str | None:
        """Summarize recent un-compacted hot entries into a warm section."""
        entries = self.hot_buffer.get_entries(limit=limit)
        if not entries:
            return None
        summary = self.summarizer.summarize_entries(entries, self._summary_context())
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
                "next_steps": summary.get("next_steps", []),
                "tier": summary.get("tier", "extend"),
                "notes_for_core": summary.get("notes_for_core", ""),
            },
            source_entry_ids=source_ids,
        )
        self.hot_buffer.mark_compacted(source_ids)
        self._apply_distillation(summary)
        return section_id

    def evaluate_offload(self) -> dict[str, Any]:
        """Evaluate the composite trigger and compact automatically when it fires."""
        entries = self.hot_buffer.get_entries(limit=None)
        result = self.offload_trigger.evaluate(entries, self.pinned_embedding, self.embed_fn)
        result["warm_section_id"] = self.compact_hot_to_warm() if result["should_offload"] else None
        return result

    def index_warm_to_cold(self, include_indexed: bool = False) -> list[str]:
        """Index warm sections into the cold tier."""
        created: list[str] = []
        for section in self.warm_store.list_sections(self.session_id, include_indexed=include_indexed):
            if include_indexed and section.get("indexed_at"):
                self.cold_store.delete_chunks_for_source_section(self.session_id, section.get("section_id"))
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
            items.append(ContextItem(content=self._format_hot_entry(entry), priority=priority))
        if query:
            for result in self.refresh_context(query, k=6):
                items.append(ContextItem(content=self._format_retrieval_result(result), priority=int(result.relevance * 10)))
        return self.active_context.build_context(items, max_tokens=max_tokens)

    def maybe_offload(self, current_task: str = "") -> dict[str, Any]:
        """Evaluate eviction and compact selected entries when needed."""
        offload_ids, pin_ids = self.eviction_manager.evaluate_for_eviction(current_task=current_task)
        for entry_id in pin_ids:
            self.hot_buffer.tag_entry(entry_id, "pin")
        section_id = None
        if offload_ids:
            entries = [e for e in self.hot_buffer.get_entries(limit=None) if e.get("entry_id") in set(offload_ids)]
            summary = self.summarizer.summarize_entries(entries, self._summary_context())
            section_id = self.warm_store.store_section(
                self.session_id,
                "evicted_hot_entries",
                summary.get("summary", ""),
                metadata={"evicted_entry_count": len(entries)},
                source_entry_ids=offload_ids,
            )
            self.hot_buffer.mark_compacted(offload_ids)
        return {"offloaded": offload_ids, "pinned": pin_ids, "warm_section_id": section_id}

    def _summary_context(self) -> dict[str, Any]:
        return {"current_task": self.session.current_task, "pinned_summary": self.session.pinned_summary}

    def _apply_distillation(self, summary: dict[str, Any]) -> None:
        distilled = str(summary.get("summary", "")).strip()
        if distilled:
            self.session.pinned_summary = distilled
            if self.embed_fn:
                self.pinned_embedding = self.embed_fn(distilled)
        for decision in summary.get("decisions", []):
            if decision and not any(item.get("description") == decision for item in self.session.active_decisions):
                self.add_decision(str(decision), "distilled")
        self.session.memory_notes = str(summary.get("notes_for_core", "")).strip()
        for next_step in summary.get("next_steps", []):
            step = str(next_step).strip()
            if step and step not in self.session.open_questions:
                self.session.open_questions.append(step)

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
