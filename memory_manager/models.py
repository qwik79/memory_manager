"""Shared data models for the tiered memory manager."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    """Return a UTC timestamp suitable for durable JSON records."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    """Create a short, stable-enough identifier with a semantic prefix."""
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class MemoryTierConfig:
    """Configuration for one memory tier."""

    name: str
    capacity_tokens: float
    target_size_bytes: int | None
    eviction_threshold: float
    read_cost_ms: float = 0.0
    write_cost_ms: float = 0.0


TIER_0_CONFIG = MemoryTierConfig("active_context", 32_000, 512 * 1024, 0.85, 0.1, 0.1)
TIER_1_CONFIG = MemoryTierConfig("hot_buffer", 200_000, 5 * 1024 * 1024, 0.90, 0.01, 0.01)
TIER_2_CONFIG = MemoryTierConfig("warm_store", 1_000_000, 50 * 1024 * 1024, 0.95, 1.0, 2.0)
TIER_3_CONFIG = MemoryTierConfig("cold_store", float("inf"), None, 0.98, 5.0, 10.0)


@dataclass
class MemorySession:
    """Durable state for a project/session."""

    session_id: str
    project_name: str = ""
    pinned_summary: str = ""
    current_task: str | None = None
    active_decisions: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    completed_subtasks: list[str] = field(default_factory=list)
    memory_notes: str = ""

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["last_updated"] = utc_now_iso()
        return data


@dataclass
class MemoryEntry:
    """One raw message/event in the hot buffer."""

    entry_id: str
    session_id: str
    entry_type: str
    content: str
    created_at: float = field(default_factory=time.time)
    context_ref: str | None = None
    tags: list[str] = field(default_factory=list)
    relevance_hint: float = 1.0
    compacted_at: float | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def create(
        cls,
        session_id: str,
        entry_type: str,
        content: str,
        context_ref: str | None = None,
        tags: list[str] | None = None,
        relevance_hint: float = 1.0,
    ) -> "MemoryEntry":
        return cls(
            entry_id=new_id("entry"),
            session_id=session_id,
            entry_type=entry_type,
            content=content,
            context_ref=context_ref,
            tags=tags or [],
            relevance_hint=relevance_hint,
        )


@dataclass
class WarmSection:
    """Canonical compacted memory stored on disk."""

    section_id: str
    session_id: str
    section_name: str
    content: str
    source_entry_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    relations: dict[str, list[str]] = field(
        default_factory=lambda: {"parent_ids": [], "child_ids": [], "related_ids": []}
    )
    created_at: float = field(default_factory=time.time)
    indexed_at: float | None = None
    schema_version: int = 1

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    """Result returned by warm/cold retrieval."""

    source: str
    content: str
    relevance: float
    metadata: dict[str, Any] = field(default_factory=dict)
    identifier: str | None = None
