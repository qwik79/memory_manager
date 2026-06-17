"""Eviction/offload scoring for moving entries between memory tiers."""

from __future__ import annotations

import math
import time
from typing import Any

from .hot_buffer import HotBufferManager


class EvictionManager:
    """Scores hot entries and chooses candidates for pinning/offload."""

    def __init__(self, hot_buffer: HotBufferManager):
        self.hot = hot_buffer
        self.access_history: dict[str, float] = {}
        self.reference_count: dict[str, int] = {}

    def record_access(self, entry_id: str) -> None:
        self.access_history[entry_id] = time.time()

    def record_reference(self, entry_id: str) -> None:
        self.reference_count[entry_id] = self.reference_count.get(entry_id, 0) + 1
        self.record_access(entry_id)

    def evaluate_for_eviction(
        self,
        current_task: str = "",
        max_offloads: int = 15,
        min_age_seconds: float = 2 * 3600,
    ) -> tuple[list[str], list[str]]:
        """Return entry IDs to offload and entry IDs to pin."""
        entries = self.hot.get_entries(limit=None)
        scored: list[dict[str, Any]] = []
        now = time.time()
        for entry in entries:
            entry_id = entry.get("entry_id")
            if not entry_id:
                continue
            last_accessed = self.access_history.get(entry_id, entry.get("created_at", now))
            age = max(0.0, now - last_accessed)
            recency_score = math.exp(-age / 86_400)
            ref_score = min(1.0, self.reference_count.get(entry_id, 0) * 0.25)
            semantic_score = self._keyword_overlap(entry.get("content", ""), current_task)
            score = 0.4 * recency_score + 0.2 * ref_score + 0.4 * semantic_score
            scored.append({"entry": entry, "score": score, "age": age})
        scored.sort(key=lambda row: row["score"])
        offload: list[str] = []
        pin: list[str] = []
        for row in scored:
            entry = row["entry"]
            tags = entry.get("tags", [])
            entry_id = entry.get("entry_id")
            if "pin" in tags or "high-priority" in tags:
                pin.append(entry_id)
            elif row["score"] < 0.35 and row["age"] >= min_age_seconds:
                offload.append(entry_id)
            if len(offload) >= max_offloads:
                break
        return offload, pin

    def _keyword_overlap(self, content: str, current_task: str) -> float:
        if not content or not current_task:
            return 0.5
        task_words = set(current_task.lower().split()[:20])
        content_words = set(content.lower().split())
        if not task_words:
            return 0.5
        return min(1.0, len(task_words & content_words) / len(task_words))
