"""Eviction/offload scoring for moving entries between memory tiers."""

from __future__ import annotations

import time
from typing import Any

from .hot_buffer import HotBufferManager
from .scoring import IMPORTANCE_WEIGHTS, compute_relevance_score


class EvictionManager:
    """Scores hot entries and chooses candidates for pinning/offload.

    Uses the composite relevance scorer from :mod:`scoring` so that
    eviction decisions are consistent with retrieval ranking and the
    in-memory ``EntryIndex``.

    Entries with importance ``"critical"`` or ``"high"`` are protected
    from eviction (treated the same as ``"pin"``-tagged entries).
    """

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
        """Return (offload_ids, pin_ids).

        Scoring uses the unified :func:`scoring.compute_relevance_score`
        composite.  High-importance entries are auto-pinned.
        """
        entries = self.hot.get_entries(limit=None)
        scored: list[dict[str, Any]] = []
        now = time.time()
        for entry in entries:
            entry_id = entry.get("entry_id")
            if not entry_id:
                continue
            ref_count = self.reference_count.get(entry_id, 0)
            score = compute_relevance_score(entry, reference_count=ref_count, now=now)
            age = max(0.0, now - self.access_history.get(
                entry_id, entry.get("created_at", now)
            ))
            scored.append({"entry": entry, "score": score, "age": age})
        scored.sort(key=lambda row: row["score"])

        offload: list[str] = []
        pin: list[str] = []
        for row in scored:
            entry = row["entry"]
            tags = entry.get("tags", [])
            entry_id = entry.get("entry_id")
            importance = entry.get("importance", "medium")

            # Protect high-value entries from eviction.
            is_protected = (
                "pin" in tags
                or "high-priority" in tags
                or IMPORTANCE_WEIGHTS.get(importance, 0.5) >= 0.8  # high or critical
            )
            if is_protected:
                pin.append(entry_id)
            elif row["score"] < 0.35 and row["age"] >= min_age_seconds:
                offload.append(entry_id)
            if len(offload) >= max_offloads:
                break
        return offload, pin
