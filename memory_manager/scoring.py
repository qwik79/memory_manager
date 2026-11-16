"""Relevance scoring, auto-importance classification, and in-memory entry index.

This module provides three capabilities:

1. **Auto-importance classification** — regex-based content analysis that
   promotes the default ``"medium"`` importance to ``"high"`` or ``"critical"``
   when the content contains decisions, config values, security constraints,
   or other high-signal patterns.  Runs in <50µs on typical messages.

2. **Composite relevance scoring** — a continuous ``[0.0, 1.0]`` score
   combining five weighted signals (importance, recency, reference count,
   content richness, tag bonuses).  Used by the eviction manager and
   ``build_context_for_llm()`` to rank entries.

3. **EntryIndex** — a lightweight in-memory index over hot buffer entries.
   Maintains dicts mapping tags/importance/phase → entry IDs and a sorted
   score list.  Updated incrementally on every ``append_entry()`` and
   ``tag_entry()``.  Provides O(1) lookups by tag and O(k) top-k retrieval.

All operations are zero-I/O and microsecond-latency.
"""

from __future__ import annotations

import bisect
import math
import re
import time
from typing import Any


# ---------------------------------------------------------------------------
# Importance weights — maps importance labels to [0.0, 1.0]
# ---------------------------------------------------------------------------

IMPORTANCE_WEIGHTS: dict[str, float] = {
    "critical": 1.0,
    "high": 0.8,
    "medium": 0.5,
    "low": 0.2,
}

# Recency half-life in hours.  After this many hours the recency signal
# contributes half its maximum weight.
_RECENCY_HALF_LIFE_HOURS = 16.0
_RECENCY_DECAY = math.log(2) / _RECENCY_HALF_LIFE_HOURS  # λ for exp(-λ·t)


# ---------------------------------------------------------------------------
# Auto-importance classification
# ---------------------------------------------------------------------------

_CRITICAL_RE = re.compile(
    r"\b(always|never|must not|must|mandatory|critical|security|secret|"
    r"do not|forbidden|unsafe|vulnerability|breaking change)\b"
    r"|\[pin\]",
    re.IGNORECASE,
)
_HIGH_RE = re.compile(
    r"\b(decision|decided|going with|will use|chosen|selected|"
    r"architecture|schema|migration|deploy|config|port|endpoint|"
    r"api[- ]?key|password|credential|secret|model)\b"
    r"|(?:[./][\w./\\-]+\.(?:py|js|ts|go|rs|yaml|json|toml|md|sh))"
    r"|(?:(?:port|url|host|model|endpoint)\s*[:=]\s*[\w./:-]+)",
    re.IGNORECASE,
)
_LOW_RE = re.compile(
    r"^\s*(?:ok|okay|sure|sounds good|thanks|thank you|got it|"
    r"understood|yes|yep|yeah|no problem|will do|ack|lgtm|"
    r"👍|✅|👌)\s*$",
    re.IGNORECASE,
)


def auto_classify_importance(content: str) -> str:
    """Classify content importance from its text.

    Returns ``"critical"``, ``"high"``, ``"medium"``, or ``"low"``.
    Designed to be called when the caller hasn't set an explicit importance
    (i.e. importance is still the default ``"medium"``).

    Runs in <50µs on typical messages — safe for inline use.
    """
    if not content or not content.strip():
        return "low"
    # Short acknowledgements → low.  Check first to avoid false positives
    # on "ok" inside a longer sentence.
    if _LOW_RE.match(content.strip()):
        return "low"
    if _CRITICAL_RE.search(content):
        return "critical"
    if _HIGH_RE.search(content):
        return "high"
    return "medium"


# ---------------------------------------------------------------------------
# Content richness
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```", re.MULTILINE)
_PATH_RE = re.compile(r"(?:[./][\w./\\-]+\.(?:py|js|ts|go|rs|yaml|json|toml|md|sh|txt))")
_CONFIG_RE = re.compile(
    r"(?:port|url|host|addr|key|model|endpoint)\s*[:=]\s*[\w./:-]+",
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"\b(?:decision|decided|going with|will use|chosen|selected)\b",
    re.IGNORECASE,
)


def compute_content_richness(content: str) -> float:
    """Score content richness on ``[0.0, 1.0]``.

    Measures presence of code blocks, file paths, config values, and
    decision language.  Higher scores indicate content that is more
    valuable to retain in the hot buffer.
    """
    if not content:
        return 0.0
    score = 0.0
    # Code blocks: each one adds 0.15 (up to 0.45)
    code_blocks = len(_CODE_BLOCK_RE.findall(content)) // 2  # pairs
    score += min(0.45, code_blocks * 0.15)
    # File paths: each adds 0.08 (up to 0.24)
    paths = len(set(_PATH_RE.findall(content)))
    score += min(0.24, paths * 0.08)
    # Config values: each adds 0.1 (up to 0.2)
    configs = len(set(_CONFIG_RE.findall(content)))
    score += min(0.20, configs * 0.10)
    # Decision language: binary 0.15
    if _DECISION_RE.search(content):
        score += 0.15
    # Length bonus: longer content is usually more substantive
    words = len(content.split())
    if words > 50:
        score += 0.05
    if words > 200:
        score += 0.05
    return min(1.0, score)


# ---------------------------------------------------------------------------
# Composite relevance score
# ---------------------------------------------------------------------------

# Signal weights (must sum to 1.0)
_W_IMPORTANCE = 0.25
_W_RECENCY = 0.25
_W_REFERENCES = 0.15
_W_RICHNESS = 0.20
_W_TAGS = 0.15

# Tag score modifiers
_TAG_SCORES: dict[str, float] = {
    "pin": 0.15,
    "decision": 0.10,
    "high-priority": 0.10,
    "critical": 0.10,
    "discard": -0.15,
    "archive": -0.10,
}


def compute_relevance_score(
    entry: dict[str, Any],
    reference_count: int = 0,
    now: float | None = None,
) -> float:
    """Compute a composite relevance score for a hot buffer entry.

    Parameters
    ----------
    entry:
        A hot buffer entry dict with at least ``content``, ``importance``,
        ``tags``, and ``created_at`` fields.
    reference_count:
        Number of times this entry has been referenced/accessed.
    now:
        Current timestamp (defaults to ``time.time()``).

    Returns
    -------
    float
        A score in ``[0.0, 1.0]`` where higher means more relevant.
    """
    if now is None:
        now = time.time()

    # 1. Importance signal
    importance = entry.get("importance", "medium")
    importance_score = IMPORTANCE_WEIGHTS.get(importance, 0.5)

    # 2. Recency signal — exponential decay
    created = entry.get("created_at", now)
    age_hours = max(0.0, (now - created) / 3600)
    recency_score = math.exp(-_RECENCY_DECAY * age_hours)

    # 3. Reference count signal — diminishing returns
    ref_score = min(1.0, reference_count / 5.0)

    # 4. Content richness signal
    richness_score = compute_content_richness(entry.get("content", ""))

    # 5. Tag bonus signal — sum of known tag modifiers, clamped [0, 1]
    tags = entry.get("tags", [])
    tag_score = sum(_TAG_SCORES.get(t, 0.0) for t in tags)
    tag_score = max(0.0, min(1.0, 0.5 + tag_score))  # center at 0.5

    composite = (
        _W_IMPORTANCE * importance_score
        + _W_RECENCY * recency_score
        + _W_REFERENCES * ref_score
        + _W_RICHNESS * richness_score
        + _W_TAGS * tag_score
    )
    return max(0.0, min(1.0, composite))


# ---------------------------------------------------------------------------
# EntryIndex — in-memory fast-lookup index
# ---------------------------------------------------------------------------


class EntryIndex:
    """Lightweight in-memory index over hot buffer entries.

    Provides O(1) lookups by tag, importance, and phase, plus O(k) top-k
    retrieval by score.  Updated incrementally — no full rebuilds.

    Thread-safety: callers must hold the hot buffer's ``_lock`` when mutating.
    """

    def __init__(self) -> None:
        self._by_tag: dict[str, set[str]] = {}
        self._by_importance: dict[str, set[str]] = {}
        self._by_phase: dict[str, set[str]] = {}
        self._scores: dict[str, float] = {}
        # (score, entry_id) sorted ascending — top-k is a tail slice.
        self._ranked: list[tuple[float, str]] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(
        self,
        entry_id: str,
        tags: list[str],
        importance: str,
        phase: str,
        score: float,
    ) -> None:
        """Index a new entry or update an existing one."""
        # Remove old entry if updating
        if entry_id in self._scores:
            self.remove(entry_id)

        self._scores[entry_id] = score
        bisect.insort(self._ranked, (score, entry_id))

        for tag in tags:
            self._by_tag.setdefault(tag, set()).add(entry_id)
        self._by_importance.setdefault(importance, set()).add(entry_id)
        if phase:
            self._by_phase.setdefault(phase, set()).add(entry_id)

    def remove(self, entry_id: str) -> None:
        """Remove an entry from the index."""
        old_score = self._scores.pop(entry_id, None)
        if old_score is not None:
            try:
                self._ranked.remove((old_score, entry_id))
            except ValueError:
                pass
        # Clean up tag/importance/phase sets
        for tag_set in self._by_tag.values():
            tag_set.discard(entry_id)
        for imp_set in self._by_importance.values():
            imp_set.discard(entry_id)
        for phase_set in self._by_phase.values():
            phase_set.discard(entry_id)

    def update_score(self, entry_id: str, new_score: float) -> None:
        """Update an entry's score and re-rank it."""
        old_score = self._scores.get(entry_id)
        if old_score is not None:
            try:
                self._ranked.remove((old_score, entry_id))
            except ValueError:
                pass
        self._scores[entry_id] = new_score
        bisect.insort(self._ranked, (new_score, entry_id))

    def add_tag(self, entry_id: str, tag: str) -> None:
        """Add a tag to an already-indexed entry."""
        self._by_tag.setdefault(tag, set()).add(entry_id)

    def rebuild(
        self,
        entries: list[dict[str, Any]],
        reference_counts: dict[str, int] | None = None,
    ) -> None:
        """Full rebuild from a list of entry dicts.  Called on startup."""
        self._by_tag.clear()
        self._by_importance.clear()
        self._by_phase.clear()
        self._scores.clear()
        self._ranked.clear()

        ref_counts = reference_counts or {}
        now = time.time()
        for entry in entries:
            eid = entry.get("entry_id", "")
            if not eid:
                continue
            score = compute_relevance_score(entry, ref_counts.get(eid, 0), now)
            self.add(
                entry_id=eid,
                tags=entry.get("tags", []),
                importance=entry.get("importance", "medium"),
                phase=entry.get("phase", ""),
                score=score,
            )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def top_k(
        self,
        k: int = 10,
        tags_filter: list[str] | None = None,
        importance_min: str | None = None,
        phase: str | None = None,
    ) -> list[str]:
        """Return up to *k* entry IDs, highest score first.

        Optional filters narrow the candidate set before ranking.
        """
        candidates: set[str] | None = None

        if tags_filter:
            tag_candidates: set[str] = set()
            for tag in tags_filter:
                tag_candidates |= self._by_tag.get(tag, set())
            candidates = tag_candidates

        if importance_min:
            min_weight = IMPORTANCE_WEIGHTS.get(importance_min, 0.0)
            eligible = set()
            for imp, ids in self._by_importance.items():
                if IMPORTANCE_WEIGHTS.get(imp, 0.0) >= min_weight:
                    eligible |= ids
            candidates = eligible if candidates is None else candidates & eligible

        if phase:
            phase_ids = self._by_phase.get(phase, set())
            candidates = phase_ids if candidates is None else candidates & phase_ids

        if candidates is not None:
            # Filter the ranked list
            result: list[str] = []
            for score, eid in reversed(self._ranked):
                if eid in candidates:
                    result.append(eid)
                    if len(result) >= k:
                        break
            return result

        # No filters — just take the top k from the ranked list
        return [eid for _, eid in reversed(self._ranked[-k:])]

    def get_score(self, entry_id: str) -> float:
        """Return the current score for an entry, or 0.0 if unknown."""
        return self._scores.get(entry_id, 0.0)

    def __len__(self) -> int:
        return len(self._scores)

    def __contains__(self, entry_id: str) -> bool:
        return entry_id in self._scores
