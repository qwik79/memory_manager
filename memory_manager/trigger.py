"""Composite signals for deciding when active conversation should be compacted."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .storage import estimate_tokens

Embedding = Sequence[float]
EmbedFn = Callable[[str], Embedding]


@dataclass(frozen=True)
class TriggerConfig:
    """Tunable offload thresholds; token pressure always has a hard override."""

    context_size: int = 32_000
    token_soft_pct: float = 0.65
    token_hard_pct: float = 0.90
    drift_threshold: float = 0.32
    composite_threshold: float = 0.40
    structural_patterns: tuple[str, ...] = field(
        default=(
            r"\b(works|passing|fixed|resolved|done|merged|committed)\b",
            r"\b(next up|switching to|moving on|different file|new module)\b",
            r"```\s*$",
        )
    )

    def __post_init__(self) -> None:
        if self.context_size <= 0:
            raise ValueError("context_size must be positive")
        if not 0 <= self.token_soft_pct < self.token_hard_pct <= 1:
            raise ValueError("token thresholds must satisfy 0 <= soft < hard <= 1")


class OffloadTrigger:
    """Combine token pressure, semantic drift, and structural seam signals."""

    def __init__(self, config: TriggerConfig | None = None):
        self.config = config or TriggerConfig()

    def evaluate(
        self,
        messages: list[dict[str, object]],
        pinned_embedding: Embedding | None = None,
        embed_fn: EmbedFn | None = None,
    ) -> dict[str, object]:
        text = "\n".join(str(message.get("content", "")) for message in messages)
        token_pct = estimate_tokens(text) / self.config.context_size
        if token_pct >= self.config.token_hard_pct:
            return {"should_offload": True, "score": 1.0, "token_pct": token_pct, "reasons": ["hard token limit"]}

        score = 0.0
        reasons: list[str] = []
        if token_pct >= self.config.token_soft_pct:
            ramp = (token_pct - self.config.token_soft_pct) / (
                self.config.token_hard_pct - self.config.token_soft_pct
            )
            score += 0.4 * min(ramp, 1.0)
            reasons.append(f"token pressure {token_pct:.0%}")

        if pinned_embedding is not None and embed_fn is not None and messages:
            recent = "\n".join(str(message.get("content", "")) for message in messages[-5:])
            drift = 1.0 - _cosine_similarity(embed_fn(recent), pinned_embedding)
            if drift >= self.config.drift_threshold:
                score += 0.4 * min(drift / 0.6, 1.0)
                reasons.append(f"semantic drift {drift:.2f}")

        last = str(messages[-1].get("content", "")).lower() if messages else ""
        if any(re.search(pattern, last) for pattern in self.config.structural_patterns):
            score += 0.2
            reasons.append("structural seam")
        return {
            "should_offload": score >= self.config.composite_threshold,
            "score": min(score, 1.0),
            "token_pct": token_pct,
            "reasons": reasons,
        }


def _cosine_similarity(left: Embedding, right: Embedding) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0
