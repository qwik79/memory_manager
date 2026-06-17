"""Summarization interfaces for compacting hot memory into warm sections."""

from __future__ import annotations

import re
from typing import Any, Protocol


class Summarizer(Protocol):
    def summarize_entries(self, entries: list[dict[str, Any]], context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Summarize raw entries into structured memory."""


class HeuristicSummarizer:
    """Deterministic fallback summarizer used when no secondary LLM is configured."""

    def summarize_entries(self, entries: list[dict[str, Any]], context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not entries:
            return {"summary": "", "decisions": [], "facts": {}, "confidence": 1.0}
        content = "\n".join(e.get("content", "") for e in entries)
        lines = [line.rstrip() for line in content.splitlines() if line.strip()]
        if len(lines) <= 30:
            summary = "\n".join(lines)
        else:
            summary = "\n".join(lines[:10] + ["...[middle compacted]..."] + lines[-10:])
        decisions = [line for line in lines if "decision" in line.lower() or "[pin]" in line.lower()]
        paths = sorted(set(re.findall(r"(?:[\w.-]+/)+[\w.-]+", content)))
        facts = {"paths": paths} if paths else {}
        confidence = self._estimate_confidence(content)
        return {
            "summary": summary,
            "decisions": decisions[:10],
            "facts": facts,
            "source_entry_ids": [e.get("entry_id") for e in entries if e.get("entry_id")],
            "confidence": confidence,
            "original_length": len(content),
        }

    def _estimate_confidence(self, content: str) -> float:
        code_blocks = len(re.findall(r"```.*?```", content, flags=re.DOTALL))
        confidence = 0.85
        if code_blocks:
            confidence -= min(0.3, code_blocks * 0.05)
        if len(content) > 20_000:
            confidence -= 0.15
        return max(0.4, min(1.0, confidence))
