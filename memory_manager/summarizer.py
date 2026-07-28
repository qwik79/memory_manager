"""Summarization interfaces for compacting hot memory into warm sections."""

from __future__ import annotations

import re
import json
from urllib.request import Request, urlopen
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


class OllamaSummarizer:
    """CPU-only Ollama memory unit with deterministic fallback on any failure."""

    SYSTEM_PROMPT = """You are a memory distillation agent. Return only one JSON object with keys:
summary (string), decisions (string list), next_steps (string list), facts (object),
tier (pin, extend, or archive), notes_for_core (string), and confidence (0 to 1).
Be terse and preserve exact technical values, names, and file paths."""

    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        endpoint: str = "http://127.0.0.1:11434/api/chat",
        timeout: float = 90.0,
        fallback: Summarizer | None = None,
    ):
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.fallback = fallback or HeuristicSummarizer()

    def summarize_entries(self, entries: list[dict[str, Any]], context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not entries:
            return self.fallback.summarize_entries(entries, context)
        previous = str((context or {}).get("pinned_summary", ""))
        content = "\n".join(f"{entry.get('entry_type', 'event').upper()}: {entry.get('content', '')}" for entry in entries)
        prompt = f"Previous project state:\n{previous}\n\nNew content:\n{content}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"num_gpu": 0, "num_ctx": 4096},
        }
        try:
            request = Request(self.endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode())["message"]["content"]
            result = json.loads(raw)
            if not isinstance(result.get("summary"), str):
                raise ValueError("memory unit omitted summary")
            result.setdefault("decisions", [])
            result.setdefault("next_steps", [])
            result.setdefault("facts", {})
            result.setdefault("notes_for_core", "")
            result.setdefault("confidence", 0.7)
            return result
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return self.fallback.summarize_entries(entries, context)
