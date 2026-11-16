"""OpenAI-compatible summarizer for LM Studio, LocalAI, vLLM, llama.cpp server.

Drop-in replacement for :class:`~summarizer.OllamaSummarizer` that calls any
server exposing the ``/v1/chat/completions`` endpoint.  Covers LM Studio,
LocalAI, vLLM, text-generation-webui, llama.cpp server, and any other
OpenAI-compatible local inference server.

Same system prompt.  Same JSON-mode parsing.  Same heuristic fallback on error.
"""

from __future__ import annotations

import json
from urllib.request import Request, urlopen
from typing import Any

from .summarizer import HeuristicSummarizer, Summarizer


class OpenAISummarizer:
    """Summarizer that calls an OpenAI-compatible ``/v1/chat/completions`` endpoint.

    Parameters
    ----------
    model:
        Model identifier the server uses to select the loaded model.
        For LM Studio this is typically the filename or alias shown in
        its UI.  For vLLM/LocalAI, the model name passed at launch.
    endpoint:
        Full URL to the ``/v1/chat/completions`` route.
    timeout:
        HTTP request timeout in seconds.
    fallback:
        Summarizer to use when the server is unreachable or returns
        invalid JSON.  Defaults to :class:`HeuristicSummarizer`.
    """

    SYSTEM_PROMPT = (
        "You are a memory distillation agent. Return only one JSON object with keys:\n"
        "summary (string), decisions (string list), next_steps (string list), facts (object),\n"
        "tier (pin, extend, or archive), notes_for_core (string), and confidence (0 to 1).\n"
        "Be terse and preserve exact technical values, names, and file paths."
    )

    def __init__(
        self,
        model: str = "local-model",
        endpoint: str = "http://127.0.0.1:1234/v1/chat/completions",
        timeout: float = 90.0,
        fallback: Summarizer | None = None,
    ):
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.fallback = fallback or HeuristicSummarizer()

    def summarize_entries(
        self,
        entries: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not entries:
            return self.fallback.summarize_entries(entries, context)

        previous = str((context or {}).get("pinned_summary", ""))
        content = "\n".join(
            f"{entry.get('entry_type', 'event').upper()}: {entry.get('content', '')}"
            for entry in entries
        )
        prompt = f"Previous project state:\n{previous}\n\nNew content:\n{content}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        try:
            request = Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            # OpenAI format: body["choices"][0]["message"]["content"]
            raw = body["choices"][0]["message"]["content"]
            result = json.loads(raw)
            if not isinstance(result.get("summary"), str):
                raise ValueError("memory unit omitted summary")
            result.setdefault("decisions", [])
            result.setdefault("next_steps", [])
            result.setdefault("facts", {})
            result.setdefault("notes_for_core", "")
            result.setdefault("confidence", 0.7)
            result.setdefault("tier", "extend")
            return result
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return self.fallback.summarize_entries(entries, context)
