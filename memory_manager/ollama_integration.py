"""Prompt/tool adapter for Ollama-style chat clients."""

from __future__ import annotations

from typing import Any

from .orchestrator import TieredMemoryManager
from .storage import estimate_tokens


class OllamaIntegration:
    """Prepare memory context and tool schemas for an Ollama chat call."""

    def __init__(self, memory_manager: TieredMemoryManager, primary_model: str = "qwen2.5:7b"):
        self.memory_manager = memory_manager
        self.primary_model = primary_model

    def get_context_for_llm_call(self, message: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        query = str(message.get("content", ""))
        system_prompt = self.memory_manager.build_context_for_llm(query=query)
        return system_prompt, self.tool_definitions()

    def estimate_context_usage(self) -> dict[str, Any]:
        context = self.memory_manager.build_context_for_llm()
        tokens = estimate_tokens(context, overhead=200)
        return {
            "total_estimate": tokens,
            "max_available": self.memory_manager.active_context.max_tokens,
            "usage_percent": min(1.0, tokens / self.memory_manager.active_context.max_tokens),
        }

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_memory",
                    "description": "Retrieve relevant warm/cold memories for a topic.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 5}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tag_memory_entry",
                    "description": "Tag a hot-buffer entry as pin/archive/discard/high-priority.",
                    "parameters": {
                        "type": "object",
                        "properties": {"entry_id": {"type": "string"}, "tag": {"type": "string"}},
                        "required": ["entry_id", "tag"],
                    },
                },
            },
        ]

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if name == "read_memory":
            return [r.__dict__ for r in self.memory_manager.refresh_context(arguments.get("query", ""), arguments.get("limit", 5))]
        if name == "tag_memory_entry":
            return self.memory_manager.hot_buffer.tag_entry(arguments["entry_id"], arguments["tag"])
        raise ValueError(f"Unknown memory tool: {name}")
