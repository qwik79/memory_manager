"""Send one memory-augmented chat request to a local Ollama server.

Start Ollama separately, for example:
    ollama serve
    ollama pull qwen2.5:7b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_manager import OllamaIntegration, TieredMemoryManager


def post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory-augmented Ollama chat example")
    parser.add_argument("message", help="User message to send")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model name")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("--session", default="ollama-demo", help="Memory session ID")
    args = parser.parse_args()

    manager = TieredMemoryManager(
        session_id=args.session,
        project_name="Ollama memory demo",
        pinned_summary="Use local tiered memory to augment an Ollama chat request.",
    )
    manager.add_user_message(args.message)

    integration = OllamaIntegration(manager, primary_model=args.model)
    system_prompt, tools = integration.get_context_for_llm_call(
        {"role": "user", "content": args.message}
    )
    payload = {
        "model": args.model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": args.message},
        ],
        "tools": tools,
    }

    result = post_json(f"{args.host.rstrip('/')}/api/chat", payload)
    answer = result.get("message", {}).get("content", "")
    manager.add_assistant_message(answer)
    manager.compact_hot_to_warm()
    manager.index_warm_to_cold()

    print(answer)


if __name__ == "__main__":
    main()
