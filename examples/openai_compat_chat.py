"""Send one memory-augmented chat request to any OpenAI-compatible local server.

Works with LM Studio, LocalAI, vLLM, llama.cpp server, text-generation-webui,
or any server exposing /v1/chat/completions.

Example with LM Studio:
    1. Start LM Studio, load a model, enable server.
    2. python examples/openai_compat_chat.py "What do we remember?"

Example with LocalAI:
    1. local-ai run qwen2.5:7b
    2. python examples/openai_compat_chat.py "Summarize the project." --host http://localhost:8080/v1

Example with vLLM:
    1. vllm serve qwen2.5-7b --port 8000
    2. python examples/openai_compat_chat.py "Next steps?" --host http://localhost:8000/v1 --model qwen2.5-7b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_manager import LLMIntegration, OpenAIDistiller, TieredMemoryManager


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
    parser = argparse.ArgumentParser(
        description="Memory-augmented chat with any OpenAI-compatible server"
    )
    parser.add_argument("message", help="User message to send")
    parser.add_argument("--model", default="local-model", help="Model name/identifier")
    parser.add_argument(
        "--host", default="http://localhost:1234/v1",
        help="Base URL (e.g. http://localhost:1234/v1 for LM Studio)"
    )
    parser.add_argument("--session", default="openai-demo", help="Memory session ID")
    parser.add_argument(
        "--distill-model", default="",
        help="Optional: use a second model for background distillation"
    )
    args = parser.parse_args()

    # Optionally set up background distillation with a secondary model.
    distiller = None
    if args.distill_model:
        distiller = OpenAIDistiller(
            model=args.distill_model,
            endpoint=f"{args.host.rstrip('/')}/chat/completions",
        )

    with TieredMemoryManager(
        session_id=args.session,
        project_name="OpenAI-compat memory demo",
        pinned_summary="Use local tiered memory to augment an OpenAI-compatible chat request.",
        distiller=distiller,
    ) as manager:
        manager.add_user_message(args.message)

        integration = LLMIntegration(manager, primary_model=args.model)
        system_prompt, tools = integration.get_context_for_llm_call(
            {"role": "user", "content": args.message}
        )

        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": args.message},
            ],
            "tools": tools if tools else None,
        }
        # Remove None values for servers that reject them.
        payload = {k: v for k, v in payload.items() if v is not None}

        result = post_json(f"{args.host.rstrip('/')}/chat/completions", payload)
        answer = result["choices"][0]["message"]["content"]
        manager.add_assistant_message(answer)
        manager.compact_hot_to_warm()

    print(answer)


if __name__ == "__main__":
    main()
