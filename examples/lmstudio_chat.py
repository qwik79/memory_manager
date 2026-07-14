"""Send one memory-augmented chat request to LM Studio's local server.

Start LM Studio separately, load a model, and enable its local server. The default
LM Studio endpoint is OpenAI-compatible at http://localhost:1234/v1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_manager import TieredMemoryManager


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
    parser = argparse.ArgumentParser(description="Memory-augmented LM Studio chat example")
    parser.add_argument("message", help="User message to send")
    parser.add_argument("--model", default="local-model", help="LM Studio model identifier")
    parser.add_argument("--host", default="http://localhost:1234/v1", help="LM Studio /v1 URL")
    parser.add_argument("--session", default="lmstudio-demo", help="Memory session ID")
    args = parser.parse_args()

    manager = TieredMemoryManager(
        session_id=args.session,
        project_name="LM Studio memory demo",
        pinned_summary="Use local tiered memory to augment an OpenAI-compatible chat request.",
    )
    manager.add_user_message(args.message)
    system_prompt = manager.build_context_for_llm(query=args.message)
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": args.message},
        ],
    }

    result = post_json(f"{args.host.rstrip('/')}/chat/completions", payload)
    answer = result["choices"][0]["message"]["content"]
    manager.add_assistant_message(answer)
    manager.compact_hot_to_warm()
    manager.index_warm_to_cold()

    print(answer)


if __name__ == "__main__":
    main()
