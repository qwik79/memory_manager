"""Interactive chat against any OpenAI-compatible server, with virtualized context.

The script owns the history (as any chat app would) and lets ContextVirtualizer
fit it to a deliberately small window, so you can watch offloading happen.

    python examples/chat_repl.py --base-url http://localhost:11434/v1 --model qwen2.5:7b --context-tokens 2048
    python examples/chat_repl.py --base-url http://localhost:1234/v1 --model local-model

Memory is ephemeral unless you pass --persist DIR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_manager import ContextVirtualizer, TieredMemoryManager, VirtualizerConfig


def complete(base_url: str, model: str, messages: list[dict]) -> str:
    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps({"model": model, "messages": messages, "stream": False}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=600) as response:
        return json.loads(response.read())["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--persist", metavar="DIR", help="keep memory on disk under DIR")
    parser.add_argument("--session", default="repl")
    args = parser.parse_args()

    if args.persist:
        manager = TieredMemoryManager.persistent(args.persist, session_id=args.session)
    else:
        manager = TieredMemoryManager.ephemeral(args.session)
    virtualizer = ContextVirtualizer(
        manager,
        VirtualizerConfig(
            max_context_tokens=args.context_tokens,
            reserve_tokens=min(512, args.context_tokens // 4),
            history_budget_tokens=args.context_tokens // 8,
            retrieval_budget_tokens=args.context_tokens // 8,
        ),
    )
    history: list[dict] = [{"role": "system", "content": "You are a helpful assistant."}]
    print("Type a message (Ctrl-D to quit).", file=sys.stderr)
    with manager:
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if not text:
                continue
            history.append({"role": "user", "content": text})
            prompt = virtualizer.rewrite(history)
            stats = virtualizer.last_stats
            print(
                f"  [{stats['tokens_in']} -> {stats['tokens_out']} est. tokens, "
                f"offloaded {stats['offloaded_messages_total']} msgs, retrieved {stats['retrieved_excerpts']}]",
                file=sys.stderr,
            )
            reply = complete(args.base_url, args.model, prompt)
            virtualizer.observe(reply)
            history.append({"role": "assistant", "content": reply})
            print(reply)


if __name__ == "__main__":
    main()
