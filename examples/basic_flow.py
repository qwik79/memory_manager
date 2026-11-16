"""Self-contained demo of the tiered memory system — no LLM server required.

Demonstrates:
- Context manager usage
- Auto-importance classification (no need to set importance manually)
- Ranked context assembly
- Hot → warm auto-cascade
- Cold retrieval
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_manager import TieredMemoryManager


def main() -> None:
    with TieredMemoryManager(
        session_id="demo",
        project_name="Memory Manager Demo",
        pinned_summary="Show how hot, warm, and cold memory tiers work together.",
    ) as manager:
        # Auto-classify detects "Decision:" → importance="high" automatically.
        manager.add_user_message("We are building a local memory layer for LLM chats.")
        manager.add_assistant_message(
            "Decision: warm storage is canonical; cold storage is rebuildable index.",
            tags=["decision"],
        )
        manager.add_user_message("What about security?")
        manager.add_assistant_message(
            "We must always sign files with HMAC before writing to disk. "
            "Never store credentials in plaintext."
        )

        # compact_hot_to_warm auto-cascades to cold — no index_warm_to_cold() needed.
        warm_section_id = manager.compact_hot_to_warm()

        # Ranked retrieval: highest-scoring entries appear first in the prompt.
        context = manager.build_context_for_llm("What did we decide about storage?")

        # Show entry scores from the in-memory index.
        ranked = manager.hot_buffer.get_entries_ranked(limit=10)
        print(f"session_id:       {manager.session_id}")
        print(f"warm_section_id:  {warm_section_id}")
        print(f"hot entries:      {len(ranked)}")
        if ranked:
            top = ranked[0]
            score = manager.hot_buffer.get_entry_score(top.get("entry_id", ""))
            print(f"top entry score:  {score:.3f}  importance={top.get('importance')}")
        print("\n--- prompt context (truncated to 500 chars) ---")
        print(context[:500])


if __name__ == "__main__":
    main()
