"""Run a small local memory-manager flow without calling an LLM server."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_manager import TieredMemoryManager


def main() -> None:
    manager = TieredMemoryManager(
        session_id="demo",
        project_name="Memory Manager Demo",
        pinned_summary="Show how hot, warm, and cold memory work together.",
    )
    manager.add_user_message("We are building a local memory layer for LLM chats.")
    manager.add_assistant_message(
        "Decision: warm storage is canonical; cold storage is rebuildable.",
        tags=["decision", "high-priority"],
    )

    warm_section_id = manager.compact_hot_to_warm()
    cold_chunk_ids = manager.index_warm_to_cold()
    context = manager.build_context_for_llm("What did we decide about storage?")

    print(f"session_id: {manager.session_id}")
    print(f"warm_section_id: {warm_section_id}")
    print(f"cold_chunk_ids: {cold_chunk_ids}")
    print("\n--- prompt context ---")
    print(context)


if __name__ == "__main__":
    main()
