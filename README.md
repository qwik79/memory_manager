# Tiered Memory Manager

A local-first, dependency-light refactor of the tiered LLM memory-manager ideas from the project notes. The implementation keeps the useful concepts from the exploratory drafts while separating responsibilities into testable modules.

## Architecture

| Tier | Module | Purpose |
| --- | --- | --- |
| Tier 0 | `active_context.py` | Render pinned project state and assemble prompt-ready context. |
| Tier 1 | `hot_buffer.py` | Store recent raw conversation entries in an atomic JSON hot buffer. |
| Tier 2 | `warm_store.py` | Persist canonical summarized sections on disk with provenance. |
| Tier 3 | `cold_store.py` | Maintain a rebuildable long-term retrieval index over warm sections. |

The warm store is treated as the canonical durable memory after compaction. The cold store is intentionally rebuildable and dependency-free for now; it uses lexical vectors rather than binding the core package to Chroma/Qdrant. A vector database adapter can later implement the same public methods.

## Design choices preserved from the drafts

- Pinned context that is always available to the main model.
- Hot/warm/cold tiers inspired by OS memory hierarchy.
- Structured entries with IDs, timestamps, tags, relevance hints, and source provenance.
- Heuristic summarization as a safe fallback for a future secondary LLM.
- Explicit eviction scoring using recency, references, semantic/keyword overlap, and tags.
- Ollama-style adapter that prepares a system prompt and memory tool definitions.

## Quick example

```python
from memory_manager import TieredMemoryManager, OllamaIntegration

manager = TieredMemoryManager(
    session_id="demo",
    project_name="Memory Manager",
    pinned_summary="Build a tiered context-extension system.",
)
manager.add_user_message("We need hot, warm, and cold memory tiers.")
manager.add_assistant_message("Decision: warm storage is canonical; cold storage is rebuildable.", tags=["decision", "high-priority"])

section_id = manager.compact_hot_to_warm()
chunk_ids = manager.index_warm_to_cold()
context = manager.build_context_for_llm("What did we decide about warm storage?")

ollama = OllamaIntegration(manager)
system_prompt, tools = ollama.get_context_for_llm_call({"role": "user", "content": "Summarize the memory design."})
```

## Development

Run the smoke tests with:

```bash
python -m pytest
```
