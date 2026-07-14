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

`TieredMemoryManager` is the main class you start with. It lives in
`memory_manager/orchestrator.py` and is re-exported from `memory_manager`, so
application code can import it directly:

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

## How to wire it into a local LLM app

The manager does not run Ollama or LM Studio for you. Instead, your app follows
the same loop for any chat backend:

1. Create one `TieredMemoryManager` per project/session.
2. Save the incoming user message with `manager.add_user_message(...)`.
3. Build the memory-aware system prompt with
   `manager.build_context_for_llm(query=user_message)`.
4. Send that system prompt plus the user message to your local model server.
5. Save the assistant reply with `manager.add_assistant_message(...)`.
6. Periodically call `manager.compact_hot_to_warm()` and
   `manager.index_warm_to_cold()` so older chat turns become retrievable memory.

### Ollama

Start Ollama separately, then call its `/api/chat` endpoint with the memory
context as the system message. A minimal runnable example is included:

```bash
python examples/ollama_chat.py "What do we remember about storage?" --model qwen2.5:7b
```

That script:

- creates a `TieredMemoryManager`;
- uses `OllamaIntegration.get_context_for_llm_call(...)` to prepare the system
  prompt and tool definitions;
- posts to `http://localhost:11434/api/chat`;
- stores the assistant reply back into memory.

### LM Studio

Start LM Studio, load a model, enable the local server, then call its
OpenAI-compatible `/v1/chat/completions` endpoint. A minimal runnable example is
included:

```bash
python examples/lmstudio_chat.py "What do we remember about storage?" --model local-model
```

That script builds the memory context directly with
`manager.build_context_for_llm(...)` and sends it as the system message to
`http://localhost:1234/v1/chat/completions`.

### Local smoke/demo script

If you just want to see the memory tiers work without starting an LLM server,
run:

```bash
python examples/basic_flow.py
```

## Development

Run the smoke tests with:

```bash
python -m pytest
```
