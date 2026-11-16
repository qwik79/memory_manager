# llm-memory-manager

**A local-first, tiered memory system for extending LLM context windows.**

[![PyPI](https://img.shields.io/pypi/v/llm-memory-manager?color=blue)](https://pypi.org/project/llm-memory-manager/)
[![Python](https://img.shields.io/pypi/pyversions/llm-memory-manager)](https://pypi.org/project/llm-memory-manager/)
[![License](https://img.shields.io/github/license/gmic/memory_manager)](LICENSE)

Memory Manager acts as an external memory tier for LLM conversations — like how CPUs use L1/L2/L3 caches to extend register capacity. It sits between your conversation and the model, automatically scoring, compacting, and retrieving context so the LLM never forgets earlier decisions, file paths, or architectural choices.

**Zero required dependencies.** Works with any local inference server — Ollama, LM Studio, LocalAI, vLLM, llama.cpp, or any OpenAI-compatible backend. All AI features are opt-in via extras.

---

## Features

- 🧠 **Three-tier memory** — Hot (recent raw) → Warm (summarized) → Cold (indexed search) with automatic cascading
- 📊 **5-signal relevance scoring** — Importance, recency, references, content richness, and tag bonuses
- 🔍 **Hybrid retrieval** — BM25-inspired warm store + semantic embedding cold store (lexical fallback always available)
- ⚡ **Background distillation** — Heuristic, LLM, Cascade, and Speculative patterns
- 🔌 **MCP server** — 8 tools for integration with Claude Code, Cursor, Codex CLI, and any MCP-compatible editor
- 🔒 **HMAC-SHA256 tamper protection** — Optional Fernet encryption for sensitive data
- 📦 **Session export/import** — Portable tar.gz archives for backup and sharing
- 🧹 **Warm store GC** — Automatic purge of old indexed sections
- 🤖 **Model auto-pull** — Checks Ollama and downloads models when missing
- 🏷️ **Auto-importance classification** — Regex-based (<50µs) or set manually
- 📄 **PEP 561 typed** — Full type annotations, `py.typed` marker

---

## Installation

```bash
pip install llm-memory-manager
```

### Extras

```bash
# MCP server (for Claude Code, Cursor, Codex CLI)
pip install "llm-memory-manager[mcp]"

# ChromaDB vector store for cold tier
pip install "llm-memory-manager[chroma]"

# Fernet encryption for sensitive data
pip install "llm-memory-manager[encrypt]"

# Everything
pip install "llm-memory-manager[all]"
```

### From source

```bash
git clone https://github.com/gmic/memory_manager.git
cd memory_manager
pip install -e ".[all]"
```

---

## Quick Start

```python
from memory_manager import TieredMemoryManager

with TieredMemoryManager(
    session_id="my-project",
    project_name="My App",
) as manager:
    # Record conversation turns — importance is auto-detected.
    manager.add_user_message("Decision: use PostgreSQL for the backend.")
    manager.add_assistant_message("Good choice. I'll set up the schema.")

    # Build ranked context for the LLM prompt.
    context = manager.build_context_for_llm(query="database setup")

    # Compact when the hot buffer grows large.
    manager.compact_hot_to_warm()  # auto-cascades to cold index
```

---

## MCP Server

The MCP server exposes memory operations as tools for any MCP-compatible editor.

### Start the server

```bash
# Direct invocation
python -m memory_manager.mcp_server --session my-project

# Or via the console script (after pip install)
memory-manager-mcp --session my-project

# With tamper protection
memory-manager-mcp --session my-project --secure
```

### Configure in Claude Code

Add to your project's `.mcp.json`:

```json
{
    "servers": {
        "memory-manager": {
            "command": "memory-manager-mcp",
            "args": ["--session", "my-project"]
        }
    }
}
```

### Configure in Cursor / Windsurf

Add to your MCP config:

```json
{
    "memory-manager": {
        "command": "python",
        "args": ["-m", "memory_manager.mcp_server", "--session", "my-project"]
    }
}
```

### Available tools

| Tool | Description |
|------|-------------|
| `read_memory` | Ranked retrieval across all tiers |
| `add_memory` | Add entry with auto-importance classification |
| `tag_entry` | Tag an entry (`pin`, `decision`, `archive`, `discard`) |
| `delete_entry` | Permanently remove a hot buffer entry |
| `build_context` | Token-budgeted prompt context assembly |
| `compact` | Trigger hot→warm→cold compaction |
| `memory_stats` | Counts and sizes across tiers |
| `list_entries` | List recent hot buffer entries with scores |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  LLM Chat Loop                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  LLMIntegration                                       │  │
│  │  ● system prompt assembly    ● tool schema generation │  │
│  └───────────────┬───────────────────────────────────────┘  │
│                  │                                           │
│  ┌───────────────▼───────────────────────────────────────┐  │
│  │  TieredMemoryManager (orchestrator)                   │  │
│  │  ● add_user_message / add_assistant_message           │  │
│  │  ● build_context_for_llm  (ranked by relevance)       │  │
│  │  ● compact_hot_to_warm    (auto-cascade to cold)      │  │
│  │  ● maybe_offload          (eviction + distillation)   │  │
│  │  ● export_session / import_session                    │  │
│  └────┬──────────┬──────────┬────────────────────────────┘  │
│       │          │          │                                │
│  ┌────▼───┐ ┌───▼────┐ ┌──▼──────┐                         │
│  │  HOT   │ │  WARM  │ │  COLD   │                         │
│  │ buffer │ │ store  │ │ store   │                         │
│  │(recent │ │(summar-│ │(indexed │                         │
│  │ raw)   │ │ ized)  │ │ search) │                         │
│  └────────┘ └────────┘ └─────────┘                         │
│       │          │          │                                │
│  ┌────▼──────────▼──────────▼────────────────────────────┐  │
│  │  Scoring Engine                                       │  │
│  │  auto-importance · content richness · recency decay   │  │
│  │  reference count · tag bonuses · EntryIndex           │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Storage tiers

| Tier | Purpose | Backing | Latency | Retrieval |
|------|---------|---------|---------|-----------|
| **Hot** | Raw recent entries | JSON on tmpfs / disk | <1 ms | Scored index, O(k) top-k |
| **Warm** | Summarized sections | JSON on disk | <5 ms | BM25 + recency + metadata |
| **Cold** | Long-term search | JSON lexical or ChromaDB | <50 ms | Hybrid semantic + lexical |

### Relevance scoring

Every hot buffer entry receives a continuous `[0.0, 1.0]` relevance score from five weighted signals:

| Signal | Weight | Description |
|--------|--------|-------------|
| Importance | 0.25 | Auto-classified (`low`/`medium`/`high`/`critical`) via regex or manual |
| Recency | 0.25 | Exponential decay with 16-hour half-life |
| References | 0.15 | Diminishing returns on access count |
| Content richness | 0.20 | Code blocks, file paths, config values, decision language |
| Tag bonuses | 0.15 | `pin` +0.15, `decision` +0.10, `discard` −0.15 |

### Auto-importance classification

When no explicit importance is set, content is analyzed by fast regex rules (<50µs):

- **Critical**: "must not", "always", "never", "security", "vulnerability", `[pin]`
- **High**: "decision", "architecture", file paths, config key=value pairs, "model"
- **Low**: Short acknowledgements ("ok", "sounds good", "thanks", "👍")
- **Medium**: Everything else

---

## Backends

### Ollama

```python
from memory_manager import OllamaSummarizer, LLMDistiller, ensure_model

# Auto-pull the model if not available
ensure_model("qwen2.5:1.5b")

summarizer = OllamaSummarizer(model="qwen2.5:1.5b", num_gpu=0)
distiller = LLMDistiller(model="qwen2.5:1.5b", num_gpu=0)
```

### LM Studio / LocalAI / vLLM / llama.cpp

```python
from memory_manager import OpenAISummarizer, OpenAIDistiller

summarizer = OpenAISummarizer(
    model="local-model",
    endpoint="http://localhost:1234/v1/chat/completions",
)
distiller = OpenAIDistiller(
    model="local-model",
    endpoint="http://localhost:1234/v1/chat/completions",
)
```

### Heuristic (zero-dependency, always available)

```python
from memory_manager import HeuristicSummarizer, HeuristicDistiller

summarizer = HeuristicSummarizer()
distiller = HeuristicDistiller()  # TF-IDF sentence scoring, ~0ms
```

---

## Distillation Patterns

The background worker processes conversation turns into structured summaries:

```python
from memory_manager import (
    CascadeDistiller, HeuristicDistiller,
    OpenAIDistiller, SpeculativeDistiller, TieredMemoryManager,
)

# Cascade: try LLM first, fall back to heuristic on failure.
distiller = CascadeDistiller([
    OpenAIDistiller(model="qwen2.5:1.5b"),
    HeuristicDistiller(),
])

# Speculative: instant heuristic draft, async LLM refinement.
distiller = SpeculativeDistiller(
    draft=HeuristicDistiller(),
    refiner=OpenAIDistiller(model="qwen2.5:3b"),
)

with TieredMemoryManager("my-session", distiller=distiller) as manager:
    manager.add_user_message("Plan the database migration.")
    manager.add_assistant_message("I'll create the migration scripts.")
```

---

## Embedding Setup

For semantic cold store retrieval, provide an embedding function:

```python
from memory_manager import TieredMemoryManager, make_ollama_embed_fn, ensure_embedding_model

# Auto-pull the embedding model
ensure_embedding_model("nomic-embed-text")

with TieredMemoryManager(
    session_id="my-session",
    embed_fn=make_ollama_embed_fn("nomic-embed-text"),
) as manager:
    # Cold store now uses hybrid semantic + lexical retrieval
    # (0.7 × semantic + 0.3 × lexical blended scoring)
    ...
```

Without `embed_fn`, cold retrieval uses pure lexical cosine similarity — still effective for keyword-heavy queries, zero dependencies.

**Recommended embedding models** (via Ollama):

| Model | Dimensions | Speed | Notes |
|-------|-----------|-------|-------|
| `nomic-embed-text` | 768 | Fast | Best balance (default) |
| `all-minilm` | 384 | Very fast | Lightweight |
| `mxbai-embed-large` | 1024 | Slower | Highest quality |

---

## Session Management

### Export / Import

```python
with TieredMemoryManager("my-session") as manager:
    # Add some data...
    manager.add_user_message("Important: API key rotation schedule is quarterly.")

    # Export to a portable archive
    archive = manager.export_session("backups/my-session")
    # → backups/my-session.tar.gz

# Later, or on a different machine:
with TieredMemoryManager("my-session") as manager:
    manager.import_session("backups/my-session.tar.gz")
```

### Entry tagging

```python
# Pin an entry to protect it from eviction and boost its score.
manager.hot_buffer.tag_entry(entry_id, "pin")

# Archive (lower priority, eligible for cold cascade).
manager.hot_buffer.tag_entry(entry_id, "archive")

# Discard (negative score modifier, evicted first).
manager.hot_buffer.tag_entry(entry_id, "discard")
```

### Ranked retrieval

```python
ranked = manager.hot_buffer.get_entries_ranked(
    limit=20,
    tags_filter=["pin", "decision"],   # only pinned/decision entries
    importance_min="high",              # high or critical only
    phase="planning",                   # filter by project phase
)
```

---

## Security

Optional HMAC-SHA256 tamper protection for all stored files:

```python
with TieredMemoryManager(
    session_id="secure-session",
    secure=True,
) as manager:
    # All reads/writes are now signed and verified.
    # Tampered files raise TamperDetectedError.
    manager.add_user_message("API key: sk-abc123")
```

With Fernet encryption (`pip install "llm-memory-manager[encrypt]"`):

```python
with TieredMemoryManager(
    session_id="encrypted-session",
    secure=True,
    encrypt=True,
) as manager:
    # Files are encrypted at rest with Fernet symmetric encryption.
    manager.add_user_message("Secret: production database password is ...")
```

---

## Model Auto-Pull

Automatically download models from Ollama when they're not available locally:

```python
from memory_manager import ensure_model, ensure_embedding_model

# Checks Ollama, pulls if missing. No-op if already available.
ensure_model("qwen2.5:1.5b")
ensure_embedding_model("nomic-embed-text")

# With progress callback
ensure_model("qwen2.5:3b", on_progress=lambda msg: print(msg))
```

**Recommended distillation models:**

| Model | Params | Use case |
|-------|--------|----------|
| `qwen2.5:1.5b` | 1.5B | Fast, good quality/speed ratio (default) |
| `qwen2.5:3b` | 3B | Better quality, fits in 4 GB RAM |
| `phi-3:mini` | 3.8B | Strong reasoning |
| `gemma2:2b` | 2B | Efficient |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_MANAGER_HOME` | `~/.local/share/memory_manager` | Base storage directory for all tiers |
| `MEMORY_MANAGER_HOT_PATH` | `/dev/shm/memory_manager` (Linux) | Hot buffer path — use tmpfs for speed |

---

## CLI Reference

```bash
# Start MCP server
memory-manager-mcp --session my-project [--project "My App"] [--secure] [--transport stdio|sse]

# Or as a Python module
python -m memory_manager.mcp_server --session my-project

# Run examples
python examples/basic_flow.py
python examples/ollama_chat.py "What's the status?" --model qwen2.5:7b
python examples/lmstudio_chat.py "Summarize decisions." --model local-model
python examples/openai_compat_chat.py "Next steps?" --host http://localhost:8080/v1
```

---

## Project Structure

```
memory_manager/
├── __init__.py            # Public API exports
├── __main__.py            # python -m memory_manager.mcp_server entry
├── active_context.py      # Token-budgeted context assembly
├── background_worker.py   # Distiller protocol + worker thread
├── cold_store.py          # Tier 3: hybrid semantic + lexical search
├── cold_store_chroma.py   # Tier 3: ChromaDB vector adapter
├── embeddings.py          # Embedding function factories (Ollama, sidecar)
├── eviction.py            # Score-based eviction manager
├── hot_buffer.py          # Tier 1: raw entries + scoring index
├── mcp_server.py          # MCP server (8 tools, stdio/sse transport)
├── model_utils.py         # Model auto-pull helpers (Ollama)
├── models.py              # Data classes (MemoryEntry, MemorySession, etc.)
├── ollama_integration.py  # LLMIntegration adapter
├── openai_summarizer.py   # OpenAI-compatible summarizer
├── orchestrator.py        # TieredMemoryManager (central coordinator)
├── scoring.py             # 5-signal relevance scoring + EntryIndex
├── security.py            # HMAC-SHA256 + Fernet encryption
├── storage.py             # Atomic JSON I/O + token estimation
├── summarizer.py          # HeuristicSummarizer + OllamaSummarizer
├── trigger.py             # Offload trigger conditions
└── warm_store.py          # Tier 2: BM25-scored summarized sections
```

---

## Comparable Tools

| Tool | What it does | How memory_manager differs |
|------|-------------|---------------------------|
| **mem0** | Cloud-hosted memory for LLM apps | Local-first, zero-dependency, file-backed |
| **Zep** | Server-side conversation memory | Library, not a server — embed in your code |
| **LangChain Memory** | In-process memory for chains | Single-tier, no scoring, no persistence |
| **LlamaIndex** | Document indexing + RAG | Different scope — this manages *conversation* memory |
| **Letta (MemGPT)** | Agent with self-managing memory | Closest cousin, but Letta is a full framework |

**Unique position:** the only *library-weight*, *local-first*, *zero-dependency* tiered memory system with importance-aware scoring and multi-backend distillation.

---

## License

MIT — see [LICENSE](LICENSE).
