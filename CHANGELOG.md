# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-11-16

Context extension becomes the core, slot-in component; persistence becomes opt-in.

### Added
- **Ephemeral mode (now the default).** `MemoryDocStore` keeps every tier in process memory: no files, no autosave thread, wiped on `close()`. Selected by default, or with `TieredMemoryManager.ephemeral()`.
- **Persistent mode is opt-in** via `persist=True`, `base_paths`, or `TieredMemoryManager.persistent(path)`. `promote(path)` moves a live ephemeral session to disk; `wipe()` deletes a session from every tier.
- `backends.py`: `DocStore` protocol with `FileDocStore` (wraps `atomic_json_write`/`read_json` and the security layer) and `MemoryDocStore`. All tiers and session state go through it.
- **`ContextVirtualizer`** (messages in, messages out) for hosts that own their history: fingerprint-deduplicated mirroring, regenerate/edit/truncation/tool-call handling, turn-aligned offloads decided by `OffloadTrigger` scoped to the history budget, a prompt layout that keeps the prefix stable for KV caching, and retrieval injected at the latest user turn.
- **OpenAI-compatible proxy** (`memory-manager-proxy` / `python -m memory_manager.proxy`): streaming and non-streaming, pass-through routes, auto per-conversation sessions, idle reaping, `--persist`, `--secure`, `--embed-model`.
- `TieredMemoryManager.add_message(role, ..., auto_offload=)` and `offload_entries(...)`; `HotBufferManager.tag_entries/get_entries_by_id/clear`; `WarmSection.raw_content`.
- MCP server `--ephemeral` flag; `examples/chat_repl.py`.

### Changed
- **Breaking:** a manager constructed without `base_paths` is ephemeral. Pass `persist=True` for the 0.1.0 behaviour. The MCP server and the single-shot chat examples opt in explicitly.
- Warm sections store full `raw_content`; cold indexing uses it (and recorded chunking), so the cold tier rebuilds from warm at full fidelity.
- Cold chunking splits on message/paragraph boundaries and keeps blocks verbatim (shared with the Chroma adapter).
- Lexical cold scoring uses sublinear TF (1 + ln tf), and query stopwords are dropped.

### Fixed
- Chunker emitted a redundant trailing chunk duplicating the end of the previous one (450 words at size 500 gave `[450, 50]`).
- Tokenizer indexed punctuation-only tokens as empty strings.
- `import_session` left the pinned context rendering the pre-import session.
- `wipe()` followed by `close()` no longer writes `session.json` back.

## [0.1.0] — 2026-08-24

Initial release.

### Added

#### Core Architecture
- Three-tier memory system: **hot** (raw recent entries) → **warm** (summarized sections) → **cold** (indexed for search)
- Automatic hot→warm→cold cascading on compaction — no manual indexing required
- Token-budgeted prompt assembly via `build_context_for_llm()` with score-ranked entries

#### Relevance Scoring (`scoring.py`)
- 5-signal composite relevance score (`[0.0, 1.0]`): importance (0.25), recency (0.25), reference count (0.15), content richness (0.20), tag bonuses (0.15)
- Auto-importance classification from content — regex-based, <50µs, no model required
  - `critical`: "must not", "always", "never", "security", "vulnerability", `[pin]`
  - `high`: "decision", "architecture", file paths, config key=value pairs
  - `low`: short acknowledgements ("ok", "sounds good", "thanks", "👍")
- In-memory `EntryIndex` with O(1) lookups by tag/importance/phase and O(k) top-k retrieval
- Eviction protects `critical`/`high` importance entries automatically

#### Background Distillation (`background_worker.py`)
- `HeuristicDistiller` — TF-IDF extractive, ~0ms, zero dependencies, always available
- `LLMDistiller` — Ollama-backed incremental distillation with CPU-only default
- `OpenAIDistiller` — any `/v1/chat/completions` server (LM Studio, LocalAI, vLLM, llama.cpp)
- `CascadeDistiller` — try each distiller in order, return first non-empty result
- `SpeculativeDistiller` — instant heuristic draft + async LLM refinement callback
- Instant distillers run inline (no thread overhead); slow distillers use a daemon thread

#### OpenAI-Compatible Backend Support
- `OpenAISummarizer` — calls `/v1/chat/completions` (LM Studio, LocalAI, vLLM, llama.cpp server)
- `OllamaSummarizer` — calls Ollama's `/api/chat` with `num_gpu` control
- `HeuristicSummarizer` — deterministic fallback, no AI required
- `LLMIntegration` (née `OllamaIntegration`) — backend-agnostic prompt assembly and tool schemas

#### Storage
- `HotBufferManager` — JSON-backed, prefers Linux tmpfs (`/dev/shm`) for speed
- `WarmStoreManager` — atomic JSON sections with relational `source_entry_ids`
- `ColdStoreManager` — lexical cosine-TF chunked index, zero dependencies
- `ChromaColdStore` — optional ChromaDB adapter for vector similarity search (`pip install .[chroma]`)
- All stores use `atomic_json_write` with `fsync` and `os.replace` for crash safety

#### Security
- Optional HMAC-SHA256 tamper protection on all stored files
- Optional Fernet encryption (requires `pip install .[encrypt]`)
- Tampered cold chunks are silently skipped during retrieval to prevent prompt injection
- Files written with mode `0600`

#### Miscellaneous
- `OffloadTrigger` — composite signal: token pressure + semantic drift + structural seam
- `EmbeddingService` sidecar + direct Ollama embedding factory
- `SecurityManager` with `keyring` integration for key storage
- `MemoryEntry.importance` and `.phase` fields for fine-grained indexing
- Session persistence with autosave timer; resumes across process restarts

### Notes
- Zero required dependencies — all AI backends are optional
- `MEMORY_MANAGER_HOME` env var overrides storage location
- `MEMORY_MANAGER_HOT_PATH` env var overrides hot buffer path (point at tmpfs)
- `OllamaIntegration` is preserved as a backward-compatible alias for `LLMIntegration`
