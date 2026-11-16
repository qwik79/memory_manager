"""MCP server exposing memory_manager as tools for LLM-integrated editors.

Start the server:

    python -m memory_manager.mcp_server --session my-project

Or add to your MCP client config (e.g. Claude Code ``mcp.json``)::

    {
        "servers": {
            "memory-manager": {
                "command": "python",
                "args": ["-m", "memory_manager.mcp_server", "--session", "my-project"]
            }
        }
    }

Tools
-----
- **read_memory** — Ranked retrieval across all tiers.
- **add_memory** — Add an entry to the hot buffer.
- **tag_entry** — Tag a hot buffer entry (pin, decision, archive, discard).
- **delete_entry** — Permanently remove a hot buffer entry.
- **build_context** — Token-budgeted prompt context assembly.
- **compact** — Trigger hot→warm→cold compaction.
- **memory_stats** — Counts and sizes across tiers.
- **list_entries** — List recent hot buffer entries.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

# MCP SDK is an optional dependency — import guarded.
try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    print(
        "The MCP SDK is required for the memory manager MCP server.\n"
        "Install it with:  pip install 'llm-memory-manager[mcp]'",
        file=sys.stderr,
    )
    sys.exit(1)

from .orchestrator import TieredMemoryManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

def create_server(
    session_id: str = "default",
    project_name: str = "",
    secure: bool = False,
    ephemeral: bool = False,
) -> MCPServer:
    """Create an MCPServer with memory_manager tools registered.

    Parameters
    ----------
    session_id:
        Persistent session identifier — reusing the same ID resumes prior state.
    project_name:
        Human-readable project label.
    secure:
        Enable HMAC-SHA256 tamper protection on stored files.
    ephemeral:
        Keep memory in RAM only for the server's lifetime (nothing on disk).
        By default the server is persistent, since tool memory usually spans
        editor sessions.
    """
    mcp = MCPServer("memory-manager")

    # The manager is created once and shared across all tool calls.
    manager = TieredMemoryManager(
        session_id=session_id,
        project_name=project_name,
        secure=secure,
        persist=not ephemeral,
    )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @mcp.tool()
    def read_memory(query: str, limit: int = 10) -> str:
        """Search memory across all tiers (hot, warm, cold) and return ranked results.

        Use this to find previously discussed decisions, file paths, config values,
        or any other information from earlier in the conversation.

        Args:
            query: What to search for — a question, keyword, or topic.
            limit: Maximum number of results to return (default 10).
        """
        results: list[dict[str, Any]] = []

        # Hot buffer — scored and ranked.
        ranked = manager.hot_buffer.get_entries_ranked(limit=limit)
        for entry in ranked:
            score = manager.hot_buffer.get_entry_score(entry.get("entry_id", ""))
            results.append({
                "tier": "hot",
                "entry_id": entry.get("entry_id"),
                "role": entry.get("entry_type"),
                "content": entry.get("content", "")[:500],
                "importance": entry.get("importance", "medium"),
                "tags": entry.get("tags", []),
                "score": round(score, 3),
            })

        # Warm + cold — retrieved by query relevance.
        retrieved = manager.refresh_context(query, k=limit)
        for result in retrieved:
            results.append({
                "tier": result.source.split("://")[0] if "://" in result.source else "unknown",
                "id": result.identifier,
                "content": result.content[:500],
                "relevance": round(result.relevance, 3),
                "metadata": result.metadata,
            })

        return json.dumps(results, indent=2, default=str)

    @mcp.tool()
    def add_memory(
        content: str,
        role: str = "user",
        importance: str = "medium",
        tags: str = "",
        phase: str = "",
    ) -> str:
        """Add a new memory entry to the hot buffer.

        Use this to record important decisions, file paths, config values,
        or any information that should be remembered for later.

        Args:
            content: The text content to remember.
            role: Either 'user' or 'assistant' (default 'user').
            importance: One of 'low', 'medium', 'high', 'critical' (auto-detected if 'medium').
            tags: Comma-separated tags, e.g. 'decision,pin'.
            phase: Optional project phase, e.g. 'planning', 'implementation'.
        """
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

        if role == "assistant":
            entry_id = manager.add_assistant_message(
                content, tags=tag_list, importance=importance, phase=phase,
            )
        else:
            entry_id = manager.add_user_message(
                content, tags=tag_list, importance=importance, phase=phase,
            )
        return json.dumps({"status": "ok", "entry_id": entry_id})

    @mcp.tool()
    def tag_entry(entry_id: str, tag: str) -> str:
        """Tag a hot buffer entry.

        Common tags:
        - 'pin' — protect from eviction, boost retrieval score.
        - 'decision' — mark as an important decision.
        - 'archive' — lower priority, eligible for cold cascade.
        - 'discard' — negative score modifier, evicted first.

        Args:
            entry_id: The ID of the entry to tag (from read_memory or add_memory).
            tag: The tag to apply.
        """
        manager.hot_buffer.tag_entry(entry_id, tag)
        return json.dumps({"status": "ok", "entry_id": entry_id, "tag": tag})

    @mcp.tool()
    def delete_entry(entry_id: str) -> str:
        """Permanently remove a hot buffer entry.

        Use sparingly — prefer tagging with 'discard' for soft removal.

        Args:
            entry_id: The ID of the entry to delete.
        """
        removed = manager.hot_buffer.delete_entry(entry_id)
        return json.dumps({"status": "ok" if removed else "not_found", "entry_id": entry_id})

    @mcp.tool()
    def build_context(query: str = "", max_tokens: int = 32000) -> str:
        """Build a token-budgeted context block from memory for use in a prompt.

        Returns a formatted string containing pinned project state plus the
        highest-scoring recent and retrieved entries, all within the token budget.

        Args:
            query: Optional query to bias retrieval toward specific topics.
            max_tokens: Maximum token budget for the context block (default 32000).
        """
        return manager.build_context_for_llm(query=query, max_tokens=max_tokens)

    @mcp.tool()
    def compact() -> str:
        """Trigger hot→warm→cold compaction.

        Summarizes recent hot buffer entries into a warm section, applies
        distillation to update the pinned summary, and auto-cascades to
        the cold index for long-term searchability.

        Call this when the conversation reaches a natural breakpoint or when
        memory_stats shows a high hot entry count.
        """
        section_id = manager.compact_hot_to_warm()
        return json.dumps({
            "status": "compacted" if section_id else "nothing_to_compact",
            "warm_section_id": section_id,
        })

    @mcp.tool()
    def memory_stats() -> str:
        """Return counts and sizes across all memory tiers.

        Useful for understanding how much memory is being used and when
        compaction might be beneficial.
        """
        hot_entries = manager.hot_buffer.get_entries(limit=None)
        hot_tokens, hot_bytes = manager.hot_buffer.get_size_estimate()
        warm_sections = manager.warm_store.list_sections(manager.session_id)
        cold_dir = manager.cold_store._index_dir(manager.session_id)
        cold_chunks = len(list(cold_dir.glob("*.json"))) if cold_dir.exists() else 0

        stats = {
            "session_id": manager.session_id,
            "project_name": manager.session.project_name,
            "hot": {
                "entries": len(hot_entries),
                "estimated_tokens": hot_tokens,
                "estimated_bytes": hot_bytes,
            },
            "warm": {
                "sections": len(warm_sections),
                "indexed": sum(1 for s in warm_sections if s.get("indexed_at")),
            },
            "cold": {
                "chunks": cold_chunks,
            },
            "pinned_summary_length": len(manager.session.pinned_summary),
            "active_decisions": len(manager.session.active_decisions),
            "open_questions": len(manager.session.open_questions),
        }
        return json.dumps(stats, indent=2)

    @mcp.tool()
    def list_entries(limit: int = 20, importance_min: str = "") -> str:
        """List recent hot buffer entries, optionally filtered by importance.

        Args:
            limit: Maximum entries to return (default 20).
            importance_min: Minimum importance level: 'low', 'medium', 'high', or 'critical'.
        """
        entries = manager.hot_buffer.get_entries_ranked(
            limit=limit,
            importance_min=importance_min or None,
        )
        result = []
        for e in entries:
            result.append({
                "entry_id": e.get("entry_id"),
                "role": e.get("entry_type"),
                "importance": e.get("importance", "medium"),
                "tags": e.get("tags", []),
                "content_preview": e.get("content", "")[:200],
                "score": round(manager.hot_buffer.get_entry_score(e.get("entry_id", "")), 3),
            })
        return json.dumps(result, indent=2, default=str)

    # Register a shutdown hook so the manager's session is saved.
    import atexit
    atexit.register(manager.close)

    return mcp


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for ``python -m memory_manager.mcp_server``."""
    parser = argparse.ArgumentParser(
        description="Memory Manager MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--session", "-s",
        default="default",
        help="Session ID — reusing the same ID resumes prior state (default: 'default').",
    )
    parser.add_argument(
        "--project", "-p",
        default="",
        help="Human-readable project name.",
    )
    parser.add_argument(
        "--secure",
        action="store_true",
        help="Enable HMAC-SHA256 tamper protection on stored files.",
    )
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="Keep memory in RAM only; nothing is written to disk and it is gone when the server exits.",
    )
    parser.add_argument(
        "--transport", "-t",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport (default: stdio).",
    )
    args = parser.parse_args()

    # Configure logging to stderr so it doesn't corrupt stdio JSON-RPC.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    server = create_server(
        session_id=args.session,
        project_name=args.project,
        secure=args.secure,
        ephemeral=args.ephemeral,
    )

    logger.info(
        "Starting Memory Manager MCP server (session=%s, transport=%s)",
        args.session,
        args.transport,
    )
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
