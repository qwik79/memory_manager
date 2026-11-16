"""Tests for the quality-pass changes: token estimation, cold store embeddings,
warm store BM25 scoring, session path dedup, and MCP server tools.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------
from memory_manager.storage import estimate_tokens


class TestEstimateTokens:
    """Verify the code-aware word-based token estimator."""

    def test_empty_returns_overhead_only(self):
        assert estimate_tokens("", overhead=10) == 10

    def test_prose_uses_lower_multiplier(self):
        text = "This is a simple sentence with no code at all."
        tokens = estimate_tokens(text)
        # 10 words × 1.33 ≈ 13 tokens
        assert 12 <= tokens <= 16

    def test_code_uses_higher_multiplier(self):
        text = "import os\nimport sys\ndef foo():\n    return bar()\nclass Baz:\n    pass"
        tokens = estimate_tokens(text)
        # 10 words × 1.5 = 15, plus specials
        assert tokens > estimate_tokens("This is ten plain words with no code here today.")

    def test_special_characters_add_weight(self):
        # {} [] () all contribute extra tokens.
        text_with_specials = "a = {x: [1, 2], y: (3, 4)}"
        text_without = "a equals x one two y three four"
        assert estimate_tokens(text_with_specials) >= estimate_tokens(text_without)

    def test_overhead_is_additive(self):
        base = estimate_tokens("Hello world")
        with_overhead = estimate_tokens("Hello world", overhead=100)
        assert with_overhead == base + 100


# ---------------------------------------------------------------------------
# Cold store — embedding-backed retrieval
# ---------------------------------------------------------------------------
from memory_manager.cold_store import ColdStoreManager, _cosine_dense


class TestColdStoreEmbeddings:
    """Verify hybrid lexical + semantic cold store retrieval."""

    @pytest.fixture()
    def cold_dir(self, tmp_path):
        return tmp_path / "cold"

    def _fake_embed_fn(self, text: str) -> list[float]:
        """Deterministic embedding: hash each word into a 4-dim vector."""
        vec = [0.0] * 4
        for word in text.lower().split():
            h = hash(word) % 997
            vec[h % 4] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def test_index_stores_embedding_when_embed_fn_provided(self, cold_dir):
        store = ColdStoreManager(cold_dir, embed_fn=self._fake_embed_fn)
        ids = store.index_section("s1", "test", "Hello world from the cold store")
        assert len(ids) >= 1
        chunk_path = cold_dir / "sessions" / "s1" / "chunks" / f"{ids[0]}.json"
        with open(chunk_path) as f:
            data = json.load(f)
        assert "embedding" in data
        assert len(data["embedding"]) == 4

    def test_index_without_embed_fn_has_no_embedding(self, cold_dir):
        store = ColdStoreManager(cold_dir)
        ids = store.index_section("s1", "test", "Hello world from the cold store")
        chunk_path = cold_dir / "sessions" / "s1" / "chunks" / f"{ids[0]}.json"
        with open(chunk_path) as f:
            data = json.load(f)
        assert "embedding" not in data

    def test_retrieve_uses_semantic_when_available(self, cold_dir):
        store = ColdStoreManager(cold_dir, embed_fn=self._fake_embed_fn)
        store.index_section("s1", "db", "PostgreSQL schema migration database")
        store.index_section("s1", "weather", "The weather today is sunny and warm")

        results = store.retrieve("s1", "database schema", k=5, min_relevance=0.01)
        # The database-related chunk should rank higher than the weather chunk.
        assert len(results) >= 1
        assert "database" in results[0].content.lower() or "postgresql" in results[0].content.lower()

    def test_retrieve_falls_back_to_lexical_for_old_chunks(self, cold_dir):
        # Index without embeddings, then enable embed_fn for retrieval.
        store_no_embed = ColdStoreManager(cold_dir)
        store_no_embed.index_section("s1", "test", "PostgreSQL database migration schema")

        store_with_embed = ColdStoreManager(cold_dir, embed_fn=self._fake_embed_fn)
        results = store_with_embed.retrieve("s1", "database migration", k=5)
        # Should still find the chunk via lexical fallback.
        assert len(results) >= 1

    def test_embedding_failure_does_not_crash_indexing(self, cold_dir):
        def broken_embed(text: str) -> list[float]:
            raise RuntimeError("embedding service down")

        store = ColdStoreManager(cold_dir, embed_fn=broken_embed)
        ids = store.index_section("s1", "test", "This should still index")
        assert len(ids) >= 1

    def test_schema_version_is_2(self, cold_dir):
        store = ColdStoreManager(cold_dir, embed_fn=self._fake_embed_fn)
        ids = store.index_section("s1", "test", "Hello")
        chunk_path = cold_dir / "sessions" / "s1" / "chunks" / f"{ids[0]}.json"
        with open(chunk_path) as f:
            data = json.load(f)
        assert data["schema_version"] == 2


class TestCosineDense:
    """Verify the dense cosine similarity helper."""

    def test_identical_vectors(self):
        assert _cosine_dense([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_dense([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        assert _cosine_dense([], []) == 0.0

    def test_different_lengths_returns_zero(self):
        assert _cosine_dense([1.0], [1.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# Warm store — BM25-inspired scoring
# ---------------------------------------------------------------------------
from memory_manager.warm_store import WarmStoreManager


class TestWarmStoreBM25:
    """Verify the BM25-inspired warm store scorer."""

    @pytest.fixture()
    def warm_store(self, tmp_path):
        return WarmStoreManager(tmp_path / "warm")

    def test_decision_sections_rank_higher(self, warm_store):
        """Sections with decisions metadata should rank higher than plain ones."""
        # Store two sections — one with decisions, one without.
        warm_store.store_section(
            "s1", "with_decision", "We decided to use PostgreSQL.",
            metadata={"decisions": ["Use PostgreSQL"]},
        )
        warm_store.store_section(
            "s1", "plain", "We discussed various options today.",
        )
        results = warm_store.retrieve_sections("s1", query="PostgreSQL", k=5, relevance_threshold=0.0)
        # The decision section should rank first.
        assert len(results) >= 2
        assert "decision" in results[0].metadata.get("section_name", results[0].content).lower() or \
               results[0].metadata.get("decisions") is not None

    def test_recency_affects_score(self, warm_store):
        """Recent sections should score higher than old ones, all else being equal."""
        # Store a "recent" section.
        sid1 = warm_store.store_section("s1", "recent", "database schema migration")
        # Store an "old" section by manipulating created_at.
        sid2 = warm_store.store_section("s1", "old", "database schema migration")
        # Manually age the second section by rewriting its created_at.
        path = warm_store._section_dir("s1") / f"{sid2}.json"
        from memory_manager.storage import read_json, atomic_json_write
        data = read_json(path)
        data["created_at"] = time.time() - 86400 * 7  # 7 days ago
        atomic_json_write(path, data)

        results = warm_store.retrieve_sections("s1", query="database schema", k=5, relevance_threshold=0.0)
        assert len(results) >= 2
        # The recent section should score higher.
        assert results[0].identifier == sid1

    def test_no_query_returns_recency_weighted(self, warm_store):
        """With an empty query, sections should still be retrievable by recency."""
        warm_store.store_section("s1", "test", "Some content here.",
                                 metadata={"decisions": ["important"]})
        results = warm_store.retrieve_sections("s1", query="", k=5, relevance_threshold=0.0)
        assert len(results) >= 1

    def test_session_dir_is_public(self, warm_store):
        """The session_dir() method should be accessible (no leading underscore)."""
        path = warm_store.session_dir("my-project")
        assert "my-project" in str(path)
        assert isinstance(path, Path)


# ---------------------------------------------------------------------------
# Orchestrator — session path dedup
# ---------------------------------------------------------------------------
from memory_manager.orchestrator import TieredMemoryManager


class TestSessionPathDedup:
    """Verify that the orchestrator delegates session path to warm_store."""

    def test_session_path_uses_warm_store_dir(self, tmp_path):
        warm_path = tmp_path / "warm"
        with TieredMemoryManager(
            session_id="dedup-test",
            base_paths={"warm": str(warm_path), "hot": str(tmp_path / "hot")},
        ) as manager:
            # The session path should be under the warm store's session dir.
            warm_session_dir = manager.warm_store.session_dir("dedup-test")
            assert str(manager._session_path).startswith(str(warm_session_dir))


# ---------------------------------------------------------------------------
# Cold store — embed_fn passed from orchestrator
# ---------------------------------------------------------------------------
class TestOrchestratorPassesEmbedFn:
    """Verify embed_fn flows through to cold store."""

    def test_embed_fn_reaches_cold_store(self, tmp_path):
        calls = []
        def tracking_embed(text: str) -> list[float]:
            calls.append(text)
            return [0.1, 0.2, 0.3]

        with TieredMemoryManager(
            session_id="embed-test",
            embed_fn=tracking_embed,
            base_paths={
                "hot": str(tmp_path / "hot"),
                "warm": str(tmp_path / "warm"),
                "cold": str(tmp_path / "cold"),
            },
        ) as manager:
            assert manager.cold_store._embed_fn is tracking_embed


# ---------------------------------------------------------------------------
# MCP server — tool registration and execution
# ---------------------------------------------------------------------------
class TestMCPServerTools:
    """Verify that the MCP server tools work correctly."""

    @pytest.fixture()
    def server_and_manager(self, tmp_path):
        """Create an MCP server with a temp-based manager."""
        from memory_manager.mcp_server import create_server
        os.environ["MEMORY_MANAGER_HOME"] = str(tmp_path / "mm_home")
        os.environ["MEMORY_MANAGER_HOT_PATH"] = str(tmp_path / "hot")
        try:
            server = create_server(session_id="mcp-test", project_name="MCP Test")
            yield server
        finally:
            os.environ.pop("MEMORY_MANAGER_HOME", None)
            os.environ.pop("MEMORY_MANAGER_HOT_PATH", None)

    def _call_tool(self, server, name: str, **kwargs) -> dict | list | str:
        """Invoke a tool function by name and parse the JSON result."""
        import asyncio
        import inspect

        tool_fn = None
        for tool in server._tool_manager._tools.values():
            if tool.name == name:
                tool_fn = tool.fn
                break
        assert tool_fn is not None, f"Tool '{name}' not found"
        result = tool_fn(**kwargs)
        # Handle both sync and async tool functions.
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result

    def test_add_and_read_memory(self, server_and_manager):
        server = server_and_manager
        # Add a memory entry.
        add_result = self._call_tool(server, "add_memory", content="Decision: use Redis for caching")
        assert add_result["status"] == "ok"
        assert "entry_id" in add_result

        # Read it back.
        read_result = self._call_tool(server, "read_memory", query="caching")
        assert isinstance(read_result, list)
        assert any("Redis" in r.get("content", "") or "redis" in r.get("content", "").lower()
                    for r in read_result)

    def test_tag_entry(self, server_and_manager):
        server = server_and_manager
        add_result = self._call_tool(server, "add_memory", content="Important config value")
        entry_id = add_result["entry_id"]

        tag_result = self._call_tool(server, "tag_entry", entry_id=entry_id, tag="pin")
        assert tag_result["status"] == "ok"
        assert tag_result["tag"] == "pin"

    def test_delete_entry(self, server_and_manager):
        server = server_and_manager
        add_result = self._call_tool(server, "add_memory", content="Temporary note")
        entry_id = add_result["entry_id"]

        del_result = self._call_tool(server, "delete_entry", entry_id=entry_id)
        assert del_result["status"] == "ok"

        # Delete again should return not_found.
        del_result2 = self._call_tool(server, "delete_entry", entry_id=entry_id)
        assert del_result2["status"] == "not_found"

    def test_memory_stats(self, server_and_manager):
        server = server_and_manager
        self._call_tool(server, "add_memory", content="Test entry for stats")

        stats = self._call_tool(server, "memory_stats")
        assert stats["session_id"] == "mcp-test"
        assert stats["hot"]["entries"] >= 1
        assert "warm" in stats
        assert "cold" in stats

    def test_compact(self, server_and_manager):
        server = server_and_manager
        self._call_tool(server, "add_memory", content="Entry to compact")
        self._call_tool(server, "add_memory", content="Another entry", role="assistant")

        result = self._call_tool(server, "compact")
        assert result["status"] in ("compacted", "nothing_to_compact")

    def test_build_context(self, server_and_manager):
        server = server_and_manager
        self._call_tool(server, "add_memory", content="We use PostgreSQL for the database.")

        context = self._call_tool(server, "build_context", query="database")
        assert isinstance(context, str)
        assert len(context) > 0

    def test_list_entries(self, server_and_manager):
        server = server_and_manager
        self._call_tool(server, "add_memory", content="First entry")
        self._call_tool(server, "add_memory", content="Security: never expose API keys", role="assistant")

        entries = self._call_tool(server, "list_entries", limit=10)
        assert isinstance(entries, list)
        assert len(entries) >= 2

    def test_all_tools_registered(self, server_and_manager):
        server = server_and_manager
        tool_names = {t.name for t in server._tool_manager._tools.values()}
        expected = {
            "read_memory", "add_memory", "tag_entry", "delete_entry",
            "build_context", "compact", "memory_stats", "list_entries",
        }
        assert expected == tool_names
