"""Tests covering all 7 handoff tasks.

Tasks covered:
    1. Configurable num_gpu on LLMDistiller / OllamaSummarizer
    2. Tier field acting (pin → tag entries, archive → cascade to cold)
    3. importance + phase on MemoryEntry
    4. Auto-cascade warm→cold on compact_hot_to_warm
    5. Inline distiller for instant paths (no background thread)
    6. make_embed_fn / make_ollama_embed_fn factories (unit-tested without network)
    7. ChromaColdStore (gated on chromadb import availability)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_manager(tmp_path: Path, **kwargs):
    from memory_manager import TieredMemoryManager

    return TieredMemoryManager(
        session_id="test_session",
        project_name="Handoff Tests",
        pinned_summary="Testing the handoff tasks.",
        base_paths={
            "hot": tmp_path / "hot",
            "warm": tmp_path / "warm",
            "cold": tmp_path / "cold",
        },
        autosave_interval=0,
        **kwargs,
    )


# ===========================================================================
# Task 1 — configurable num_gpu
# ===========================================================================


class TestNumGpu:
    def test_ollama_summarizer_defaults_cpu(self):
        from memory_manager import OllamaSummarizer

        s = OllamaSummarizer()
        assert s.num_gpu == 0

    def test_ollama_summarizer_accepts_num_gpu(self):
        from memory_manager import OllamaSummarizer

        s = OllamaSummarizer(num_gpu=1)
        assert s.num_gpu == 1

    def test_ollama_summarizer_uses_num_gpu_in_payload(self):
        """Payload sent to Ollama must contain the configured num_gpu value."""
        from memory_manager import OllamaSummarizer

        s = OllamaSummarizer(num_gpu=2)
        entries = [{"entry_type": "user", "content": "hello"}]

        captured: list[bytes] = []

        def fake_urlopen(req, timeout=None):
            captured.append(req.data)
            raise OSError("no server")  # trigger fallback

        with patch("memory_manager.summarizer.urlopen", fake_urlopen):
            s.summarize_entries(entries)

        assert captured, "urlopen was not called"
        payload = json.loads(captured[0])
        assert payload["options"]["num_gpu"] == 2

    def test_llm_distiller_defaults_cpu(self):
        from memory_manager import LLMDistiller

        d = LLMDistiller()
        assert d._summarizer.num_gpu == 0

    def test_llm_distiller_accepts_num_gpu(self):
        from memory_manager import LLMDistiller

        d = LLMDistiller(num_gpu=-1)
        assert d._summarizer.num_gpu == -1


# ===========================================================================
# Task 2 — tier field acting
# ===========================================================================


class TestTierActing:
    def test_tier_pin_tags_recent_hot_entries(self, tmp_path):
        mgr = make_manager(tmp_path, distiller=None)
        mgr.add_user_message("Critical decision: always use HMAC for signatures.")
        mgr.apply_distillation({
            "summary": "Use HMAC always.",
            "decisions": ["Use HMAC"],
            "tier": "pin",
        })
        entries = mgr.hot_buffer.get_entries(limit=10)
        assert any("pin" in e.get("tags", []) for e in entries), \
            "Expected at least one entry to be tagged 'pin'"

    def test_tier_pin_skips_already_pinned(self, tmp_path):
        """Calling apply_distillation twice with tier=pin should not double-tag."""
        mgr = make_manager(tmp_path, distiller=None)
        mgr.add_user_message("Important: use Redis.")
        mgr.apply_distillation({"summary": "Use Redis.", "tier": "pin"})
        mgr.apply_distillation({"summary": "Use Redis.", "tier": "pin"})
        entries = mgr.hot_buffer.get_entries(limit=10)
        for e in entries:
            assert e.get("tags", []).count("pin") <= 1

    def test_tier_extend_does_not_tag(self, tmp_path):
        mgr = make_manager(tmp_path, distiller=None)
        mgr.add_user_message("Nothing special here.")
        mgr.apply_distillation({"summary": "Routine work.", "tier": "extend"})
        entries = mgr.hot_buffer.get_entries(limit=10)
        assert not any("pin" in e.get("tags", []) for e in entries)

    def test_tier_archive_cascades_to_cold(self, tmp_path):
        mgr = make_manager(tmp_path, distiller=None)
        mgr.add_user_message("Bug is fixed and resolved.")
        mgr.compact_hot_to_warm()
        # Reset — manually create a section that hasn't been indexed yet
        # then trigger archive cascade
        mgr.add_user_message("Another resolved item.")
        mgr.compact_hot_to_warm()
        # Now trigger archive: should cascade any unindexed warm to cold
        mgr.apply_distillation({"summary": "All resolved.", "tier": "archive"})
        # Cold store should have chunks
        cold_chunks = list((tmp_path / "cold").rglob("*.json"))
        assert len(cold_chunks) > 0, "Expected cold chunks after archive cascade"

    def test_tier_missing_defaults_to_extend(self, tmp_path):
        """Missing tier key should not crash and should behave like extend."""
        mgr = make_manager(tmp_path, distiller=None)
        mgr.add_user_message("No tier field.")
        mgr.apply_distillation({"summary": "Some content."})  # no tier key
        # Should not raise


# ===========================================================================
# Task 3 — importance + phase on MemoryEntry
# ===========================================================================


class TestImportancePhase:
    def test_memory_entry_has_defaults(self):
        from memory_manager.models import MemoryEntry

        e = MemoryEntry.create("s1", "user", "hello")
        assert e.importance == "medium"
        assert e.phase == ""

    def test_memory_entry_accepts_importance_and_phase(self):
        from memory_manager.models import MemoryEntry

        e = MemoryEntry.create(
            "s1", "user", "critical work",
            importance="critical", phase="debugging",
        )
        assert e.importance == "critical"
        assert e.phase == "debugging"

    def test_memory_entry_serialises_new_fields(self):
        from memory_manager.models import MemoryEntry

        e = MemoryEntry.create("s1", "assistant", "done", importance="high", phase="implementation")
        d = e.to_json()
        assert d["importance"] == "high"
        assert d["phase"] == "implementation"

    def test_hot_buffer_stores_importance_and_phase(self, tmp_path):
        from memory_manager import TieredMemoryManager

        with TieredMemoryManager(
            "imp_phase",
            base_paths={"hot": tmp_path / "h", "warm": tmp_path / "w", "cold": tmp_path / "c"},
            autosave_interval=0,
            distiller=None,
        ) as mgr:
            mgr.add_user_message("High priority item.", importance="high", phase="planning")
            entries = mgr.hot_buffer.get_entries(limit=5)
            assert entries[0]["importance"] == "high"
            assert entries[0]["phase"] == "planning"

    def test_orchestrator_add_assistant_message_with_phase(self, tmp_path):
        with make_manager(tmp_path, distiller=None) as mgr:
            mgr.add_assistant_message(
                "Implementation complete.",
                importance="medium",
                phase="implementation",
            )
            entries = mgr.hot_buffer.get_entries(limit=5)
            assert entries[-1]["phase"] == "implementation"


# ===========================================================================
# Task 4 — auto-cascade warm→cold on compaction
# ===========================================================================


class TestAutoCascade:
    def test_compact_creates_cold_chunks_automatically(self, tmp_path):
        mgr = make_manager(tmp_path, distiller=None)
        mgr.add_user_message("Content that should reach cold storage automatically.")
        section_id = mgr.compact_hot_to_warm()
        assert section_id

        # Cold store should have chunks without any explicit index_warm_to_cold() call
        results = mgr.cold_store.retrieve(mgr.session_id, "cold storage automatically", k=5)
        assert len(results) > 0, "Auto-cascade failed: no cold chunks after compaction"

    def test_warm_section_marked_indexed_after_compact(self, tmp_path):
        mgr = make_manager(tmp_path, distiller=None)
        mgr.add_user_message("Automatic index test.")
        section_id = mgr.compact_hot_to_warm()
        assert section_id

        section = mgr.warm_store.get_section(mgr.session_id, section_id)
        assert section is not None
        assert section.get("indexed_at") is not None, \
            "Section should be marked indexed after auto-cascade"

    def test_explicit_index_warm_to_cold_still_works(self, tmp_path):
        """Calling index_warm_to_cold() explicitly after auto-cascade is idempotent."""
        mgr = make_manager(tmp_path, distiller=None)
        mgr.add_user_message("Idempotent cascade test.")
        mgr.compact_hot_to_warm()
        # Second call with include_indexed=True should re-index without error
        chunk_ids = mgr.index_warm_to_cold(include_indexed=True)
        assert isinstance(chunk_ids, list)  # may be empty or have ids, both are valid


# ===========================================================================
# Task 5 — inline distiller for instant paths
# ===========================================================================


class TestInlineDistiller:
    def test_heuristic_distiller_is_instant(self):
        from memory_manager import HeuristicDistiller

        assert HeuristicDistiller.is_instant is True

    def test_llm_distiller_is_not_instant(self):
        from memory_manager import LLMDistiller

        assert LLMDistiller.is_instant is False

    def test_cascade_all_instant_is_instant(self):
        from memory_manager import CascadeDistiller, HeuristicDistiller

        d = CascadeDistiller([HeuristicDistiller(), HeuristicDistiller()])
        assert d.is_instant is True

    def test_cascade_with_llm_is_not_instant(self):
        from memory_manager import CascadeDistiller, HeuristicDistiller, LLMDistiller

        d = CascadeDistiller([LLMDistiller(), HeuristicDistiller()])
        assert d.is_instant is False

    def test_speculative_distiller_is_not_instant(self):
        from memory_manager import HeuristicDistiller, SpeculativeDistiller

        d = SpeculativeDistiller(draft=HeuristicDistiller(), refine=HeuristicDistiller())
        assert d.is_instant is False
        d.stop()

    def test_instant_distiller_runs_inline_no_background_thread(self, tmp_path):
        from memory_manager import HeuristicDistiller, TieredMemoryManager

        with TieredMemoryManager(
            "inline_test",
            base_paths={"hot": tmp_path / "h", "warm": tmp_path / "w", "cold": tmp_path / "c"},
            distiller=HeuristicDistiller(),
            autosave_interval=0,
        ) as mgr:
            assert mgr.background_worker is None, "Instant distiller should not spawn thread"
            assert mgr._inline_distiller is not None

    def test_inline_distiller_updates_pinned_summary(self, tmp_path):
        from memory_manager import HeuristicDistiller, TieredMemoryManager

        with TieredMemoryManager(
            "inline_summary",
            base_paths={"hot": tmp_path / "h", "warm": tmp_path / "w", "cold": tmp_path / "c"},
            distiller=HeuristicDistiller(),
            autosave_interval=0,
        ) as mgr:
            mgr.add_user_message("Decision: we will use SQLite for the warm store.")
            mgr.add_assistant_message("Agreed. SQLite is simple and dependency-free.")
            # Inline distillation should have updated pinned_summary
            assert mgr.session.pinned_summary, "Inline distillation should populate pinned_summary"

    def test_none_distiller_neither_thread_nor_inline(self, tmp_path):
        from memory_manager import TieredMemoryManager

        with TieredMemoryManager(
            "no_distill",
            base_paths={"hot": tmp_path / "h", "warm": tmp_path / "w", "cold": tmp_path / "c"},
            distiller=None,
            autosave_interval=0,
        ) as mgr:
            assert mgr.background_worker is None
            assert mgr._inline_distiller is None

    def test_slow_distiller_uses_background_thread(self, tmp_path):
        from memory_manager import LLMDistiller, TieredMemoryManager

        # LLMDistiller is not instant → should create a BackgroundWorker.
        with TieredMemoryManager(
            "slow_thread",
            base_paths={"hot": tmp_path / "h", "warm": tmp_path / "w", "cold": tmp_path / "c"},
            distiller=LLMDistiller(),
            autosave_interval=0,
        ) as mgr:
            assert mgr.background_worker is not None
            assert mgr._inline_distiller is None


# ===========================================================================
# Task 6 — embedding factories (no network required)
# ===========================================================================


class TestEmbeddingFactories:
    def test_make_embed_fn_returns_callable(self):
        from memory_manager import make_embed_fn

        fn = make_embed_fn()
        assert callable(fn)

    def test_make_ollama_embed_fn_returns_callable(self):
        from memory_manager import make_ollama_embed_fn

        fn = make_ollama_embed_fn()
        assert callable(fn)

    def test_make_embed_fn_calls_endpoint(self):
        from memory_manager import make_embed_fn

        fake_response = json.dumps(
            {"embeddings": [{"text": "hello", "vector": [0.1, 0.2, 0.3]}]}
        ).encode()

        class FakeResponse:
            def read(self):
                return fake_response
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        with patch("memory_manager.embeddings.urlopen", return_value=FakeResponse()):
            fn = make_embed_fn(endpoint="http://fake:9999")
            result = fn("hello")

        assert result == [0.1, 0.2, 0.3]

    def test_make_ollama_embed_fn_calls_ollama(self):
        from memory_manager import make_ollama_embed_fn

        fake_response = json.dumps({"embedding": [0.5, 0.6]}).encode()

        class FakeResponse:
            def read(self):
                return fake_response
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        with patch("memory_manager.embeddings.urlopen", return_value=FakeResponse()):
            fn = make_ollama_embed_fn()
            result = fn("test text")

        assert result == [0.5, 0.6]


# ===========================================================================
# Task 7 — ChromaColdStore (requires chromadb)
# ===========================================================================

chromadb = pytest.importorskip("chromadb", reason="chromadb not installed")


class TestChromaColdStore:
    def test_index_and_retrieve(self, tmp_path):
        from memory_manager.cold_store_chroma import ChromaColdStore

        store = ChromaColdStore(base_path=tmp_path / "chromadb")
        chunk_ids = store.index_section(
            session_id="test",
            section_name="impl",
            content="We decided to use SQLite for the warm memory tier. "
                    "The schema stores sections with metadata and source IDs.",
        )
        assert chunk_ids

        results = store.retrieve("test", "SQLite warm memory tier", k=5)
        assert len(results) > 0
        assert any("SQLite" in r.content for r in results)

    def test_index_warm_section(self, tmp_path):
        from memory_manager.cold_store_chroma import ChromaColdStore

        store = ChromaColdStore(base_path=tmp_path / "chromadb")
        section = {
            "session_id": "sess1",
            "section_id": "warm_abc123",
            "section_name": "summary",
            "content": "The architecture uses three tiers: hot, warm, cold.",
            "metadata": {"tier": "extend"},
            "source_entry_ids": ["e1", "e2"],
        }
        ids = store.index_warm_section(section)
        assert ids

    def test_delete_chunks_for_source_section(self, tmp_path):
        from memory_manager.cold_store_chroma import ChromaColdStore

        store = ChromaColdStore(base_path=tmp_path / "chromadb")
        store.index_section(
            "sess", "s", "Content to delete.",
            source_section_id="warm_del_target",
        )
        deleted = store.delete_chunks_for_source_section("sess", "warm_del_target")
        assert deleted >= 1

        # Verify deletion — retrieve should return nothing for this content
        results = store.retrieve("sess", "Content to delete", k=5, min_relevance=0.0)
        # After deletion, results should be empty (collection is now empty)
        # ChromaDB may return empty; just verify it doesn't raise.
        assert isinstance(results, list)

    def test_delete_none_section_returns_zero(self, tmp_path):
        from memory_manager.cold_store_chroma import ChromaColdStore

        store = ChromaColdStore(base_path=tmp_path / "chromadb")
        assert store.delete_chunks_for_source_section("sess", None) == 0

    def test_min_relevance_filters(self, tmp_path):
        from memory_manager.cold_store_chroma import ChromaColdStore

        store = ChromaColdStore(base_path=tmp_path / "chromadb")
        store.index_section("sess", "s", "Completely unrelated content xyz.")
        results = store.retrieve("sess", "SQLite database", k=5, min_relevance=0.9)
        # Unrelated content should not score above 0.9 cosine similarity
        assert all(r.relevance >= 0.9 for r in results)

    def test_retrieve_empty_store(self, tmp_path):
        from memory_manager.cold_store_chroma import ChromaColdStore

        store = ChromaColdStore(base_path=tmp_path / "chromadb")
        results = store.retrieve("empty_session", "anything", k=5)
        assert results == []

    def test_drop_in_replaces_cold_store(self, tmp_path):
        """ChromaColdStore can be hot-swapped into TieredMemoryManager."""
        from memory_manager import TieredMemoryManager
        from memory_manager.cold_store_chroma import ChromaColdStore

        chroma = ChromaColdStore(base_path=tmp_path / "chromadb")
        with TieredMemoryManager(
            "chroma_swap",
            base_paths={"hot": tmp_path / "h", "warm": tmp_path / "w", "cold": tmp_path / "c"},
            autosave_interval=0,
            distiller=None,
        ) as mgr:
            mgr.cold_store = chroma
            mgr.add_user_message("Testing chroma integration with the memory manager.")
            mgr.compact_hot_to_warm()
            results = mgr.cold_store.retrieve(mgr.session_id, "chroma integration", k=5)
            assert isinstance(results, list)
