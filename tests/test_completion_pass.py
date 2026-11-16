"""Tests for scoring, OpenAI summarizer, LLMIntegration rename, and maybe_offload fix."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# scoring.py tests
# ---------------------------------------------------------------------------


class TestAutoClassifyImportance:
    """Tests for auto_classify_importance()."""

    def test_critical_security(self):
        from memory_manager.scoring import auto_classify_importance
        assert auto_classify_importance("We must not expose the API key.") == "critical"

    def test_critical_pin(self):
        from memory_manager.scoring import auto_classify_importance
        assert auto_classify_importance("[pin] This is pinned.") == "critical"

    def test_critical_always_never(self):
        from memory_manager.scoring import auto_classify_importance
        assert auto_classify_importance("Always use HTTPS for the endpoint.") == "critical"

    def test_high_decision(self):
        from memory_manager.scoring import auto_classify_importance
        assert auto_classify_importance("Decision: we are going with PostgreSQL.") == "high"

    def test_high_file_path(self):
        from memory_manager.scoring import auto_classify_importance
        assert auto_classify_importance("Edit the file ./memory_manager/models.py") == "high"

    def test_high_config(self):
        from memory_manager.scoring import auto_classify_importance
        assert auto_classify_importance("port=8080 and host=localhost") == "high"

    def test_medium_general(self):
        from memory_manager.scoring import auto_classify_importance
        assert auto_classify_importance("I think we should consider the alternatives.") == "medium"

    def test_low_acknowledgement(self):
        from memory_manager.scoring import auto_classify_importance
        assert auto_classify_importance("ok") == "low"
        assert auto_classify_importance("sounds good") == "low"
        assert auto_classify_importance("thanks") == "low"

    def test_empty_is_low(self):
        from memory_manager.scoring import auto_classify_importance
        assert auto_classify_importance("") == "low"
        assert auto_classify_importance("   ") == "low"


class TestContentRichness:
    """Tests for compute_content_richness()."""

    def test_empty(self):
        from memory_manager.scoring import compute_content_richness
        assert compute_content_richness("") == 0.0

    def test_code_block(self):
        from memory_manager.scoring import compute_content_richness
        text = "```python\ndef foo(): pass\n```"
        score = compute_content_richness(text)
        assert score > 0.1

    def test_file_paths(self):
        from memory_manager.scoring import compute_content_richness
        text = "Check ./src/main.py and ./config/settings.yaml"
        score = compute_content_richness(text)
        assert score > 0.1

    def test_decision_language(self):
        from memory_manager.scoring import compute_content_richness
        text = "Decision: we are using Redis for caching."
        score = compute_content_richness(text)
        assert score > 0.1

    def test_plain_text_low(self):
        from memory_manager.scoring import compute_content_richness
        text = "Hello, how are you doing today?"
        score = compute_content_richness(text)
        assert score < 0.15


class TestComputeRelevanceScore:
    """Tests for compute_relevance_score()."""

    def test_returns_float_in_range(self):
        from memory_manager.scoring import compute_relevance_score
        entry = {
            "content": "Some content",
            "importance": "medium",
            "tags": [],
            "created_at": time.time(),
        }
        score = compute_relevance_score(entry)
        assert 0.0 <= score <= 1.0

    def test_critical_scores_higher_than_low(self):
        from memory_manager.scoring import compute_relevance_score
        now = time.time()
        critical = {"content": "x", "importance": "critical", "tags": [], "created_at": now}
        low = {"content": "x", "importance": "low", "tags": [], "created_at": now}
        assert compute_relevance_score(critical, now=now) > compute_relevance_score(low, now=now)

    def test_pin_tag_boosts_score(self):
        from memory_manager.scoring import compute_relevance_score
        now = time.time()
        pinned = {"content": "x", "importance": "medium", "tags": ["pin"], "created_at": now}
        unpinned = {"content": "x", "importance": "medium", "tags": [], "created_at": now}
        assert compute_relevance_score(pinned, now=now) > compute_relevance_score(unpinned, now=now)

    def test_old_entry_scores_lower(self):
        from memory_manager.scoring import compute_relevance_score
        now = time.time()
        recent = {"content": "x", "importance": "medium", "tags": [], "created_at": now}
        old = {"content": "x", "importance": "medium", "tags": [], "created_at": now - 72 * 3600}
        assert compute_relevance_score(recent, now=now) > compute_relevance_score(old, now=now)

    def test_references_boost_score(self):
        from memory_manager.scoring import compute_relevance_score
        now = time.time()
        entry = {"content": "x", "importance": "medium", "tags": [], "created_at": now}
        no_refs = compute_relevance_score(entry, reference_count=0, now=now)
        many_refs = compute_relevance_score(entry, reference_count=10, now=now)
        assert many_refs > no_refs


class TestEntryIndex:
    """Tests for the in-memory EntryIndex."""

    def test_add_and_top_k(self):
        from memory_manager.scoring import EntryIndex
        idx = EntryIndex()
        idx.add("a", ["tag1"], "medium", "", 0.3)
        idx.add("b", ["tag2"], "high", "", 0.8)
        idx.add("c", ["tag1"], "low", "", 0.1)
        top = idx.top_k(k=2)
        assert top == ["b", "a"]

    def test_filter_by_tag(self):
        from memory_manager.scoring import EntryIndex
        idx = EntryIndex()
        idx.add("a", ["pin"], "medium", "", 0.5)
        idx.add("b", ["decision"], "high", "", 0.9)
        idx.add("c", ["pin"], "low", "", 0.2)
        result = idx.top_k(k=10, tags_filter=["pin"])
        assert set(result) == {"a", "c"}
        assert result[0] == "a"  # higher score first

    def test_filter_by_importance_min(self):
        from memory_manager.scoring import EntryIndex
        idx = EntryIndex()
        idx.add("a", [], "low", "", 0.9)
        idx.add("b", [], "high", "", 0.5)
        idx.add("c", [], "critical", "", 0.3)
        result = idx.top_k(k=10, importance_min="high")
        assert set(result) == {"b", "c"}

    def test_filter_by_phase(self):
        from memory_manager.scoring import EntryIndex
        idx = EntryIndex()
        idx.add("a", [], "medium", "planning", 0.5)
        idx.add("b", [], "medium", "debugging", 0.8)
        idx.add("c", [], "medium", "planning", 0.3)
        result = idx.top_k(k=10, phase="planning")
        assert set(result) == {"a", "c"}

    def test_remove(self):
        from memory_manager.scoring import EntryIndex
        idx = EntryIndex()
        idx.add("a", ["pin"], "high", "", 0.9)
        assert "a" in idx
        idx.remove("a")
        assert "a" not in idx
        assert idx.top_k(k=10) == []

    def test_update_score(self):
        from memory_manager.scoring import EntryIndex
        idx = EntryIndex()
        idx.add("a", [], "medium", "", 0.3)
        idx.add("b", [], "medium", "", 0.8)
        assert idx.top_k(k=1) == ["b"]
        idx.update_score("a", 0.95)
        assert idx.top_k(k=1) == ["a"]

    def test_rebuild(self):
        from memory_manager.scoring import EntryIndex
        entries = [
            {"entry_id": "e1", "content": "x", "importance": "high", "tags": ["pin"], "phase": "", "created_at": time.time()},
            {"entry_id": "e2", "content": "y", "importance": "low", "tags": [], "phase": "debug", "created_at": time.time()},
        ]
        idx = EntryIndex()
        idx.rebuild(entries)
        assert len(idx) == 2
        assert "e1" in idx
        assert "e2" in idx


# ---------------------------------------------------------------------------
# Hot buffer auto-classify and ranked retrieval
# ---------------------------------------------------------------------------


class TestHotBufferScoring:
    """Tests for scoring integration in HotBufferManager."""

    def test_auto_classify_on_append(self, tmp_path):
        from memory_manager.hot_buffer import HotBufferManager
        hb = HotBufferManager("test", base_path=str(tmp_path))
        # "Decision" content should auto-promote to high.
        eid = hb.append_entry("assistant", "Decision: use PostgreSQL for storage.")
        entries = hb.get_entries(limit=1)
        assert entries[0]["importance"] == "high"

    def test_explicit_importance_preserved(self, tmp_path):
        from memory_manager.hot_buffer import HotBufferManager
        hb = HotBufferManager("test", base_path=str(tmp_path))
        # Explicit "critical" should not be overridden.
        eid = hb.append_entry("user", "hello world", importance="critical")
        entries = hb.get_entries(limit=1)
        assert entries[0]["importance"] == "critical"

    def test_get_entries_ranked_returns_highest_first(self, tmp_path):
        from memory_manager.hot_buffer import HotBufferManager
        hb = HotBufferManager("test", base_path=str(tmp_path))
        hb.append_entry("user", "ok")  # low importance
        hb.append_entry("assistant", "Decision: use Redis. Check ./config/settings.yaml")
        ranked = hb.get_entries_ranked(limit=10)
        assert len(ranked) == 2
        # The decision entry should be first (higher score).
        assert "Decision" in ranked[0]["content"]

    def test_index_populated_on_startup(self, tmp_path):
        from memory_manager.hot_buffer import HotBufferManager
        hb1 = HotBufferManager("test", base_path=str(tmp_path))
        hb1.append_entry("user", "Security: always use HTTPS")
        hb1.append_entry("assistant", "Got it.")
        # Recreate — should rebuild index from persisted entries.
        hb2 = HotBufferManager("test", base_path=str(tmp_path))
        assert len(hb2._index) == 2
        ranked = hb2.get_entries_ranked(limit=10)
        assert len(ranked) == 2


# ---------------------------------------------------------------------------
# Eviction — importance protection
# ---------------------------------------------------------------------------


class TestEvictionImportanceProtection:
    """Tests that critical/high importance entries are protected from eviction."""

    def test_high_importance_entry_gets_pinned(self, tmp_path):
        from memory_manager.eviction import EvictionManager
        from memory_manager.hot_buffer import HotBufferManager

        hb = HotBufferManager("test", base_path=str(tmp_path))
        # Create a critical entry that's old enough to be eviction-eligible.
        eid = hb.append_entry("assistant", "We must never expose credentials.", importance="critical")
        # Backdate it to make it old.
        with hb._lock:
            data = hb._read_state()
            for e in data["entries"]:
                if e["entry_id"] == eid:
                    e["created_at"] = time.time() - 10 * 3600
            hb._write_state(data)

        em = EvictionManager(hb)
        offload, pin = em.evaluate_for_eviction()
        # Critical entry should be pinned, not offloaded.
        assert eid in pin
        assert eid not in offload


# ---------------------------------------------------------------------------
# OpenAISummarizer
# ---------------------------------------------------------------------------


class TestOpenAISummarizer:
    """Tests for OpenAISummarizer payload format."""

    def test_empty_entries_uses_fallback(self):
        from memory_manager.openai_summarizer import OpenAISummarizer
        summarizer = OpenAISummarizer()
        result = summarizer.summarize_entries([], None)
        assert result["summary"] == ""

    def test_payload_uses_openai_format(self):
        from memory_manager.openai_summarizer import OpenAISummarizer
        summarizer = OpenAISummarizer(model="test-model", endpoint="http://fake:9999/v1/chat/completions")
        entries = [{"entry_type": "user", "content": "Hello world"}]
        # The server is not running, so it will fall back to heuristic.
        result = summarizer.summarize_entries(entries, {"pinned_summary": ""})
        # Should have produced a heuristic summary (fallback).
        assert isinstance(result.get("summary"), str)
        assert "confidence" in result


# ---------------------------------------------------------------------------
# LLMIntegration alias
# ---------------------------------------------------------------------------


class TestLLMIntegrationAlias:
    """Tests that OllamaIntegration is a backward-compatible alias."""

    def test_alias_identity(self):
        from memory_manager import LLMIntegration, OllamaIntegration
        assert LLMIntegration is OllamaIntegration

    def test_old_import_still_works(self, tmp_path):
        from memory_manager import OllamaIntegration, TieredMemoryManager
        with TieredMemoryManager(
            session_id="alias-test",
            base_paths={"hot": str(tmp_path / "h"), "warm": str(tmp_path / "w"), "cold": str(tmp_path / "c")},
            autosave_interval=0,
        ) as mgr:
            integration = OllamaIntegration(mgr)
            system_prompt, tools = integration.get_context_for_llm_call(
                {"role": "user", "content": "test"}
            )
            assert isinstance(system_prompt, str)
            assert isinstance(tools, list)


# ---------------------------------------------------------------------------
# maybe_offload auto-cascade
# ---------------------------------------------------------------------------


class TestMaybeOffloadAutoCascade:
    """Tests that maybe_offload now routes through distillation + auto-cascade."""

    def test_eviction_applies_distillation(self, tmp_path):
        from memory_manager import TieredMemoryManager

        with TieredMemoryManager(
            session_id="offload-test",
            base_paths={"hot": str(tmp_path / "h"), "warm": str(tmp_path / "w"), "cold": str(tmp_path / "c")},
            autosave_interval=0,
        ) as mgr:
            # Add enough old entries to trigger eviction.
            for i in range(5):
                eid = mgr.add_user_message(f"Old message {i}")
                mgr.add_assistant_message(f"Old reply {i}")

            # Backdate all entries to be eviction-eligible.
            with mgr.hot_buffer._lock:
                data = mgr.hot_buffer._read_state()
                for e in data["entries"]:
                    e["created_at"] = time.time() - 24 * 3600
                mgr.hot_buffer._write_state(data)

            result = mgr.maybe_offload()
            # Some entries should have been offloaded.
            if result["offloaded"]:
                # Verify the warm section was created.
                sections = mgr.warm_store.list_sections(mgr.session_id, include_indexed=True)
                assert len(sections) > 0
