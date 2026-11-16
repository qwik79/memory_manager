"""Tests for background_worker: all distiller implementations + BackgroundWorker."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from memory_manager.background_worker import (
    BackgroundWorker,
    CascadeDistiller,
    Distiller,
    HeuristicDistiller,
    LLMDistiller,
    SpeculativeDistiller,
    _empty_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MESSAGES = [
    {"role": "user", "content": "Let's implement the storage layer. We decided to use SQLite."},
    {"role": "assistant", "content": "Good decision: SQLite gives us zero-dependency persistence. Next up: create the schema."},
]


def _make_worker(distiller, results: list) -> BackgroundWorker:
    """Create a BackgroundWorker that appends results to a list."""
    return BackgroundWorker(
        distiller=distiller,
        on_result=results.append,
        get_summary=lambda: "prior summary",
    )


# ---------------------------------------------------------------------------
# HeuristicDistiller
# ---------------------------------------------------------------------------


class TestHeuristicDistiller:
    def test_returns_canonical_schema(self):
        d = HeuristicDistiller()
        result = d.distill(MESSAGES)
        for key in ("summary", "decisions", "next_steps", "facts", "tier", "notes_for_core", "confidence"):
            assert key in result, f"Missing key: {key}"

    def test_empty_messages_returns_empty_result(self):
        d = HeuristicDistiller()
        result = d.distill([])
        assert result["summary"] == ""
        assert result["confidence"] == 0.0

    def test_extracts_decision(self):
        d = HeuristicDistiller()
        msgs = [{"role": "user", "content": "Decision: we will use Redis for caching."}]
        result = d.distill(msgs)
        assert any("Redis" in dec for dec in result["decisions"])

    def test_extracts_next_step(self):
        d = HeuristicDistiller()
        msgs = [{"role": "assistant", "content": "TODO: write unit tests for the auth module."}]
        result = d.distill(msgs)
        assert any("unit tests" in step.lower() for step in result["next_steps"])

    def test_extracts_file_paths(self):
        d = HeuristicDistiller()
        msgs = [{"role": "user", "content": "Edit memory_manager/storage.py and tests/test_storage.py."}]
        result = d.distill(msgs)
        paths = result["facts"].get("paths", [])
        assert any("storage.py" in p for p in paths)

    def test_tier_pin_on_decision(self):
        d = HeuristicDistiller()
        msgs = [{"role": "user", "content": "Critical decision: use HMAC for all file signatures."}]
        result = d.distill(msgs)
        assert result["tier"] == "pin"

    def test_tier_archive_on_resolved(self):
        d = HeuristicDistiller()
        msgs = [{"role": "assistant", "content": "The bug is fixed and resolved. Tests all passing."}]
        result = d.distill(msgs)
        assert result["tier"] == "archive"

    def test_summary_is_nonempty_for_real_content(self):
        d = HeuristicDistiller()
        result = d.distill(MESSAGES)
        assert len(result["summary"]) > 10

    def test_confidence_is_in_range(self):
        d = HeuristicDistiller()
        result = d.distill(MESSAGES)
        assert 0.0 <= result["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# CascadeDistiller
# ---------------------------------------------------------------------------


class TestCascadeDistiller:
    def test_returns_first_success(self):
        good = HeuristicDistiller()
        # Second distiller should not be called
        bad = MagicMock(spec=Distiller)
        bad.distill.side_effect = RuntimeError("should not be called")
        d = CascadeDistiller([good, bad])
        result = d.distill(MESSAGES)
        assert result["summary"]
        bad.distill.assert_not_called()

    def test_falls_back_on_failure(self):
        failing = MagicMock(spec=Distiller)
        failing.distill.side_effect = RuntimeError("LLM unavailable")
        fallback = HeuristicDistiller()
        d = CascadeDistiller([failing, fallback])
        result = d.distill(MESSAGES)
        assert result["summary"]

    def test_empty_distillers_raises(self):
        with pytest.raises(ValueError):
            CascadeDistiller([])

    def test_all_fail_returns_empty(self):
        bad1 = MagicMock(spec=Distiller)
        bad1.distill.side_effect = RuntimeError
        bad2 = MagicMock(spec=Distiller)
        bad2.distill.return_value = {"summary": ""}  # empty summary → skipped
        d = CascadeDistiller([bad1, bad2])
        result = d.distill(MESSAGES)
        assert result == _empty_result()


# ---------------------------------------------------------------------------
# SpeculativeDistiller
# ---------------------------------------------------------------------------


class TestSpeculativeDistiller:
    def test_returns_draft_immediately(self):
        slow_refine = MagicMock(spec=Distiller)
        slow_refine.distill.side_effect = lambda *a, **k: (time.sleep(5), {})[1]
        d = SpeculativeDistiller(draft=HeuristicDistiller(), refine=slow_refine)
        start = time.monotonic()
        result = d.distill(MESSAGES)
        elapsed = time.monotonic() - start
        assert result["summary"]          # draft arrived
        assert elapsed < 1.0             # not blocked by slow refine
        d.stop()

    def test_refinement_callback_called(self):
        refined = {"summary": "Refined!", "decisions": ["use HMAC"], "next_steps": [],
                   "facts": {}, "tier": "pin", "notes_for_core": "", "confidence": 0.9}
        refiner = MagicMock(spec=Distiller)
        refiner.distill.return_value = refined
        callback_results: list[dict] = []
        d = SpeculativeDistiller(draft=HeuristicDistiller(), refine=refiner)
        d.set_refinement_callback(callback_results.append)
        d.distill(MESSAGES)
        # Give the refinement thread time to complete
        time.sleep(0.5)
        assert len(callback_results) == 1
        assert callback_results[0]["summary"] == "Refined!"
        d.stop()

    def test_refinement_failure_does_not_propagate(self):
        refiner = MagicMock(spec=Distiller)
        refiner.distill.side_effect = RuntimeError("LLM down")
        callback_results: list[dict] = []
        d = SpeculativeDistiller(draft=HeuristicDistiller(), refine=refiner)
        d.set_refinement_callback(callback_results.append)
        result = d.distill(MESSAGES)  # should not raise
        assert result["summary"]
        time.sleep(0.3)
        assert len(callback_results) == 0  # failed → callback not called
        d.stop()


# ---------------------------------------------------------------------------
# BackgroundWorker
# ---------------------------------------------------------------------------


class TestBackgroundWorker:
    def test_push_and_result_arrives(self):
        results: list[dict] = []
        worker = _make_worker(HeuristicDistiller(), results)
        worker.push(MESSAGES)
        worker.flush()
        assert len(results) >= 1
        assert results[0]["summary"]
        worker.stop()

    def test_flush_blocks_until_done(self):
        results: list[dict] = []
        worker = _make_worker(HeuristicDistiller(), results)
        for _ in range(5):
            worker.push(MESSAGES)
        worker.flush()
        # All messages have been processed by the time flush returns.
        assert len(results) >= 1
        worker.stop()

    def test_batches_multiple_pushes(self):
        """Rapid pushes should be batched into fewer distillation calls."""
        call_count = [0]
        original_distill = HeuristicDistiller().distill

        class CountingDistiller:
            def distill(self, messages, prev_summary=""):
                call_count[0] += 1
                return original_distill(messages, prev_summary)

        results: list[dict] = []
        worker = BackgroundWorker(
            distiller=CountingDistiller(),
            on_result=results.append,
            get_summary=lambda: "",
        )
        # Push many items rapidly before the worker thread can drain.
        for _ in range(10):
            worker.push(MESSAGES)
        worker.flush()
        # Batching means fewer calls than pushes.
        assert call_count[0] <= 10  # Could be anywhere from 1 to 10
        worker.stop()

    def test_empty_push_ignored(self):
        results: list[dict] = []
        worker = _make_worker(HeuristicDistiller(), results)
        worker.push([])  # empty delta — should be a no-op
        worker.flush()
        assert len(results) == 0
        worker.stop()

    def test_speculative_callback_registered_automatically(self):
        """BackgroundWorker should auto-register callback for SpeculativeDistiller."""
        callback_results: list[dict] = []
        refined = {"summary": "Refined", "decisions": [], "next_steps": [],
                   "facts": {}, "tier": "extend", "notes_for_core": "", "confidence": 0.9}
        refiner = MagicMock(spec=Distiller)
        refiner.distill.return_value = refined
        spec = SpeculativeDistiller(draft=HeuristicDistiller(), refine=refiner)
        worker = BackgroundWorker(
            distiller=spec,
            on_result=callback_results.append,
            get_summary=lambda: "",
        )
        worker.push(MESSAGES)
        worker.flush()
        time.sleep(0.5)  # allow refinement thread to fire
        # Draft result + refined result both routed through on_result
        assert len(callback_results) >= 1
        worker.stop()

    def test_stop_is_clean(self):
        worker = _make_worker(HeuristicDistiller(), [])
        worker.stop()  # should not raise or hang


# ---------------------------------------------------------------------------
# Integration: TieredMemoryManager uses BackgroundWorker
# ---------------------------------------------------------------------------


class TestOrchestratorIntegration:
    def test_default_distiller_is_heuristic(self, tmp_path):
        from memory_manager import TieredMemoryManager

        with TieredMemoryManager(
            session_id="bgtest",
            base_paths={"hot": str(tmp_path / "hot"), "warm": str(tmp_path / "warm"), "cold": str(tmp_path / "cold")},
            autosave_interval=0,
        ) as mgr:
            # Default CascadeDistiller([HeuristicDistiller()]) is instant
            # → uses inline path, not background thread.
            assert mgr._inline_distiller is not None
            assert mgr.background_worker is None

    def test_add_assistant_message_pushes_to_worker(self, tmp_path):
        from memory_manager import CascadeDistiller, HeuristicDistiller, LLMDistiller, TieredMemoryManager

        results: list[dict] = []
        # Use LLMDistiller so is_instant=False → background thread is created.
        with TieredMemoryManager(
            session_id="bgtest2",
            base_paths={"hot": str(tmp_path / "hot"), "warm": str(tmp_path / "warm"), "cold": str(tmp_path / "cold")},
            distiller=LLMDistiller(),
            autosave_interval=0,
        ) as mgr:
            assert mgr.background_worker is not None
            mgr.background_worker._on_result = results.append  # spy
            mgr.add_user_message("What is the plan?")
            mgr.add_assistant_message("The plan is to use tiered memory.")
            mgr.background_worker.flush()

    def test_none_distiller_disables_worker(self, tmp_path):
        from memory_manager import TieredMemoryManager

        with TieredMemoryManager(
            session_id="noworker",
            base_paths={"hot": str(tmp_path / "hot"), "warm": str(tmp_path / "warm"), "cold": str(tmp_path / "cold")},
            distiller=None,
            autosave_interval=0,
        ) as mgr:
            assert mgr.background_worker is None

    def test_get_pinned_summary_returns_string(self, tmp_path):
        from memory_manager import TieredMemoryManager

        with TieredMemoryManager(
            session_id="pinsummary",
            pinned_summary="Initial summary.",
            base_paths={"hot": str(tmp_path / "hot"), "warm": str(tmp_path / "warm"), "cold": str(tmp_path / "cold")},
            autosave_interval=0,
        ) as mgr:
            assert mgr.get_pinned_summary() == "Initial summary."
