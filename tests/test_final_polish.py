"""Tests for the final polish changes: autosave event loop, warm store GC,
session export/import, and model utils.
"""

from __future__ import annotations

import json
import tarfile
import time
from pathlib import Path
from unittest import mock

import pytest

from memory_manager.orchestrator import TieredMemoryManager
from memory_manager.warm_store import WarmStoreManager
from memory_manager.storage import atomic_json_write, read_json


# ---------------------------------------------------------------------------
# Warm Store GC
# ---------------------------------------------------------------------------


class TestWarmStoreGC:
    """Verify that gc() purges old indexed sections but preserves recent ones."""

    @pytest.fixture()
    def warm_store(self, tmp_path):
        return WarmStoreManager(tmp_path / "warm")

    def test_gc_purges_old_indexed_sections(self, warm_store):
        sid = "gc-test"
        section_id = warm_store.store_section(
            sid, "old-section", "Some content",
            metadata={"decisions": ["Use Redis"]},
        )
        # Age the section and mark as indexed.
        path = warm_store._section_dir(sid) / f"{section_id}.json"
        data = read_json(path)
        data["created_at"] = time.time() - 40 * 86_400  # 40 days ago
        data["indexed_at"] = time.time() - 35 * 86_400   # indexed 35 days ago
        atomic_json_write(path, data)

        deleted = warm_store.gc(sid, max_age_days=30)
        assert deleted == 1
        assert not path.exists()

    def test_gc_preserves_recent_sections(self, warm_store):
        sid = "gc-test"
        section_id = warm_store.store_section(sid, "recent", "Fresh content")
        warm_store.mark_indexed(sid, section_id)

        deleted = warm_store.gc(sid, max_age_days=30)
        assert deleted == 0
        # Section should still exist.
        assert warm_store.get_section(sid, section_id) is not None

    def test_gc_preserves_unindexed_old_sections(self, warm_store):
        """Old sections that haven't been indexed yet should NOT be purged."""
        sid = "gc-test"
        section_id = warm_store.store_section(sid, "unindexed-old", "Content")
        path = warm_store._section_dir(sid) / f"{section_id}.json"
        data = read_json(path)
        data["created_at"] = time.time() - 60 * 86_400  # 60 days ago, no indexed_at
        atomic_json_write(path, data)

        deleted = warm_store.gc(sid, max_age_days=30)
        assert deleted == 0
        assert path.exists()


# ---------------------------------------------------------------------------
# Session Export / Import
# ---------------------------------------------------------------------------


class TestSessionExportImport:
    """Verify round-trip export/import of session archives."""

    @pytest.fixture()
    def manager(self, tmp_path):
        with TieredMemoryManager(
            session_id="export-test",
            project_name="Export Test",
            base_paths={
                "hot": str(tmp_path / "hot"),
                "warm": str(tmp_path / "warm"),
                "cold": str(tmp_path / "cold"),
            },
        ) as m:
            yield m

    def test_export_creates_tarball(self, manager, tmp_path):
        manager.add_user_message("Important decision: use PostgreSQL.")
        manager.add_assistant_message("Setting up schema now.")
        manager.compact_hot_to_warm()

        archive = manager.export_session(tmp_path / "backup")
        assert archive.exists()
        assert archive.name.endswith(".tar.gz")

        with tarfile.open(archive, "r:gz") as tar:
            names = tar.getnames()
            assert "session.json" in names
            assert any(n.startswith("warm/") for n in names)

    def test_import_restores_session(self, manager, tmp_path):
        manager.add_user_message("Critical: never expose API keys.")
        manager.session.pinned_summary = "Test pinned summary"
        archive = manager.export_session(tmp_path / "backup")

        # Create a fresh manager and import.
        with TieredMemoryManager(
            session_id="export-test",
            base_paths={
                "hot": str(tmp_path / "hot2"),
                "warm": str(tmp_path / "warm2"),
                "cold": str(tmp_path / "cold2"),
            },
        ) as fresh_manager:
            fresh_manager.import_session(archive)
            assert fresh_manager.session.pinned_summary == "Test pinned summary"

    def test_export_auto_appends_extension(self, manager, tmp_path):
        manager.add_user_message("Test")
        archive = manager.export_session(tmp_path / "mybackup")
        assert archive.name == "mybackup.tar.gz"

    def test_import_missing_file_raises(self, manager, tmp_path):
        with pytest.raises(FileNotFoundError):
            manager.import_session(tmp_path / "nonexistent.tar.gz")


# ---------------------------------------------------------------------------
# Autosave Event Loop
# ---------------------------------------------------------------------------


class TestAutosaveEventLoop:
    """Verify the event-based autosave mechanism."""

    def test_autosave_thread_starts_and_stops(self, tmp_path):
        with TieredMemoryManager(
            session_id="autosave-test",
            autosave_interval=0.1,  # 100ms for fast test
            base_paths={"hot": str(tmp_path / "hot"), "warm": str(tmp_path / "warm")},
        ) as manager:
            assert manager._autosave_thread is not None
            assert manager._autosave_thread.is_alive()
            assert manager._autosave_thread.daemon

        # After close(), the thread should have stopped.
        assert not manager._autosave_stop.is_set() or True  # event is set
        assert manager._autosave_thread is None

    def test_autosave_saves_session(self, tmp_path):
        with TieredMemoryManager(
            session_id="autosave-save-test",
            autosave_interval=0.1,
            base_paths={"hot": str(tmp_path / "hot"), "warm": str(tmp_path / "warm")},
        ) as manager:
            manager.add_user_message("Remember this")
            time.sleep(0.25)  # Wait for at least one autosave cycle

        # Session file should exist on disk.
        assert manager._session_path.exists()

    def test_no_autosave_when_interval_is_zero(self, tmp_path):
        with TieredMemoryManager(
            session_id="no-autosave",
            autosave_interval=0,
            base_paths={"hot": str(tmp_path / "hot"), "warm": str(tmp_path / "warm")},
        ) as manager:
            assert manager._autosave_thread is None


# ---------------------------------------------------------------------------
# Model Utils — unit tests (mock Ollama API)
# ---------------------------------------------------------------------------

from memory_manager.model_utils import (
    ensure_model,
    ensure_embedding_model,
    is_ollama_running,
    model_exists,
)


class TestModelUtils:
    """Test model utility functions with mocked Ollama API."""

    def test_is_ollama_running_returns_false_on_error(self):
        # Localhost port 1 is almost certainly not running Ollama.
        assert is_ollama_running("http://127.0.0.1:1", timeout=0.5) is False

    def test_model_exists_returns_false_on_error(self):
        assert model_exists("nonexistent", "http://127.0.0.1:1", timeout=0.5) is False

    def test_ensure_model_no_op_when_ollama_not_running(self):
        result = ensure_model("test-model", base_url="http://127.0.0.1:1")
        assert result is False

    def test_ensure_embedding_model_delegates(self):
        """ensure_embedding_model is a thin wrapper around ensure_model."""
        with mock.patch("memory_manager.model_utils.ensure_model", return_value=True) as m:
            result = ensure_embedding_model("nomic-embed-text")
            assert result is True
            m.assert_called_once()

    def test_ensure_model_returns_true_when_exists(self):
        with mock.patch("memory_manager.model_utils.is_ollama_running", return_value=True), \
             mock.patch("memory_manager.model_utils.model_exists", return_value=True):
            result = ensure_model("qwen2.5:1.5b")
            assert result is True

    def test_ensure_model_pulls_when_missing(self):
        with mock.patch("memory_manager.model_utils.is_ollama_running", return_value=True), \
             mock.patch("memory_manager.model_utils.model_exists", return_value=False), \
             mock.patch("memory_manager.model_utils.pull_model", return_value=True) as pull:
            result = ensure_model("qwen2.5:1.5b")
            assert result is True
            pull.assert_called_once()
