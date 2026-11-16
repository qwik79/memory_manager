import tempfile

from memory_manager import TieredMemoryManager


def test_default_mode_is_ephemeral_and_touches_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    manager = TieredMemoryManager("demo")
    assert manager.is_ephemeral
    manager.add_user_message("ephemeral memory should never reach the disk")
    manager.add_assistant_message("understood, it stays in RAM")
    assert manager.compact_hot_to_warm()  # auto-cascades into cold
    assert manager.cold_store.list_chunks(manager.session_id)
    assert manager.refresh_context("ephemeral disk")
    manager.export_session(tmp_path / "out" / "snap")  # the one deliberate write
    assert [p.name for p in tmp_path.rglob("*") if p.is_file()] == ["snap.tar.gz"]
    (tmp_path / "out" / "snap.tar.gz").unlink()
    (tmp_path / "out").rmdir()

    assert list(tmp_path.rglob("*")) == []


def test_closing_an_ephemeral_session_wipes_every_tier():
    with TieredMemoryManager.ephemeral("demo") as manager:
        manager.add_user_message("short lived")
        manager.compact_hot_to_warm()
    assert manager.closed
    assert manager.hot_buffer.get_entries(limit=None, include_compacted=True) == []
    assert manager.warm_store.list_sections(manager.session_id) == []
    assert manager.cold_store.list_chunks(manager.session_id) == []


def test_persistent_mode_survives_reopen(tmp_path):
    with TieredMemoryManager.persistent(tmp_path, session_id="keep") as manager:
        assert manager.is_persistent
        manager.add_user_message("durable fact about the warm store")
        manager.compact_hot_to_warm()

    reopened = TieredMemoryManager.persistent(tmp_path, session_id="keep")
    assert reopened.cold_store.retrieve("keep", "durable warm store")
    assert (tmp_path / "hot" / "keep_hot.json").exists()


def test_promote_moves_an_ephemeral_session_to_disk(tmp_path):
    manager = TieredMemoryManager.ephemeral("later")
    manager.add_user_message("promote me to persistent storage")
    manager.compact_hot_to_warm()
    manager.add_user_message("still hot when promoted")

    manager.promote(tmp_path)
    assert manager.is_persistent
    manager.close()  # persistent: close keeps the data

    reopened = TieredMemoryManager.persistent(tmp_path, session_id="later")
    assert [e["content"] for e in reopened.hot_buffer.get_entries()] == ["still hot when promoted"]
    assert reopened.warm_store.list_sections("later")
    assert reopened.cold_store.retrieve("later", "promote persistent")


def test_cold_index_rebuilds_from_warm_at_full_fidelity():
    manager = TieredMemoryManager.ephemeral("rebuild")
    lines = [f"line {i} mentions token{i}" for i in range(60)]
    manager.add_user_message("\n".join(lines))
    section_id = manager.offload_entries([e["entry_id"] for e in manager.hot_buffer.get_entries()])
    section = manager.warm_store.get_section("rebuild", section_id)
    assert "...[middle compacted]..." in section["content"]  # the summary is lossy...

    manager.cold_store.clear("rebuild")
    manager.index_warm_to_cold(include_indexed=True)
    hits = manager.cold_store.retrieve("rebuild", "token30")
    assert hits and "token30" in hits[0].content  # ...but the rebuilt index is not


def test_export_import_round_trips_an_ephemeral_session(tmp_path):
    source = TieredMemoryManager.ephemeral("trip", project_name="Round trip")
    source.add_user_message("the archive should carry this fact about zeppelins")
    source.compact_hot_to_warm()
    source.add_user_message("and this hot entry")
    archive = source.export_session(tmp_path / "trip")

    target = TieredMemoryManager.ephemeral("trip")
    target.import_session(archive)
    assert target.session.project_name == "Round trip"
    assert [e["content"] for e in target.hot_buffer.get_entries()] == ["and this hot entry"]
    assert target.cold_store.retrieve("trip", "zeppelins")


def test_secure_flag_is_ignored_when_ephemeral():
    manager = TieredMemoryManager.ephemeral("sec", secure=True)
    assert manager.security is None
    manager.add_user_message("nothing to sign in RAM")
    assert manager.hot_buffer.get_entries()


def test_persistent_session_resumes_state(tmp_path):
    with TieredMemoryManager.persistent(tmp_path, session_id="resume", project_name="Resumable") as manager:
        manager.add_decision("keep warm canonical", "architecture")
    reopened = TieredMemoryManager.persistent(tmp_path, session_id="resume")
    assert reopened.session.project_name == "Resumable"
    assert "keep warm canonical" in reopened.active_context.get_pinned_context()
    reopened.close()


def test_import_session_refreshes_pinned_context(tmp_path):
    source = TieredMemoryManager.ephemeral("imp", project_name="From archive")
    archive = source.export_session(tmp_path / "imp")
    target = TieredMemoryManager.ephemeral("imp", project_name="Before import")
    target.import_session(archive)
    assert "From archive" in target.active_context.get_pinned_context()


def test_wipe_then_close_leaves_nothing_on_disk(tmp_path):
    manager = TieredMemoryManager.persistent(tmp_path, session_id="gone")
    manager.add_user_message("delete me")
    manager.compact_hot_to_warm()
    manager.wipe()
    manager.close()
    leftovers = [p for p in tmp_path.rglob("*.json") if "gone" in str(p) and "_hot" not in p.name]
    assert leftovers == []
    hot = TieredMemoryManager.persistent(tmp_path, session_id="gone")
    assert hot.hot_buffer.get_entries(limit=None, include_compacted=True) == []
    hot.close()
