from memory_manager import OllamaIntegration, TieredMemoryManager
from memory_manager.storage import atomic_json_write, read_json


def make_manager(tmp_path):
    return TieredMemoryManager(
        session_id="test_session",
        project_name="Test Project",
        pinned_summary="Build a memory system.",
        base_paths={
            "hot": tmp_path / "hot",
            "warm": tmp_path / "warm",
            "cold": tmp_path / "cold",
        },
    )


def test_hot_to_warm_to_cold_flow(tmp_path):
    manager = make_manager(tmp_path)
    first = manager.add_user_message("We need hot, warm, and cold memory tiers.")
    manager.add_assistant_message(
        "Decision: warm storage is canonical and cold storage is rebuildable.",
        tags=["decision", "high-priority"],
    )

    assert first.startswith("entry_")
    assert len(manager.hot_buffer.get_entries()) == 2

    section_id = manager.compact_hot_to_warm()
    assert section_id and section_id.startswith("warm_")
    assert manager.hot_buffer.get_entries() == []

    chunks = manager.index_warm_to_cold()
    assert chunks

    results = manager.refresh_context("canonical rebuildable storage")
    assert results
    assert any("canonical" in result.content for result in results)


def test_context_and_ollama_adapter(tmp_path):
    manager = make_manager(tmp_path)
    manager.session.current_task = "Design context assembly"
    manager.add_decision("Use pinned context with explicit safety boundaries.", "architecture")
    entry_id = manager.add_user_message("Context assembly should include recent entries.")

    context = manager.build_context_for_llm("context assembly")
    assert "Current user instructions override" in context
    assert "Design context assembly" in context
    assert "recent entries" in context
    assert "tier=hot" in context
    assert entry_id in context

    adapter = OllamaIntegration(manager)
    system_prompt, tools = adapter.get_context_for_llm_call({"content": "context assembly"})
    assert system_prompt
    assert tools[0]["function"]["name"] == "read_memory"
    usage = adapter.estimate_context_usage()
    assert usage["total_estimate"] > 0


def test_atomic_json_write_uses_unique_temp_files(tmp_path):
    path = tmp_path / "state.json"

    for index in range(5):
        atomic_json_write(path, {"index": index})

    assert read_json(path) == {"index": 4}
    assert not (tmp_path / "state.json.tmp").exists()
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_reindexing_warm_section_replaces_existing_cold_chunks(tmp_path):
    manager = make_manager(tmp_path)
    manager.add_user_message("alpha beta gamma " * 600)

    section_id = manager.compact_hot_to_warm()
    assert section_id
    first_chunks = manager.index_warm_to_cold()
    second_chunks = manager.index_warm_to_cold(include_indexed=True)

    assert first_chunks
    assert second_chunks
    results = manager.cold_store.retrieve(manager.session_id, "alpha beta gamma", k=20)
    matching_results = [
        result for result in results if result.metadata.get("source_section_id") == section_id
    ]
    assert len(matching_results) == len(second_chunks)
    assert not set(first_chunks) & set(second_chunks)


def test_session_id_is_normalized_across_storage_layers(tmp_path):
    manager = TieredMemoryManager(
        session_id="Project A / demo!",
        base_paths={
            "hot": tmp_path / "hot",
            "warm": tmp_path / "warm",
            "cold": tmp_path / "cold",
        },
    )

    assert manager.session_id == "ProjectAdemo"
    assert manager.hot_buffer.file_path.name == "ProjectAdemo_hot.json"
    manager.add_user_message("normalization links warm and cold storage")
    section_id = manager.compact_hot_to_warm()
    assert section_id
    section = manager.warm_store.get_section(manager.session_id, section_id)
    assert section["session_id"] == "ProjectAdemo"

    manager.index_warm_to_cold()
    result = manager.cold_store.retrieve(manager.session_id, "normalization", k=1)[0]
    assert result.source.startswith("cold://ProjectAdemo/")


def test_retrieved_context_includes_memory_provenance(tmp_path):
    manager = make_manager(tmp_path)
    manager.add_user_message("provenance should appear for retrieved canonical context")
    section_id = manager.compact_hot_to_warm()
    assert section_id
    manager.index_warm_to_cold()

    context = manager.build_context_for_llm("provenance canonical context")

    assert "source=warm://" in context or "source=cold://" in context
    assert "relevance=" in context
    assert section_id in context or "section=session_summary" in context
