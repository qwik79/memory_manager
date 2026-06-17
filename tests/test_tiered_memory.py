from memory_manager import OllamaIntegration, TieredMemoryManager


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
    manager.add_user_message("Context assembly should include recent entries.")

    context = manager.build_context_for_llm("context assembly")
    assert "Current user instructions override" in context
    assert "Design context assembly" in context
    assert "recent entries" in context

    adapter = OllamaIntegration(manager)
    system_prompt, tools = adapter.get_context_for_llm_call({"content": "context assembly"})
    assert system_prompt
    assert tools[0]["function"]["name"] == "read_memory"
    usage = adapter.estimate_context_usage()
    assert usage["total_estimate"] > 0
