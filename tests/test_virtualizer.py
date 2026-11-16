import pytest

from memory_manager import ContextVirtualizer, TieredMemoryManager, VirtualizerConfig
from memory_manager.virtualizer import SUPERSEDED_TAG, message_tokens

TOPICS = [
    "the hot buffer lives on tmpfs under /dev/shm for fast writes",
    "chromadb will replace the lexical cold index later in the roadmap",
    "the memory unit runs qwen2.5:1.5b on cpu with num_gpu set to zero",
    "the offload trigger combines token fill, semantic drift and structural seams",
    "nomic-embed-text embeddings are served by a fastapi service on port 8001",
    "the background worker distills message deltas between exchanges",
    "warm storage is canonical and cold storage is rebuildable",
    "the proxy forwards requests to an openai-compatible endpoint",
]

SMALL = dict(max_context_tokens=900, reserve_tokens=150, history_budget_tokens=120, retrieval_budget_tokens=150)


def make(config=None, manager=None):
    manager = manager or TieredMemoryManager.ephemeral("v")
    return manager, ContextVirtualizer(manager, VirtualizerConfig(**(config or SMALL)))


def converse(virtualizer, history, topics=TOPICS, observe=True):
    outputs = []
    for topic in topics:
        history.append({"role": "user", "content": f"Let's talk about this: {topic}. Explain the tradeoffs please."})
        outputs.append(virtualizer.rewrite(history))
        reply = f"Regarding {topic}: there are considerations around latency, durability and complexity."
        if observe:
            virtualizer.observe(reply)
        history.append({"role": "assistant", "content": reply})
    return outputs


def mirror_entries(manager):
    return [e for e in manager.hot_buffer.get_entries(limit=None, include_compacted=True) if SUPERSEDED_TAG not in e["tags"]]


def test_rewrite_keeps_prompt_within_budget_and_preserves_host_system():
    manager, v = make()
    history = [{"role": "system", "content": "You are a terse assistant."}]
    for out in converse(v, history):
        assert out[0]["role"] == "system"
        assert out[0]["content"].startswith("You are a terse assistant.")
        assert sum(message_tokens(m) for m in out) <= 900 - 150
    assert v.last_stats["offloaded_messages_total"] > 0
    assert history[0]["content"] == "You are a terse assistant."  # input untouched


def test_offloaded_turns_are_retrieved_into_the_latest_user_message():
    _, v = make()
    history = []
    converse(v, history)
    history.append({"role": "user", "content": "Which model does the memory unit use, and on what device?"})
    out = v.rewrite(history)
    last = out[-1]
    assert last["role"] == "user"
    assert last["content"].startswith("<retrieved_memory>")
    assert "qwen2.5:1.5b" in last["content"]
    assert last["content"].endswith("Which model does the memory unit use, and on what device?")
    assert all("<retrieved_memory>" not in m["content"] for m in out[:-1])


def test_resending_history_never_duplicates_memory():
    manager, v = make(dict(max_context_tokens=100_000))
    history = []
    converse(v, history, observe=True)
    count = len(mirror_entries(manager))
    v.rewrite(history)
    v.rewrite(history)
    assert len(mirror_entries(manager)) == count == len(history)


def test_observe_is_optional():
    manager, v = make(dict(max_context_tokens=100_000))
    history = []
    converse(v, history, observe=False)
    v.rewrite(history)
    assert len(mirror_entries(manager)) == len(history)


def test_system_prefix_only_changes_when_an_offload_happens():
    _, v = make()
    history = []
    previous = None
    changes = offloads = 0
    for topic in TOPICS + TOPICS:
        history.append({"role": "user", "content": f"Tell me about {topic} in depth please."})
        out = v.rewrite(history)
        offloads += bool(v.last_stats["offloaded_this_turn"])
        system = out[0]["content"] if out[0]["role"] == "system" else None
        if previous is not None and system != previous:
            changes += 1
            assert v.last_stats["offloaded_this_turn"], "system prefix changed without an offload"
        previous = system
        reply = f"Notes on {topic}: tradeoffs exist."
        v.observe(reply)
        history.append({"role": "assistant", "content": reply})
    assert offloads >= 2
    assert changes <= offloads


def test_tool_call_turns_are_never_split():
    _, v = make(dict(SMALL, min_recent_turns=1))
    history = []
    for i in range(10):
        history += [
            {"role": "user", "content": f"Look up item {i} " + "with plenty of detail " * 8},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "lookup", "arguments": f'{{"i": {i}}}'}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": f"result {i} " + "data " * 20},
            {"role": "assistant", "content": f"Item {i} looks fine."},
        ]
        out = v.rewrite(history)
        convo = [m for m in out if m["role"] != "system"]
        assert convo[0]["role"] == "user"
        for j, m in enumerate(convo):
            if m["role"] == "tool":
                assert convo[j - 1].get("tool_calls"), "tool result separated from its call"


def test_regenerate_and_edit_supersede_stale_turns():
    manager, v = make(dict(max_context_tokens=100_000))
    history = [{"role": "user", "content": "first question"}]
    v.rewrite(history)
    v.observe("first answer")
    # Regenerate: the client resends history without the reply we observed.
    v.rewrite(history)
    v.observe("second answer")
    history.append({"role": "assistant", "content": "second answer"})
    # Edit: the user changes their next message after a reply was observed.
    history.append({"role": "user", "content": "original follow-up"})
    v.rewrite(history)
    v.observe("reply to original")
    history[-1] = {"role": "user", "content": "edited follow-up"}
    v.rewrite(history)

    live = [e["content"] for e in mirror_entries(manager)]
    assert live == ["first question", "second answer", "edited follow-up"]


def test_client_side_truncation_preserves_dropped_messages_as_memory():
    manager, v = make(dict(max_context_tokens=100_000))
    history = []
    converse(v, history, observe=False)
    truncated = history[-4:]  # a client that only sends a sliding window
    truncated.append({"role": "user", "content": "Where does the hot buffer live?"})
    out = v.rewrite(truncated)
    assert v.last_stats["offloaded_messages_total"] == len(history) - 4
    # Dropped turns come back as distilled memory in the system message.
    assert out[0]["role"] == "system" and "/dev/shm" in out[0]["content"]
    assert all("/dev/shm" not in m["content"] for m in out[1:])


def test_persistent_virtualizer_resumes_after_restart(tmp_path):
    manager = TieredMemoryManager.persistent(tmp_path, session_id="resume")
    _, v = make(manager=manager)
    history = []
    converse(v, history)
    boundary = v.last_stats["offloaded_messages_total"]
    assert boundary > 0
    entries_before = len(mirror_entries(manager))

    restarted = TieredMemoryManager.persistent(tmp_path, session_id="resume")
    _, v2 = make(manager=restarted)
    v2.rewrite(history)
    assert len(mirror_entries(restarted)) == entries_before
    assert v2.last_stats["offloaded_messages_total"] >= boundary


def test_system_message_placement_option():
    _, v = make(dict(SMALL, retrieval_placement="system_message"))
    history = []
    converse(v, history)
    history.append({"role": "user", "content": "Which model does the memory unit use?"})
    out = v.rewrite(history)
    assert out[-2]["role"] == "system" and "<retrieved_memory>" in out[-2]["content"]
    assert out[-1]["content"] == "Which model does the memory unit use?"


def test_multipart_content_is_handled():
    _, v = make()
    history = []
    converse(v, history)
    history.append({"role": "user", "content": [{"type": "text", "text": "What device does the memory unit use?"},
                                                 {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]})
    out = v.rewrite(history)
    parts = out[-1]["content"]
    assert parts[0]["type"] == "text" and "<retrieved_memory>" in parts[0]["text"]
    assert parts[-1]["type"] == "image_url"


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        VirtualizerConfig(offload_target=1.2)
    with pytest.raises(ValueError):
        VirtualizerConfig(min_recent_turns=0)
    with pytest.raises(ValueError):
        VirtualizerConfig(max_context_tokens=500, reserve_tokens=500)
