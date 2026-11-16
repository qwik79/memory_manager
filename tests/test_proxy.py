import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

import pytest

from memory_manager.proxy import MemoryProxy, ProxyConfig
from memory_manager.virtualizer import VirtualizerConfig


class FakeUpstream(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible server that records what it received."""

    received: list = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "fake-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeUpstream.received.append({"path": self.path, "payload": payload, "auth": self.headers.get("Authorization")})
        last = payload["messages"][-1]["content"]
        reply = f"echo: {str(last)[-40:]}"
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in (reply[:10], reply[10:]):
                chunk = {"choices": [{"index": 0, "delta": {"content": piece}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            body = json.dumps({"choices": [{"index": 0, "message": {"role": "assistant", "content": reply}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def serve(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@pytest.fixture
def stack():
    FakeUpstream.received = []
    upstream = serve(ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream))
    config = ProxyConfig(
        upstream=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
        port=0,
        virtualizer=VirtualizerConfig(max_context_tokens=900, reserve_tokens=150,
                                      history_budget_tokens=120, retrieval_budget_tokens=150),
    )
    proxy = MemoryProxy(config)
    server = serve(proxy.make_server())
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield proxy, base
    server.shutdown()
    upstream.shutdown()
    proxy.close()


def post(url, payload, headers=None):
    request = Request(url, data=json.dumps(payload).encode(), method="POST",
                      headers={"Content-Type": "application/json", **(headers or {})})
    with urlopen(request, timeout=10) as response:
        return response.read().decode()


def chat(base, history, stream=False, headers=None):
    raw = post(f"{base}/v1/chat/completions", {"model": "m", "messages": history, "stream": stream}, headers)
    if not stream:
        return json.loads(raw)["choices"][0]["message"]["content"]
    text = ""
    for line in raw.splitlines():
        if line.startswith("data:") and "[DONE]" not in line:
            text += json.loads(line[5:])["choices"][0]["delta"].get("content", "")
    return text


@pytest.mark.parametrize("stream", [False, True])
def test_proxy_extends_context_transparently(stack, stream):
    proxy, base = stack
    history = [{"role": "system", "content": "Be brief."}]
    for i in range(10):
        history.append({"role": "user", "content": f"Fact {i}: the code word for item{i} is word{i}. " + "context " * 25})
        reply = chat(base, history, stream=stream, headers={"Authorization": "Bearer k"})
        assert reply.startswith("echo:")
        history.append({"role": "assistant", "content": reply})

    sent = FakeUpstream.received[-1]["payload"]["messages"]
    assert len(sent) < len(history)  # history was trimmed upstream...
    assert sent[0]["content"].startswith("Be brief.")
    assert FakeUpstream.received[-1]["auth"] == "Bearer k"
    assert FakeUpstream.received[-1]["path"] == "/v1/chat/completions"

    history.append({"role": "user", "content": "What is the code word for item1?"})
    chat(base, history, stream=stream)
    sent = FakeUpstream.received[-1]["payload"]["messages"]
    assert "word1" in json.dumps(sent)  # ...but the old fact still reaches the model

    sessions = proxy.session_ids()
    assert len(sessions) == 1 and sessions[0].startswith("auto-")
    virtualizer = proxy.session(sessions[0]).virtualizer
    assert virtualizer.last_stats["offloaded_messages_total"] > 0


def test_auto_sessions_split_conversations_and_header_overrides(stack):
    proxy, base = stack
    chat(base, [{"role": "user", "content": "conversation one"}])
    chat(base, [{"role": "user", "content": "conversation two"}])
    chat(base, [{"role": "user", "content": "pinned"}], headers={"X-Memory-Session": "my project"})
    sessions = proxy.session_ids()
    assert len(sessions) == 3 and "myproject" in sessions


def test_passthrough_routes_and_session_management(stack):
    proxy, base = stack
    with urlopen(f"{base}/v1/models", timeout=10) as response:
        assert json.loads(response.read())["data"][0]["id"] == "fake-model"

    chat(base, [{"role": "user", "content": "hello"}], headers={"X-Memory-Session": "s1"})
    with urlopen(f"{base}/memory/sessions", timeout=10) as response:
        assert json.loads(response.read())["sessions"] == ["s1"]
    manager = proxy.session("s1").manager

    request = Request(f"{base}/memory/sessions/s1", method="DELETE")
    with urlopen(request, timeout=10) as response:
        assert json.loads(response.read())["ended"] is True
    assert proxy.session_ids() == []
    assert manager.hot_buffer.get_entries(limit=None, include_compacted=True) == []  # ephemeral: wiped


def test_upstream_down_returns_502():
    proxy = MemoryProxy(ProxyConfig(upstream="http://127.0.0.1:9/v1", port=0))
    server = serve(proxy.make_server())
    try:
        with pytest.raises(Exception) as info:
            post(f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions",
                 {"messages": [{"role": "user", "content": "hi"}]})
        assert getattr(info.value, "code", None) == 502
    finally:
        server.shutdown()


def test_idle_sessions_are_reaped():
    proxy = MemoryProxy(ProxyConfig(idle_timeout=0.01))
    sess = proxy.session("old")
    sess.manager.add_user_message("forget me")
    sess.last_used -= 10
    assert proxy.reap_idle() == ["old"]
    assert sess.manager.closed
