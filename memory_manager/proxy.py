"""OpenAI-compatible context-extension proxy.

Point any OpenAI-compatible client at this proxy instead of your inference
server. Chat requests are rewritten by a per-session :class:`ContextVirtualizer`
and forwarded upstream; every other route is passed through untouched. The proxy
does no inference itself, so the backend (Ollama, LM Studio, llama.cpp server,
vLLM, ...) stays interchangeable.

    python -m memory_manager.proxy --upstream http://localhost:11434/v1 --context-tokens 8192
    # then point your client at http://127.0.0.1:8787/v1

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .orchestrator import TieredMemoryManager
from .storage import safe_id
from .virtualizer import SYSTEM_ROLES, ContextVirtualizer, VirtualizerConfig, fingerprint

DEFAULT_SESSION_HEADERS = ("X-Memory-Session",)
_FORWARD_REQUEST_HEADERS = ("authorization", "content-type", "accept", "openai-organization", "openai-project")
_SKIP_RESPONSE_HEADERS = {"connection", "keep-alive", "transfer-encoding", "content-length", "content-encoding", "date", "server"}


@dataclass
class ProxyConfig:
    upstream: str = "http://localhost:11434/v1"
    host: str = "127.0.0.1"
    port: int = 8787
    session: str = "auto"
    """``auto`` derives one session per conversation; any other value pins every request to that session."""
    session_headers: tuple[str, ...] = DEFAULT_SESSION_HEADERS
    """Request headers checked (in order) for an explicit session ID."""
    persist_dir: str | None = None
    """Directory for persistent memory; ``None`` keeps every session ephemeral (in RAM only)."""
    secure: bool = False
    """HMAC-sign persisted files (persistent mode only)."""
    embed_model: str | None = None
    """Ollama embedding model (e.g. ``nomic-embed-text``) for hybrid semantic + lexical retrieval."""
    embed_url: str = "http://127.0.0.1:11434/api/embeddings"
    idle_timeout: float = 3600.0
    """Close sessions idle this long (seconds; 0 disables). Ephemeral sessions are wiped on close."""
    upstream_timeout: float = 600.0
    virtualizer: VirtualizerConfig = field(default_factory=VirtualizerConfig)
    verbose: bool = False


@dataclass
class _Session:
    manager: TieredMemoryManager
    virtualizer: ContextVirtualizer
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_used: float = field(default_factory=time.time)


class MemoryProxy:
    """Session registry plus request handling; ``make_server()`` returns a ready HTTP server."""

    def __init__(self, config: ProxyConfig | None = None):
        self.config = config or ProxyConfig()
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()
        self._embed_fn = None
        if self.config.embed_model:
            from .embeddings import make_ollama_embed_fn

            self._embed_fn = make_ollama_embed_fn(model=self.config.embed_model, endpoint=self.config.embed_url)

    def _new_manager(self, session_id: str) -> TieredMemoryManager:
        kwargs = {"embed_fn": self._embed_fn} if self._embed_fn else {}
        if self.config.persist_dir:
            return TieredMemoryManager.persistent(
                self.config.persist_dir, session_id=session_id, secure=self.config.secure, **kwargs
            )
        return TieredMemoryManager.ephemeral(session_id, **kwargs)

    # ------------------------------------------------------------------ #
    # Sessions
    # ------------------------------------------------------------------ #

    def resolve_session_id(self, headers: Any, messages: list[dict[str, Any]]) -> str:
        for name in self.config.session_headers:
            value = headers.get(name)
            if value:
                return safe_id(value)
        if self.config.session != "auto":
            return safe_id(self.config.session)
        # One session per conversation: keyed on the system prompt plus the first non-system message.
        system = next((m for m in messages if m.get("role") in SYSTEM_ROLES), {})
        first = next((m for m in messages if m.get("role") not in SYSTEM_ROLES), {})
        return "auto-" + fingerprint({"role": "key", "content": fingerprint(system) + fingerprint(first)})[:20]

    def session(self, session_id: str) -> _Session:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                manager = self._new_manager(session_id)
                sess = _Session(manager, ContextVirtualizer(manager, self.config.virtualizer))
                self._sessions[session_id] = sess
            sess.last_used = time.time()
            return sess

    def session_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._sessions)

    def end_session(self, session_id: str, wipe: bool = False) -> bool:
        """Close a session. Ephemeral memory is always wiped; ``wipe`` also deletes persistent memory."""
        with self._lock:
            sess = self._sessions.pop(safe_id(session_id), None)
        if sess is None:
            if wipe and self.config.persist_dir:
                manager = self._new_manager(safe_id(session_id))
                manager.wipe()
                manager.close()
                return True
            return False
        with sess.lock:
            if wipe:
                sess.manager.wipe()
            sess.manager.close()
        return True

    def reap_idle(self) -> list[str]:
        timeout = self.config.idle_timeout
        if timeout <= 0:
            return []
        cutoff = time.time() - timeout
        with self._lock:
            idle = [sid for sid, s in self._sessions.items() if s.last_used < cutoff and not s.lock.locked()]
        closed = [sid for sid in idle if self.end_session(sid)]
        if closed:
            self.log(f"closed idle sessions: {', '.join(closed)}")
        return closed

    def close(self) -> None:
        for sid in self.session_ids():
            self.end_session(sid)

    # ------------------------------------------------------------------ #
    # Plumbing
    # ------------------------------------------------------------------ #

    def upstream_url(self, path_and_query: str) -> str:
        parts = urlsplit(path_and_query)
        path = parts.path
        if path == "/v1" or path.startswith("/v1/"):
            path = path[3:]
        url = self.config.upstream.rstrip("/") + path
        return f"{url}?{parts.query}" if parts.query else url

    def log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[memory-proxy] {message}", file=sys.stderr, flush=True)

    def make_server(self) -> ThreadingHTTPServer:
        proxy = self

        class Handler(_ProxyHandler):
            pass

        Handler.proxy = proxy
        server = ThreadingHTTPServer((self.config.host, self.config.port), Handler)
        server.daemon_threads = True
        return server

    def serve_forever(self) -> None:
        server = self.make_server()
        host, port = server.server_address[:2]
        mode = f"persistent ({self.config.persist_dir})" if self.config.persist_dir else "ephemeral"
        print(f"memory proxy on http://{host}:{port}/v1 -> {self.config.upstream} [{mode} memory]", file=sys.stderr)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            self.close()


class _ProxyHandler(BaseHTTPRequestHandler):
    proxy: MemoryProxy
    protocol_version = "HTTP/1.0"  # close after each response, so streams need no chunked encoding

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        self.proxy.log(format % args)

    # ---- routing ------------------------------------------------------ #

    def do_GET(self) -> None:
        if urlsplit(self.path).path.rstrip("/") == "/memory/sessions":
            self._send_json(200, {"sessions": self.proxy.session_ids()})
        else:
            self._forward()

    def do_DELETE(self) -> None:
        parts = urlsplit(self.path)
        if parts.path.startswith("/memory/sessions/"):
            session_id = parts.path[len("/memory/sessions/"):].strip("/")
            ended = self.proxy.end_session(session_id, wipe="wipe=1" in parts.query)
            self._send_json(200 if ended else 404, {"session": safe_id(session_id), "ended": ended})
        else:
            self._forward()

    def do_POST(self) -> None:
        self.proxy.reap_idle()
        if urlsplit(self.path).path.rstrip("/").endswith("/chat/completions"):
            self._chat()
        else:
            self._forward()

    # ---- chat --------------------------------------------------------- #

    def _chat(self) -> None:
        raw = self._read_body()
        try:
            body = json.loads(raw or b"{}")
            messages = body.get("messages")
            if not isinstance(messages, list):
                raise ValueError("'messages' must be a list")
        except (ValueError, AttributeError) as exc:
            self._send_json(400, {"error": {"message": f"memory proxy: invalid request body: {exc}"}})
            return

        session_id = self.proxy.resolve_session_id(self.headers, messages)
        sess = self.proxy.session(session_id)
        with sess.lock:
            payload = dict(body)
            payload["messages"] = sess.virtualizer.rewrite(messages)
            self.proxy.log(f"{session_id}: {json.dumps(sess.virtualizer.last_stats)}")
            upstream = self._open_upstream("POST", json.dumps(payload).encode("utf-8"))
            if upstream is None:
                return
            with upstream:
                if body.get("stream"):
                    reply = self._relay_stream(upstream)
                else:
                    reply = self._relay_completion(upstream)
            if reply is not None:
                sess.virtualizer.observe(reply)
            sess.last_used = time.time()

    def _relay_completion(self, upstream: Any) -> dict[str, Any] | None:
        data = upstream.read()
        self._send_raw(upstream.status, upstream.headers, data)
        try:
            return json.loads(data)["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError):
            return None

    def _relay_stream(self, upstream: Any) -> dict[str, Any] | None:
        self.send_response(upstream.status)
        for key, value in upstream.headers.items():
            if key.lower() not in _SKIP_RESPONSE_HEADERS:
                self.send_header(key, value)
        self.end_headers()
        content: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        try:
            for line in upstream:
                self.wfile.write(line)
                self.wfile.flush()
                _accumulate_sse(line, content, tool_calls)
        except (BrokenPipeError, ConnectionResetError):
            self.proxy.log("client disconnected mid-stream; reply not recorded")
            return None
        message: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        return message

    # ---- pass-through ------------------------------------------------- #

    def _forward(self) -> None:
        data = self._read_body() if self.command in ("POST", "PUT", "PATCH") else None
        upstream = self._open_upstream(self.command, data)
        if upstream is None:
            return
        with upstream:
            if "text/event-stream" in (upstream.headers.get("Content-Type") or ""):
                self._relay_stream(upstream)
            else:
                self._send_raw(upstream.status, upstream.headers, upstream.read())

    def _open_upstream(self, method: str, data: bytes | None) -> Any:
        headers = {k: v for k, v in self.headers.items() if k.lower() in _FORWARD_REQUEST_HEADERS}
        if data is not None:
            headers.setdefault("Content-Type", "application/json")
        request = Request(self.proxy.upstream_url(self.path), data=data, headers=headers, method=method)
        try:
            return urlopen(request, timeout=self.proxy.config.upstream_timeout)
        except HTTPError as exc:
            try:
                self._send_raw(exc.code, exc.headers, exc.read())
            finally:
                exc.close()
        except (URLError, OSError) as exc:
            self._send_json(502, {"error": {"message": f"memory proxy: upstream unreachable: {exc}"}})
        return None

    # ---- io helpers --------------------------------------------------- #

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _send_raw(self, status: int, headers: Any, data: bytes) -> None:
        self.send_response(status)
        for key, value in (headers.items() if headers else []):
            if key.lower() not in _SKIP_RESPONSE_HEADERS:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send_raw(status, {"Content-Type": "application/json"}, json.dumps(payload).encode("utf-8"))


def _accumulate_sse(line: bytes, content: list[str], tool_calls: dict[int, dict[str, Any]]) -> None:
    text = line.decode("utf-8", errors="replace").strip()
    if not text.startswith("data:"):
        return
    data = text[5:].strip()
    if not data or data == "[DONE]":
        return
    try:
        delta = json.loads(data)["choices"][0].get("delta") or {}
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return
    if isinstance(delta.get("content"), str):
        content.append(delta["content"])
    for call in delta.get("tool_calls") or []:
        slot = tool_calls.setdefault(
            int(call.get("index", 0)), {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
        )
        if call.get("id"):
            slot["id"] = call["id"]
        fn = call.get("function") or {}
        slot["function"]["name"] += fn.get("name") or ""
        slot["function"]["arguments"] += fn.get("arguments") or ""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible context-extension proxy for local LLM servers")
    parser.add_argument("--upstream", default="http://localhost:11434/v1",
                        help="upstream OpenAI-compatible base URL (Ollama: http://localhost:11434/v1, LM Studio: http://localhost:1234/v1)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--context-tokens", type=int, default=8192, help="the backend model's context window (e.g. Ollama num_ctx)")
    parser.add_argument("--reserve-tokens", type=int, default=1024, help="room to leave for the reply")
    parser.add_argument("--session", default="auto", help="'auto' (one session per conversation) or a fixed session name")
    parser.add_argument("--session-header", action="append", dest="session_headers",
                        help=f"request header carrying a session ID; repeatable (default: {DEFAULT_SESSION_HEADERS[0]})")
    parser.add_argument("--persist", metavar="DIR", help="opt in to persistent memory under DIR (default: ephemeral, RAM only)")
    parser.add_argument("--secure", action="store_true", help="HMAC-sign persisted files (with --persist)")
    parser.add_argument("--embed-model", metavar="MODEL",
                        help="Ollama embedding model (e.g. nomic-embed-text) for semantic retrieval; default is lexical only")
    parser.add_argument("--embed-url", default="http://127.0.0.1:11434/api/embeddings", help="Ollama embeddings endpoint")
    parser.add_argument("--idle-timeout", type=float, default=3600, help="close sessions idle this many seconds (0 = never)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    config = ProxyConfig(
        upstream=args.upstream,
        host=args.host,
        port=args.port,
        session=args.session,
        session_headers=tuple(args.session_headers or DEFAULT_SESSION_HEADERS),
        persist_dir=args.persist,
        secure=args.secure,
        embed_model=args.embed_model,
        embed_url=args.embed_url,
        idle_timeout=args.idle_timeout,
        virtualizer=VirtualizerConfig(max_context_tokens=args.context_tokens, reserve_tokens=args.reserve_tokens),
        verbose=args.verbose,
    )
    MemoryProxy(config).serve_forever()


if __name__ == "__main__":
    main()
