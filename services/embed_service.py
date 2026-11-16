"""Lightweight local embedding service wrapping Ollama's /api/embeddings endpoint.

This is an optional sidecar for the memory manager's semantic drift scoring
and ChromaDB cold store.  It provides a simple HTTP POST endpoint that batches
embedding requests to Ollama.

Requirements
------------
- Ollama running on localhost:11434
- ``ollama pull nomic-embed-text`` (768-dim, fast on CPU)

Usage
-----
Start the service::

    python -m services.embed_service
    # or:
    python services/embed_service.py

Then use it via the client factory::

    from memory_manager import make_embed_fn
    embed_fn = make_embed_fn()  # connects to http://127.0.0.1:8001

API
---
POST /
    Body: {"texts": ["text1", "text2", ...]}
    Response: {"embeddings": [{"text": "text1", "vector": [0.1, ...]}, ...]}
"""

from __future__ import annotations

import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import Request, urlopen

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("embed_service")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
REQUEST_TIMEOUT = 30.0


class EmbedHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            texts: list[str] = body.get("texts", [])

            results: list[dict] = []
            for text in texts:
                vector = _embed_one(text)
                results.append({"text": text, "vector": vector})

            payload = json.dumps({"embeddings": results}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        except Exception as exc:
            logger.exception("Embed request failed: %s", exc)
            error = json.dumps({"error": str(exc)}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(error)))
            self.end_headers()
            self.wfile.write(error)

    def do_GET(self) -> None:
        """Health check."""
        body = json.dumps({"status": "ok", "model": EMBED_MODEL}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        logger.debug(format, *args)


def _embed_one(text: str) -> list[float]:
    """Embed a single string via Ollama."""
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    req = Request(
        OLLAMA_EMBED_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read())["embedding"]


def main(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    server = HTTPServer((host, port), EmbedHandler)
    logger.info("Embedding service started on http://%s:%d", host, port)
    logger.info("Model: %s  |  Ollama: %s", EMBED_MODEL, OLLAMA_EMBED_URL)
    logger.info("Pull model with: ollama pull %s", EMBED_MODEL)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    main(port=port)
