"""Client-side embedding factory for the memory manager.

Provides a zero-dependency ``embed_fn`` compatible with
``TieredMemoryManager(embed_fn=...)``.  Connects to the lightweight
embedding service (``services/embed_service.py``) which wraps Ollama's
``/api/embeddings`` endpoint.

Usage::

    from memory_manager import TieredMemoryManager, make_embed_fn

    manager = TieredMemoryManager(
        "my_session",
        embed_fn=make_embed_fn(),  # default: http://127.0.0.1:8001
    )

To start the embedding service::

    python -m services.embed_service
    # or: python services/embed_service.py
"""

from __future__ import annotations

import json
from typing import Callable
from urllib.request import Request, urlopen


def make_embed_fn(
    endpoint: str = "http://127.0.0.1:8001",
    timeout: float = 15.0,
) -> Callable[[str], list[float]]:
    """Return an ``embed_fn`` that calls the local embedding service.

    Parameters
    ----------
    endpoint:
        Base URL of the embedding service.  Default is the local service
        started by ``python -m services.embed_service``.
    timeout:
        Network timeout in seconds.  Embedding a short string takes <1s
        locally; increase for long chunks or slow hardware.

    Returns
    -------
    Callable[[str], list[float]]
        A single-argument function suitable for passing to
        ``TieredMemoryManager(embed_fn=...)``.
    """

    def embed_fn(text: str) -> list[float]:
        payload = json.dumps({"texts": [text]}).encode()
        req = Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return data["embeddings"][0]["vector"]

    return embed_fn


def make_ollama_embed_fn(
    model: str = "nomic-embed-text",
    endpoint: str = "http://127.0.0.1:11434/api/embeddings",
    timeout: float = 15.0,
) -> Callable[[str], list[float]]:
    """Return an ``embed_fn`` that calls Ollama's embeddings API directly.

    Use this when you don't want to run the separate embedding service.
    Requires ``ollama pull nomic-embed-text`` (or another embedding model).

    Parameters
    ----------
    model:
        Ollama embedding model name.  ``nomic-embed-text`` produces 768-dim
        vectors and is very fast on CPU.
    endpoint:
        Ollama embeddings API URL.
    timeout:
        Network timeout in seconds.
    """

    def embed_fn(text: str) -> list[float]:
        payload = json.dumps({"model": model, "prompt": text}).encode()
        req = Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())["embedding"]

    return embed_fn
