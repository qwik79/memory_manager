"""Model availability helpers for Ollama-based workflows.

Provides ``ensure_model`` and ``ensure_embedding_model`` for checking whether
a model is available locally and auto-pulling it if not.  Uses only stdlib
(``urllib``) — no Ollama SDK required.

Usage::

    from memory_manager.model_utils import ensure_model, ensure_embedding_model

    # Checks Ollama, pulls if missing. No-op if Ollama isn't running.
    ensure_model("qwen2.5:1.5b")
    ensure_embedding_model("nomic-embed-text")
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# Default Ollama API base.
_OLLAMA_BASE = "http://127.0.0.1:11434"


def is_ollama_running(base_url: str = _OLLAMA_BASE, timeout: float = 3.0) -> bool:
    """Return True if the Ollama API is reachable."""
    try:
        with urlopen(f"{base_url}/api/version", timeout=timeout) as resp:
            return resp.status == 200
    except (URLError, OSError, TimeoutError):
        return False


def model_exists(
    model: str,
    base_url: str = _OLLAMA_BASE,
    timeout: float = 5.0,
) -> bool:
    """Return True if a model is already pulled locally."""
    try:
        req = Request(
            f"{base_url}/api/show",
            data=json.dumps({"name": model}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (URLError, OSError, TimeoutError):
        return False


def pull_model(
    model: str,
    base_url: str = _OLLAMA_BASE,
    timeout: float = 600.0,
    on_progress: Callable[[str], None] | None = None,
) -> bool:
    """Pull a model from the Ollama registry.

    Parameters
    ----------
    model:
        Model name (e.g. ``"qwen2.5:1.5b"``, ``"nomic-embed-text"``).
    base_url:
        Ollama API base URL.
    timeout:
        Total timeout for the pull (default 10 minutes).
    on_progress:
        Optional callback that receives progress status strings.

    Returns
    -------
    bool
        True if the pull succeeded.
    """
    try:
        req = Request(
            f"{base_url}/api/pull",
            data=json.dumps({"name": model, "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=timeout) as resp:
            for line in resp:
                try:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if on_progress and status:
                        # Build a nice progress string.
                        total = data.get("total", 0)
                        completed = data.get("completed", 0)
                        if total:
                            pct = int(completed / total * 100)
                            on_progress(f"{status}: {pct}%")
                        else:
                            on_progress(status)
                except (json.JSONDecodeError, KeyError):
                    pass
        return True
    except (URLError, OSError, TimeoutError) as exc:
        logger.warning("Failed to pull model '%s': %s", model, exc)
        return False


def ensure_model(
    model: str = "qwen2.5:1.5b",
    base_url: str = _OLLAMA_BASE,
    on_progress: Callable[[str], None] | None = None,
) -> bool:
    """Ensure a model is available locally, pulling it if necessary.

    Parameters
    ----------
    model:
        Ollama model name.  Recommended distillation models:

        - ``qwen2.5:1.5b`` — fast, 1.5B params, good quality/speed ratio
        - ``qwen2.5:3b`` — better quality, still fits in 4 GB RAM
        - ``phi-3:mini`` — Microsoft, 3.8B, strong reasoning
        - ``gemma2:2b`` — Google, 2B, efficient
    base_url:
        Ollama API base URL.
    on_progress:
        Optional callback for download progress.

    Returns
    -------
    bool
        True if the model is available (was already present or pulled
        successfully).  False if Ollama isn't running or the pull failed.
    """
    if not is_ollama_running(base_url):
        logger.info("Ollama is not running at %s — skipping model check.", base_url)
        return False

    if model_exists(model, base_url):
        logger.debug("Model '%s' is already available.", model)
        return True

    logger.info("Model '%s' not found — pulling from registry...", model)
    if on_progress is None:
        # Default: print to stderr for CLI visibility.
        def on_progress(msg: str) -> None:
            print(f"  ↓ {model}: {msg}", file=sys.stderr, flush=True)

    return pull_model(model, base_url, on_progress=on_progress)


def ensure_embedding_model(
    model: str = "nomic-embed-text",
    base_url: str = _OLLAMA_BASE,
    on_progress: Callable[[str], None] | None = None,
) -> bool:
    """Ensure an embedding model is available locally.

    Parameters
    ----------
    model:
        Embedding model name.  Recommended:

        - ``nomic-embed-text`` — 768-dim, fast, good quality (default)
        - ``all-minilm`` — 384-dim, very fast, lighter
        - ``mxbai-embed-large`` — 1024-dim, highest quality, slower
    base_url:
        Ollama API base URL.
    on_progress:
        Optional callback for download progress.

    Returns
    -------
    bool
        True if the model is available.
    """
    return ensure_model(model, base_url, on_progress=on_progress)
