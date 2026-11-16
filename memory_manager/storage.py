"""Small storage helpers shared by memory tiers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def safe_id(value: str | None, default: str = "default") -> str:
    """Return a filesystem-safe identifier."""
    value = value or default
    cleaned = "".join(c for c in value if c.isalnum() or c in ("-", "_"))[:100]
    return cleaned or default


def default_base_path(name: str) -> Path:
    """Return a durable, OS-friendly base path for canonical memory state."""
    root = Path(os.environ.get("MEMORY_MANAGER_HOME", Path.home() / ".local" / "share" / "memory_manager"))
    return root / name


def default_hot_path() -> Path:
    """Prefer Linux tmpfs for the disposable hot tier, with a portable fallback."""
    configured = os.environ.get("MEMORY_MANAGER_HOT_PATH")
    if configured:
        return Path(configured)
    shared_memory = Path("/dev/shm")
    if shared_memory.is_dir() and os.access(shared_memory, os.W_OK):
        return shared_memory / "memory_manager"
    return Path(tempfile.gettempdir()) / "memory_manager_hot"


def atomic_json_write(
    path: Path,
    data: dict[str, Any],
    security: Any = None,
) -> None:
    """Atomically write JSON by writing a temp file then replacing the target.

    If *security* is a :class:`~memory_manager.security.SecurityManager`, the
    payload is wrapped in a signed (and optionally encrypted) envelope before
    serialisation.  The final file is always written with mode ``0o600``.
    """
    import secrets as _secrets

    path.parent.mkdir(parents=True, exist_ok=True)
    if security is not None:
        data = security.sign(data)

    serialized = json.dumps(data, indent=2, default=str, sort_keys=True) + "\n"
    tmp_path = path.parent / f".{path.name}.{_secrets.token_hex(8)}.tmp"
    try:
        fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(serialized)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        os.chmod(path, 0o600)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def read_json(
    path: Path,
    default: dict[str, Any] | None = None,
    security: Any = None,
) -> dict[str, Any]:
    """Read a JSON object, returning *default* when the file is absent.

    If *security* is a :class:`~memory_manager.security.SecurityManager`, the
    envelope signature is verified before the payload is returned.  Legacy
    unsigned files are passed through with a warning.
    """
    if not path.exists():
        return default or {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if security is not None:
        data = security.verify(data)
    return data


def estimate_tokens(text: str, overhead: int = 0) -> int:
    """Code-aware token estimate for local budgeting without model tokenizers.

    Uses word count with a context-sensitive multiplier:

    * **Prose**: ~1.33 tokens per word (English text, typical BPE tokenizer).
    * **Code-heavy**: ~1.5 tokens per word (special characters, camelCase
      splitting, indentation tokens).

    Special characters that commonly tokenize as single tokens (braces,
    brackets, operators) are counted separately at 0.5 weight.

    This is still a heuristic — exact counts require a model-specific
    tokenizer — but it is substantially more accurate than ``len(text) // 4``
    for mixed code/prose content.
    """
    words = text.split()
    if not words:
        return max(0, overhead)
    # Characters that most BPE tokenizers emit as individual tokens.
    specials = sum(1 for c in text if c in "{}[]()<>@#$%^&*+=|\\~`")
    # Heuristic: ≥2 code markers → code-heavy content.
    code_markers = (
        text.count("```")
        + text.count("def ")
        + text.count("class ")
        + text.count("import ")
        + text.count("  return ")
    )
    multiplier = 1.5 if code_markers >= 2 else 1.33
    return max(0, int(len(words) * multiplier + specials * 0.5) + overhead)
