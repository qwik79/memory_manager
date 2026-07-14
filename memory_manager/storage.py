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
    """Return an OS-friendly default base path for local memory state."""
    return Path(tempfile.gettempdir()) / name


def atomic_json_write(path: Path, data: dict[str, Any]) -> None:
    """Atomically write JSON by writing a temp file then replacing the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_file.name)
    try:
        with tmp_file as f:
            json.dump(data, f, indent=2, default=str, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read a JSON object, returning default when the file is absent."""
    if not path.exists():
        return default or {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def estimate_tokens(text: str, overhead: int = 0) -> int:
    """Conservative token estimate for local budgeting without model tokenizers."""
    return max(0, len(text) // 4 + overhead)
