"""Document stores the memory tiers persist through.

Every tier stores JSON documents at paths (one hot-buffer file per session, one
file per warm section or cold chunk). A :class:`DocStore` owns *where* those
documents live; the tier logic never touches the filesystem directly.

* :class:`FileDocStore` writes real files via :func:`atomic_json_write` and
  :func:`read_json`, including the optional HMAC/Fernet security layer. This is
  persistent mode.
* :class:`MemoryDocStore` keeps the same documents in a dict keyed by path. Paths
  are purely virtual: nothing is ever created on disk, so an ephemeral session
  leaves no trace once it is closed or the process exits (crash and SIGKILL
  included). Security envelopes are skipped because the data never leaves the
  process.
"""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, runtime_checkable

from .storage import atomic_json_write, read_json

#: Virtual root used for in-memory stores. It is only ever a dictionary key.
MEMORY_ROOT = Path("memory:")


@runtime_checkable
class DocStore(Protocol):
    """Minimal JSON-document interface the tiers use instead of the filesystem."""

    persistent: bool

    def read(self, path: Path, default: dict[str, Any] | None = None) -> dict[str, Any] | None: ...

    def write(self, path: Path, data: dict[str, Any]) -> None: ...

    def exists(self, path: Path) -> bool: ...

    def delete(self, path: Path) -> bool: ...

    def list(self, directory: Path) -> list[Path]:
        """``*.json`` documents directly inside *directory*, sorted by name."""
        ...

    def walk(self, directory: Path) -> list[Path]:
        """All documents anywhere under *directory*, sorted."""
        ...

    def ensure_dir(self, directory: Path) -> None: ...

    def size(self, path: Path) -> int: ...


class FileDocStore:
    """Real files on disk, optionally signed/encrypted by a SecurityManager."""

    persistent = True

    def __init__(self, security: Any = None) -> None:
        self.security = security

    def read(self, path: Path, default: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not path.exists():
            return default
        return read_json(path, security=self.security)

    def write(self, path: Path, data: dict[str, Any]) -> None:
        atomic_json_write(path, data, security=self.security)

    def exists(self, path: Path) -> bool:
        return path.exists()

    def delete(self, path: Path) -> bool:
        if path.exists():
            path.unlink(missing_ok=True)
            return True
        return False

    def list(self, directory: Path) -> list[Path]:
        return sorted(directory.glob("*.json")) if directory.exists() else []

    def walk(self, directory: Path) -> list[Path]:
        return sorted(p for p in directory.rglob("*.json") if p.is_file()) if directory.exists() else []

    def ensure_dir(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)

    def size(self, path: Path) -> int:
        return path.stat().st_size if path.exists() else 0


class MemoryDocStore:
    """Process-memory documents keyed by (virtual) path. Never touches disk."""

    persistent = False
    security = None

    def __init__(self) -> None:
        self._docs: dict[PurePosixPath, dict[str, Any]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(path: Path) -> PurePosixPath:
        return PurePosixPath(Path(path).as_posix())

    def read(self, path: Path, default: dict[str, Any] | None = None) -> dict[str, Any] | None:
        with self._lock:
            doc = self._docs.get(self._key(path))
            return copy.deepcopy(doc) if doc is not None else default

    def write(self, path: Path, data: dict[str, Any]) -> None:
        # Round-trip through JSON so stored data has exactly the shape a file
        # store would return (tuples become lists, non-JSON values are str()-ed).
        snapshot = json.loads(json.dumps(data, default=str))
        with self._lock:
            self._docs[self._key(path)] = snapshot

    def exists(self, path: Path) -> bool:
        with self._lock:
            return self._key(path) in self._docs

    def delete(self, path: Path) -> bool:
        with self._lock:
            return self._docs.pop(self._key(path), None) is not None

    def list(self, directory: Path) -> list[Path]:
        parent = self._key(directory)
        with self._lock:
            keys = [k for k in self._docs if k.parent == parent and k.suffix == ".json"]
        return [Path(str(k)) for k in sorted(keys, key=lambda k: k.name)]

    def walk(self, directory: Path) -> list[Path]:
        root = self._key(directory)
        with self._lock:
            keys = [k for k in self._docs if root in k.parents]
        return [Path(str(k)) for k in sorted(keys)]

    def ensure_dir(self, directory: Path) -> None:
        return None

    def size(self, path: Path) -> int:
        doc = self.read(path)
        return len(json.dumps(doc).encode("utf-8")) if doc is not None else 0

    def clear(self) -> int:
        with self._lock:
            count = len(self._docs)
            self._docs.clear()
            return count

    def __len__(self) -> int:
        with self._lock:
            return len(self._docs)
