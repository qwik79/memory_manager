"""Tamper-detection layer for all memory tier files.

Every JSON file written through this module is wrapped in a signed envelope::

    {
        "env": 1,
        "alg": "hmac-sha256",
        "signed_at": "2026-08-14T00:00:00Z",
        "payload": "<base64(json bytes)>",
        "sig": "<hex HMAC-SHA256>"
    }

On read, the signature is verified with ``hmac.compare_digest`` before the
payload is deserialised.  A mismatch raises ``TamperDetectedError`` so the
caller can decide whether to abort or fall back to a higher tier.

Key storage priority:
  1. OS keyring via the optional ``keyring`` package.
  2. A 600-permission key file under ``~/.config/memory_manager/keys/``.

Optional Fernet encryption (``pip install memory-manager[encrypt]``) adds
confidentiality on top of the HMAC; the same key material is used.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
from pathlib import Path
from typing import Any

from .models import utc_now_iso

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class TamperDetectedError(Exception):
    """Raised when a stored memory file fails signature verification."""


# ---------------------------------------------------------------------------
# Key management helpers
# ---------------------------------------------------------------------------


def _config_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "memory_manager" / "keys"


def _load_key_file(key_id: str) -> bytes | None:
    path = _config_dir() / f"{key_id}.key"
    if path.exists():
        try:
            return path.read_bytes()
        except OSError:
            return None
    return None


def _save_key_file(key_id: str, key: bytes) -> None:
    key_dir = _config_dir()
    key_dir.mkdir(parents=True, exist_ok=True)
    path = key_dir / f"{key_id}.key"
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(key)
    except Exception:
        # fd ownership passed to fdopen; if that failed the fd is already closed
        if path.exists():
            path.unlink(missing_ok=True)
        raise


def _try_keyring_load(key_id: str) -> bytes | None:
    try:
        import keyring  # type: ignore[import]

        value = keyring.get_password("memory_manager", key_id)
        if value:
            return bytes.fromhex(value)
    except Exception:
        pass
    return None


def _try_keyring_save(key_id: str, key: bytes) -> bool:
    try:
        import keyring  # type: ignore[import]

        keyring.set_password("memory_manager", key_id, key.hex())
        return True
    except Exception:
        return False


def _get_or_create_key(key_id: str) -> bytes:
    """Load the HMAC signing key, creating and persisting one if absent."""
    key = _try_keyring_load(key_id)
    if key:
        return key
    key = _load_key_file(key_id)
    if key:
        return key
    key = secrets.token_bytes(32)
    if not _try_keyring_save(key_id, key):
        _save_key_file(key_id, key)
        logger.info("Memory signing key written to %s", _config_dir() / f"{key_id}.key")
    else:
        logger.info("Memory signing key stored in OS keyring (key_id=%s)", key_id)
    return key


# ---------------------------------------------------------------------------
# SecurityManager
# ---------------------------------------------------------------------------


_MANAGER_CACHE: dict[str, SecurityManager] = {}
_CACHE_LOCK = threading.Lock()


def get_security_manager(key_id: str = "default", encrypt: bool = False) -> SecurityManager:
    """Return a cached ``SecurityManager`` for the given *key_id*."""
    cache_key = f"{key_id}:{encrypt}"
    with _CACHE_LOCK:
        if cache_key not in _MANAGER_CACHE:
            _MANAGER_CACHE[cache_key] = SecurityManager(key_id=key_id, encrypt=encrypt)
        return _MANAGER_CACHE[cache_key]


class SecurityManager:
    """Signs, verifies, and optionally encrypts memory tier files.

    Callers should generally use :func:`get_security_manager` rather than
    constructing this class directly, so the same key is shared across all
    tier managers in a process.
    """

    def __init__(self, key_id: str = "default", encrypt: bool = False) -> None:
        self.key_id = key_id
        self.encrypt = encrypt
        self._key: bytes = _get_or_create_key(key_id)
        self._fernet: Any = None
        if encrypt:
            self._init_fernet()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_fernet(self) -> None:
        try:
            from cryptography.fernet import Fernet  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Install the 'cryptography' package for encryption support: "
                "pip install 'memory-manager[encrypt]'"
            ) from exc
        # Derive a 32-byte key from the raw signing key and wrap it for Fernet.
        derived = hashlib.sha256(b"fernet:" + self._key).digest()
        fernet_key = base64.urlsafe_b64encode(derived)
        self._fernet = Fernet(fernet_key)

    def _hmac(self, data: str) -> str:
        return hmac.new(self._key, data.encode(), hashlib.sha256).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sign(self, data: dict[str, Any]) -> dict[str, Any]:
        """Wrap *data* in a signed (and optionally encrypted) envelope."""
        payload_bytes = json.dumps(data, sort_keys=True, default=str).encode()

        if self.encrypt and self._fernet is not None:
            token = self._fernet.encrypt(payload_bytes)
            payload_b64 = base64.b64encode(token).decode()
            alg = "fernet"
        else:
            payload_b64 = base64.b64encode(payload_bytes).decode()
            alg = "hmac-sha256"

        sig = self._hmac(payload_b64)
        return {
            "env": 1,
            "alg": alg,
            "signed_at": utc_now_iso(),
            "payload": payload_b64,
            "sig": sig,
        }

    def verify(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Verify *envelope* and return the inner payload dict.

        Raises :class:`TamperDetectedError` if the signature does not match.
        Legacy files (no ``"env"`` key) are returned as-is with a warning;
        they will be signed on the next write.
        """
        if "env" not in envelope:
            logger.warning(
                "Loading unsigned legacy memory file — it will be signed on next write."
            )
            return envelope

        payload_b64: str = envelope.get("payload", "")
        stored_sig: str = envelope.get("sig", "")
        alg: str = envelope.get("alg", "hmac-sha256")

        actual_sig = self._hmac(payload_b64)
        if not hmac.compare_digest(actual_sig, stored_sig):
            raise TamperDetectedError(
                f"Memory file signature verification failed (alg={alg}). "
                "The file may have been tampered with. Refusing to load."
            )

        payload_bytes = base64.b64decode(payload_b64)

        if alg == "fernet":
            if self._fernet is None:
                raise TamperDetectedError(
                    "File was written with Fernet encryption but encryption is not "
                    "currently enabled. Initialise TieredMemoryManager with encrypt=True."
                )
            try:
                payload_bytes = self._fernet.decrypt(payload_bytes)
            except Exception as exc:
                raise TamperDetectedError(f"Fernet decryption failed: {exc}") from exc

        return json.loads(payload_bytes)

    @staticmethod
    def is_legacy(data: dict[str, Any]) -> bool:
        """Return ``True`` if *data* is a legacy unsigned file."""
        return "env" not in data
