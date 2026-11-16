"""Tests for the security layer: sign/verify, tamper detection, legacy files."""

from __future__ import annotations

import base64
import json
import os
import stat
import tempfile
from pathlib import Path

import pytest

from memory_manager.security import SecurityManager, TamperDetectedError, get_security_manager
from memory_manager.storage import atomic_json_write, read_json


# ---------------------------------------------------------------------------
# SecurityManager — sign / verify
# ---------------------------------------------------------------------------


def _make_manager(tmp_path: Path) -> SecurityManager:
    """Create a manager backed by a temp key file, isolated from the real keyring."""
    key_dir = tmp_path / "keys"
    key_dir.mkdir()
    key = os.urandom(32)
    key_path = key_dir / "default.key"
    key_path.write_bytes(key)
    # Monkeypatch: bypass keyring and file discovery by injecting key directly.
    mgr = SecurityManager.__new__(SecurityManager)
    mgr.key_id = "default"
    mgr.encrypt = False
    mgr._key = key
    mgr._fernet = None
    return mgr


class TestSignVerify:
    def test_round_trip(self, tmp_path):
        mgr = _make_manager(tmp_path)
        payload = {"foo": "bar", "nested": {"x": 1}}
        envelope = mgr.sign(payload)
        result = mgr.verify(envelope)
        assert result == payload

    def test_envelope_has_required_fields(self, tmp_path):
        mgr = _make_manager(tmp_path)
        envelope = mgr.sign({"a": 1})
        assert envelope["env"] == 1
        assert envelope["alg"] == "hmac-sha256"
        assert "payload" in envelope
        assert "sig" in envelope
        assert "signed_at" in envelope

    def test_tampered_payload_raises(self, tmp_path):
        mgr = _make_manager(tmp_path)
        envelope = mgr.sign({"secret": "value"})
        # Decode, modify, re-encode payload without updating sig.
        raw = json.loads(base64.b64decode(envelope["payload"]))
        raw["secret"] = "INJECTED"
        envelope["payload"] = base64.b64encode(json.dumps(raw).encode()).decode()
        with pytest.raises(TamperDetectedError):
            mgr.verify(envelope)

    def test_tampered_signature_raises(self, tmp_path):
        mgr = _make_manager(tmp_path)
        envelope = mgr.sign({"data": "ok"})
        envelope["sig"] = "0" * len(envelope["sig"])
        with pytest.raises(TamperDetectedError):
            mgr.verify(envelope)

    def test_swapped_envelope_raises(self, tmp_path):
        """Envelope signed by a different key should fail."""
        mgr1 = _make_manager(tmp_path)
        key2 = os.urandom(32)
        mgr2 = SecurityManager.__new__(SecurityManager)
        mgr2.key_id = "other"
        mgr2.encrypt = False
        mgr2._key = key2
        mgr2._fernet = None

        envelope = mgr1.sign({"msg": "hello"})
        with pytest.raises(TamperDetectedError):
            mgr2.verify(envelope)

    def test_legacy_unsigned_file_passes_with_warning(self, tmp_path):
        mgr = _make_manager(tmp_path)
        legacy = {"schema_version": 1, "entries": []}
        result = mgr.verify(legacy)
        assert result == legacy

    def test_is_legacy_detection(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert SecurityManager.is_legacy({"plain": "dict"})
        assert not SecurityManager.is_legacy(mgr.sign({"plain": "dict"}))


# ---------------------------------------------------------------------------
# storage helpers — permissions + security integration
# ---------------------------------------------------------------------------


class TestAtomicJsonWrite:
    def test_file_created_with_mode_600(self, tmp_path):
        path = tmp_path / "test.json"
        atomic_json_write(path, {"hello": "world"})
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"

    def test_signed_file_is_envelope(self, tmp_path):
        mgr = _make_manager(tmp_path)
        path = tmp_path / "signed.json"
        atomic_json_write(path, {"value": 42}, security=mgr)
        raw = json.loads(path.read_text())
        assert raw.get("env") == 1, "File should be wrapped in a signed envelope"

    def test_read_json_verifies_signature(self, tmp_path):
        mgr = _make_manager(tmp_path)
        path = tmp_path / "data.json"
        original = {"key": "value", "num": 99}
        atomic_json_write(path, original, security=mgr)
        loaded = read_json(path, security=mgr)
        assert loaded == original

    def test_read_json_tamper_detected(self, tmp_path):
        mgr = _make_manager(tmp_path)
        path = tmp_path / "data.json"
        atomic_json_write(path, {"safe": "content"}, security=mgr)
        # Directly corrupt the file on disk.
        envelope = json.loads(path.read_text())
        envelope["sig"] = "deadbeef" * 8
        path.write_text(json.dumps(envelope))
        with pytest.raises(TamperDetectedError):
            read_json(path, security=mgr)

    def test_read_json_legacy_no_security(self, tmp_path):
        path = tmp_path / "legacy.json"
        data = {"schema_version": 1, "entries": []}
        path.write_text(json.dumps(data))
        result = read_json(path)
        assert result == data

    def test_no_tmp_file_left_behind(self, tmp_path):
        path = tmp_path / "clean.json"
        atomic_json_write(path, {"x": 1})
        leftover = list(tmp_path.glob(".*.tmp"))
        assert not leftover, f"Temp files left behind: {leftover}"


# ---------------------------------------------------------------------------
# get_security_manager — caching
# ---------------------------------------------------------------------------


class TestGetSecurityManager:
    def test_same_key_id_returns_same_instance(self):
        mgr1 = get_security_manager(key_id="test_cache_key")
        mgr2 = get_security_manager(key_id="test_cache_key")
        assert mgr1 is mgr2

    def test_different_encrypt_flag_returns_different_instance(self):
        # encrypt=False vs encrypt=True should yield different cached entries.
        mgr_plain = get_security_manager(key_id="test_enc_key", encrypt=False)
        # encrypt=True would require the cryptography package; skip if absent.
        pytest.importorskip("cryptography")
        mgr_enc = get_security_manager(key_id="test_enc_key", encrypt=True)
        assert mgr_plain is not mgr_enc
