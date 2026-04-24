"""API key authentication — header-based, constant-time comparison, hashed storage."""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets

from fastapi import Header, HTTPException, status

# Keys loaded from env as comma-separated SHA256 hashes of raw keys.
# For dev, we auto-generate one key at startup if none are configured.
_ENV_KEYS = "QUANTFORGE_API_KEYS"  # comma-separated SHA256 hex digests
_ENV_ALLOW_UNAUTH = "QUANTFORGE_ALLOW_UNAUTH"


def _sha256(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_allowed_hashes() -> set[str]:
    raw = os.environ.get(_ENV_KEYS, "").strip()
    if not raw:
        return set()
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


_ALLOWED_HASHES = _load_allowed_hashes()
_DEV_KEY: str | None = None


def get_or_create_dev_key() -> str:
    """For local dev only — autogen a key and register its hash in-process.

    In prod, set QUANTFORGE_API_KEYS to a comma-separated list of SHA256 hex digests.
    """
    global _DEV_KEY
    if _DEV_KEY is None and not _ALLOWED_HASHES:
        _DEV_KEY = "qf_dev_" + secrets.token_urlsafe(24)
        _ALLOWED_HASHES.add(_sha256(_DEV_KEY))
    return _DEV_KEY or ""


def _unauth_allowed() -> bool:
    return os.environ.get(_ENV_ALLOW_UNAUTH, "").lower() in ("1", "true", "yes")


def verify_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:
    """FastAPI dependency: validates the API key header in constant time.

    If QUANTFORGE_ALLOW_UNAUTH=1, skips auth (for dashboards behind other gateways).
    """
    if _unauth_allowed():
        return "anonymous"

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if len(x_api_key) > 128 or len(x_api_key) < 8:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid key")

    key_hash = _sha256(x_api_key)
    for allowed in _ALLOWED_HASHES:
        if hmac.compare_digest(key_hash, allowed):
            # Return a stable, safe key id (first 8 hex of hash) for logging
            return key_hash[:8]

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid key")


def register_raw_key(raw_key: str) -> None:
    """Add a raw key (SHA-256 hashed internally). Useful for tests."""
    _ALLOWED_HASHES.add(_sha256(raw_key))


def reset_keys() -> None:
    """Reset for tests."""
    global _DEV_KEY
    _ALLOWED_HASHES.clear()
    _DEV_KEY = None


def is_dev_mode() -> bool:
    """True when running with an auto-generated dev key (no QUANTFORGE_API_KEYS).

    Used by /v1/meta/dev-key to decide whether to expose the auto-key to the UI.
    """
    return bool(_DEV_KEY) and not os.environ.get(_ENV_KEYS, "").strip()


def get_dev_key_if_dev_mode() -> str | None:
    """Return the dev key ONLY if we're in dev mode (never leak real keys)."""
    return _DEV_KEY if is_dev_mode() else None
