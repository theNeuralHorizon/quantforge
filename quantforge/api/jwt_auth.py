"""JWT authentication — alternative to API keys. HS256 only; asymmetric optional."""
from __future__ import annotations

import os
import time

import jwt  # PyJWT
from fastapi import Header, HTTPException, status

_ENV_JWT_SECRET = "QUANTFORGE_JWT_SECRET"
_ENV_JWT_ISSUER = "QUANTFORGE_JWT_ISSUER"
_ENV_JWT_AUDIENCE = "QUANTFORGE_JWT_AUDIENCE"
_ENV_JWT_ALG = "QUANTFORGE_JWT_ALG"


class JWTConfig:
    def __init__(self):
        self.secret = os.environ.get(_ENV_JWT_SECRET, "").strip()
        self.issuer = os.environ.get(_ENV_JWT_ISSUER, "quantforge").strip()
        self.audience = os.environ.get(_ENV_JWT_AUDIENCE, "quantforge-api").strip()
        self.alg = os.environ.get(_ENV_JWT_ALG, "HS256").strip().upper()

    @property
    def enabled(self) -> bool:
        return bool(self.secret)


_config = JWTConfig()


def reload_config() -> None:
    global _config
    _config = JWTConfig()


def issue_token(
    subject: str,
    scopes: set[str] | None = None,
    ttl_seconds: int = 3600,
) -> str:
    """Issue a JWT for `subject` (typically a user id or service name).

    scopes: optional set of permissions; stored in the 'scopes' claim.
    """
    if not _config.enabled:
        raise RuntimeError("JWT not configured; set QUANTFORGE_JWT_SECRET")
    now = int(time.time())
    payload = {
        "iss": _config.issuer,
        "aud": _config.audience,
        "sub": subject,
        "iat": now,
        "exp": now + ttl_seconds,
        "scopes": sorted(scopes) if scopes else [],
    }
    return jwt.encode(payload, _config.secret, algorithm=_config.alg)


def decode_token(token: str) -> dict:
    if not _config.enabled:
        raise HTTPException(status_code=500, detail="JWT not configured")
    try:
        return jwt.decode(
            token, _config.secret, algorithms=[_config.alg],
            audience=_config.audience, issuer=_config.issuer,
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="token expired") from None
    except jwt.InvalidAudienceError:
        raise HTTPException(status_code=401, detail="invalid audience") from None
    except jwt.InvalidIssuerError:
        raise HTTPException(status_code=401, detail="invalid issuer") from None
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"invalid token: {e}") from e


def verify_bearer(authorization: str | None = Header(None)) -> dict:
    """FastAPI dependency: returns the decoded claims or raises 401."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split(None, 1)[1].strip()
    return decode_token(token)


_AUTH_HEADER = Header(None, alias="Authorization")


def require_scopes(*required: str):
    """Dependency factory: ensures the JWT carries all required scopes."""
    req = set(required)
    def dep(claims: dict = _AUTH_HEADER) -> dict:
        # Actually decode via verify_bearer; this wrapper composes cleanly
        claims = verify_bearer(claims if isinstance(claims, str) else None)
        have = set(claims.get("scopes", []))
        missing = req - have
        if missing:
            raise HTTPException(
                status_code=403,
                detail=f"missing scopes: {sorted(missing)}",
            )
        return claims
    return dep
