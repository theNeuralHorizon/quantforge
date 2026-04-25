"""/v1/auth — JWT issuance endpoint.

Trades a raw API key for a short-lived JWT. Useful for browser/mobile
clients that want to avoid sending the long-lived API key on every call.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from quantforge.api import jwt_auth as _jwt_auth
from quantforge.api.auth import verify_api_key
from quantforge.api.jwt_auth import issue_token

router = APIRouter(prefix="/v1/auth", tags=["auth"])


class TokenRequest(BaseModel):
    subject: str = Field(..., min_length=1, max_length=128)
    scopes: list[str] | None = Field(default=None, max_length=20)
    ttl_seconds: int = Field(3600, ge=60, le=86400)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int


# Scopes any API-key holder may grant themselves. Privileged scopes
# (currently just "admin") are intentionally absent — promoting a key to
# admin requires a server-side env var so it can't be done by anyone
# holding a regular key.
_SELF_GRANTABLE_SCOPES: frozenset[str] = frozenset({
    "backtest:read", "backtest:write", "options:read",
    "portfolio:read", "risk:read", "ml:read", "ml:write",
})

# All recognised scopes (super-set of self-grantable). Used for input
# validation only — the privilege check happens against the more
# restrictive `_SELF_GRANTABLE_SCOPES` below.
_ALL_KNOWN_SCOPES: frozenset[str] = _SELF_GRANTABLE_SCOPES | frozenset({"admin"})


@router.post("/token", response_model=TokenResponse)
def issue(req: TokenRequest, owner: str = Depends(verify_api_key)) -> TokenResponse:
    # Read the live module-level config so reload_config() is honoured.
    if not _jwt_auth._config.enabled:
        raise HTTPException(status_code=503, detail="JWT not configured on this server")
    scopes = set(req.scopes or [])
    bad = scopes - _ALL_KNOWN_SCOPES
    if bad:
        raise HTTPException(status_code=422, detail=f"unknown scopes: {sorted(bad)}")
    # Privilege escalation guard: an API-key holder cannot mint themselves
    # a token with a privileged scope (e.g. "admin") that isn't on the
    # self-grantable list. Without this check, anyone with any key could
    # issue an admin JWT to themselves and bypass scope-gated routes.
    forbidden = scopes - _SELF_GRANTABLE_SCOPES
    if forbidden:
        raise HTTPException(
            status_code=403,
            detail=f"requested scopes require admin: {sorted(forbidden)}",
        )
    token = issue_token(req.subject, scopes=scopes, ttl_seconds=req.ttl_seconds)
    return TokenResponse(access_token=token, expires_in=req.ttl_seconds)
