"""/v1/auth — JWT issuance endpoint.

Trades a raw API key for a short-lived JWT. Useful for browser/mobile
clients that want to avoid sending the long-lived API key on every call.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from quantforge.api.auth import verify_api_key
from quantforge.api.jwt_auth import issue_token, _config


router = APIRouter(prefix="/v1/auth", tags=["auth"])


class TokenRequest(BaseModel):
    subject: str = Field(..., min_length=1, max_length=128)
    scopes: Optional[List[str]] = Field(default=None, max_length=20)
    ttl_seconds: int = Field(3600, ge=60, le=86400)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int


@router.post("/token", response_model=TokenResponse)
def issue(req: TokenRequest, _owner: str = Depends(verify_api_key)) -> TokenResponse:
    if not _config.enabled:
        raise HTTPException(status_code=503, detail="JWT not configured on this server")
    # Whitelist scopes
    allowed = {"backtest:read", "backtest:write", "options:read",
               "portfolio:read", "risk:read", "ml:read", "ml:write", "admin"}
    scopes = set(req.scopes or [])
    bad = scopes - allowed
    if bad:
        raise HTTPException(status_code=422, detail=f"unknown scopes: {sorted(bad)}")
    token = issue_token(req.subject, scopes=scopes, ttl_seconds=req.ttl_seconds)
    return TokenResponse(access_token=token, expires_in=req.ttl_seconds)
