"""/v1/audit — read-only audit log access (admin-flavoured)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from quantforge.api.audit import get_audit_log
from quantforge.api.auth import verify_api_key

router = APIRouter(prefix="/v1/audit", tags=["audit"])


@router.get("")
def recent(
    limit: int = Query(100, ge=1, le=1000),
    owner: str = Depends(verify_api_key),
) -> list[dict]:
    """Return recent audit entries for the calling API key's owner only.

    There is no 'list-everyone' endpoint: privacy-first. If you need
    aggregate audit data, read the SQLite file directly from inside the
    cluster.
    """
    if owner == "anonymous":
        raise HTTPException(status_code=401, detail="audit requires an API key")
    return get_audit_log().recent(limit=limit, owner=owner)
