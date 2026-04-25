"""/v1/audit — read-only audit log access (admin-flavoured)."""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Query

from quantforge.api.audit import get_audit_log
from quantforge.api.auth import verify_api_key

router = APIRouter(prefix="/v1/audit", tags=["audit"])


def _is_demo() -> bool:
    return os.environ.get("QUANTFORGE_ALLOW_UNAUTH", "").lower() in ("1", "true", "yes")


@router.get("")
def recent(
    limit: int = Query(100, ge=1, le=1000),
    owner: str = Depends(verify_api_key),
) -> list[dict]:
    """Return recent audit entries.

    Privacy model:
      - Authenticated callers see their *own* audit history (scoped by
        the SHA-256 prefix of their API key).
      - On demo deploys (`QUANTFORGE_ALLOW_UNAUTH=true`), unauthenticated
        callers see a global stream with the `owner` column masked, so
        the UI's "Recent Activity" feed has something to show without
        leaking which key generated which call. Real owner hashes are
        only revealed when the caller authenticates.
      - On hardened deploys, anonymous callers get 401 (unchanged).

    There is no 'list every owner unmasked' endpoint: that would be a
    cross-tenant leak. If you need full audit data, read the SQLite
    file directly from inside the cluster.
    """
    if owner == "anonymous":
        if not _is_demo():
            raise HTTPException(status_code=401, detail="audit requires an API key")
        # Demo mode: return the global stream with owner masked.
        rows = get_audit_log().recent(limit=limit, owner=None)
        for r in rows:
            r["owner"] = "•••"  # never reveal real key hashes to anon callers
        return rows
    return get_audit_log().recent(limit=limit, owner=owner)
