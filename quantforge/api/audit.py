"""SQLite-backed audit log middleware.

Records every authenticated API call (timestamp, owner, method, path,
status, latency, request_id). Compliance-grade: append-only, no updates.
Rotates automatically once the DB hits 200k rows.
"""
from __future__ import annotations

import contextlib
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


_DEFAULT_DB_FALLBACK = "data/audit.db"
MAX_ROWS = 200_000


def _default_db_path() -> str:
    """Read env var each time so tests can override per-fixture."""
    return os.environ.get("QUANTFORGE_AUDIT_DB", _DEFAULT_DB_FALLBACK)


class AuditLog:
    def __init__(self, db_path: Optional[str] = None):
        db_path = db_path or _default_db_path()
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init()

    def _conn(self):
        return sqlite3.connect(self.db_path, timeout=5, isolation_level=None)

    def _init(self) -> None:
        with self._lock, contextlib.closing(self._conn()) as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    owner TEXT,
                    method TEXT,
                    path TEXT,
                    status INTEGER,
                    latency_ms REAL,
                    request_id TEXT,
                    client_ip TEXT
                )"""
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit(ts)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_audit_owner ON audit(owner)")

    def append(self, owner: Optional[str], method: str, path: str,
                status: int, latency_ms: float, request_id: str,
                client_ip: Optional[str] = None) -> None:
        with self._lock, contextlib.closing(self._conn()) as c:
            c.execute(
                "INSERT INTO audit (ts, owner, method, path, status, latency_ms, request_id, client_ip) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), owner, method, path, int(status), float(latency_ms), request_id, client_ip),
            )
            # Cheap rotation: delete oldest 10% once we exceed MAX_ROWS
            row = c.execute("SELECT COUNT(*) FROM audit").fetchone()
            if row and row[0] > MAX_ROWS:
                cutoff = c.execute(
                    "SELECT ts FROM audit ORDER BY ts LIMIT 1 OFFSET ?", (MAX_ROWS // 10,)
                ).fetchone()
                if cutoff:
                    c.execute("DELETE FROM audit WHERE ts < ?", (cutoff[0],))

    def recent(self, limit: int = 100, owner: Optional[str] = None) -> list[dict]:
        with self._lock, contextlib.closing(self._conn()) as c:
            if owner:
                rows = c.execute(
                    "SELECT ts, owner, method, path, status, latency_ms, request_id "
                    "FROM audit WHERE owner = ? ORDER BY ts DESC LIMIT ?",
                    (owner, limit),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT ts, owner, method, path, status, latency_ms, request_id "
                    "FROM audit ORDER BY ts DESC LIMIT ?", (limit,),
                ).fetchall()
        return [
            {"ts": r[0], "owner": r[1], "method": r[2], "path": r[3],
             "status": r[4], "latency_ms": r[5], "request_id": r[6]}
            for r in rows
        ]


_log: Optional[AuditLog] = None


def get_audit_log(db_path: Optional[str] = None) -> AuditLog:
    global _log
    if _log is None:
        _log = AuditLog(db_path or _default_db_path())
    return _log


def reset_for_tests() -> None:
    global _log
    _log = None


class AuditMiddleware(BaseHTTPMiddleware):
    # Exact matches (no wildcards) + one prefix for static UI files
    _EXACT_SKIP = {"/", "/healthz", "/readyz", "/metrics", "/docs", "/redoc",
                    "/openapi.json", "/favicon.ico"}
    _PREFIX_SKIP = ("/ui/",)

    def __init__(self, app):
        super().__init__(app)
        self.log = get_audit_log()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        if path in self._EXACT_SKIP or any(path.startswith(p) for p in self._PREFIX_SKIP):
            return await call_next(request)
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000.0
        # owner = API-key id (first 8 of sha256); headers['X-Api-Key'] is raw, but
        # we only log the 8-char id we already compute in auth.
        key = request.headers.get("X-API-Key") or ""
        if key:
            import hashlib
            owner = hashlib.sha256(key.encode()).hexdigest()[:8]
        else:
            owner = "anonymous"
        rid = getattr(request.state, "request_id", "")
        client_ip = request.client.host if request.client else None
        try:
            self.log.append(owner, request.method, request.url.path,
                             response.status_code, latency_ms, rid, client_ip)
        except Exception:
            pass  # never let audit failure break the response
        return response
