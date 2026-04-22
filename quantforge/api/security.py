"""Security middleware: headers, request-id, max-body, CORS."""
from __future__ import annotations

import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


SECURE_HEADERS = {
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' cdn.tailwindcss.com cdn.jsdelivr.net unpkg.com; "
        "style-src 'self' 'unsafe-inline' fonts.googleapis.com cdn.jsdelivr.net; "
        "font-src 'self' fonts.gstatic.com data:; "
        "img-src 'self' data: https:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'"
    ),
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-site",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        for k, v in SECURE_HEADERS.items():
            response.headers.setdefault(k, v)
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        rid = request.headers.get("X-Request-ID", "").strip() or uuid.uuid4().hex[:16]
        # keep request id short + alnum to prevent header smuggling
        if not rid.replace("-", "").isalnum() or len(rid) > 64:
            rid = uuid.uuid4().hex[:16]
        request.state.request_id = rid
        request.state.start_ns = time.perf_counter_ns()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        elapsed_ms = (time.perf_counter_ns() - request.state.start_ns) / 1e6
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Block requests larger than a configured byte limit. Defends against DoS."""

    def __init__(self, app: FastAPI, max_bytes: int = 1_000_000):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        cl = request.headers.get("Content-Length")
        if cl and cl.isdigit() and int(cl) > self.max_bytes:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"error": "request body too large",
                         "detail": f"max {self.max_bytes} bytes"},
            )
        return await call_next(request)


def build_cors_config(extra_origins: list[str] | None = None) -> dict:
    """Strict CORS: allowlist only; no wildcard with credentials."""
    origins = [
        "http://localhost:3000", "http://localhost:5173",
        "http://localhost:8080", "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]
    if extra_origins:
        origins.extend(extra_origins)
    return {
        "allow_origins": origins,
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
        "expose_headers": ["X-Request-ID", "X-Response-Time-Ms", "X-RateLimit-Remaining"],
        "max_age": 600,
    }
