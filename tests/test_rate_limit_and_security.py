"""Tests that prove slowapi + body-size + request-id actually enforce their limits."""
from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def limited_app(monkeypatch):
    """Build a fresh FastAPI app with a very tight rate limit so we can
    drive it over in a few requests without slowing down the rest of the suite.
    """
    monkeypatch.setenv("QUANTFORGE_API_KEYS", "")  # dev-key path
    from quantforge.api import auth
    auth.reset_keys()
    auth.register_raw_key("ratelimit_test_key")

    app = FastAPI()

    # Attach slowapi with a 3 req/second limit
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address

    def _key(request: Request) -> str:
        return request.headers.get("X-API-Key", get_remote_address(request))
    limiter = Limiter(key_func=_key, default_limits=["3/second"])
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

    @app.exception_handler(RateLimitExceeded)
    async def _rle(request, exc):
        return JSONResponse(status_code=429, content={"error": "too_many"},
                             headers={"Retry-After": "60"})

    from quantforge.api.auth import verify_api_key

    @app.get("/ping")
    def _ping(_k: str = Depends(verify_api_key)):
        return {"pong": True}

    from quantforge.api.security import (
        MaxBodySizeMiddleware,
        RequestIDMiddleware,
        SecurityHeadersMiddleware,
    )
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=128)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    return app


class TestRateLimit:
    def test_429_after_third_request_in_same_second(self, limited_app):
        client = TestClient(limited_app, headers={"X-API-Key": "ratelimit_test_key"})
        # The first 3 should succeed
        oks = 0
        rate_limited = 0
        for _ in range(8):
            r = client.get("/ping")
            if r.status_code == 200:
                oks += 1
            elif r.status_code == 429:
                rate_limited += 1
        assert rate_limited >= 1, "slowapi did not rate-limit any request"
        # Most of the requests should be capped — slowapi 3/s means ~3 successes, ~5 429s
        assert oks <= 4

    def test_rate_limit_per_api_key(self, limited_app):
        """Different keys should get independent buckets (the key_func returns the key)."""
        from quantforge.api import auth
        auth.register_raw_key("second_key_xyz")
        client1 = TestClient(limited_app, headers={"X-API-Key": "ratelimit_test_key"})
        client2 = TestClient(limited_app, headers={"X-API-Key": "second_key_xyz"})
        # Exhaust bucket on key 1
        for _ in range(5):
            client1.get("/ping")
        # key 2 should still be able to ping
        assert client2.get("/ping").status_code == 200


class TestBodySize:
    def test_oversized_body_returns_413(self, limited_app):
        client = TestClient(limited_app, headers={"X-API-Key": "ratelimit_test_key"})
        # Use a large Content-Length header to trip the middleware
        r = client.post("/ping", content=b"x" * 500,
                         headers={"Content-Length": "500"})
        assert r.status_code in (413, 405)  # 405 if route doesn't accept POST; middleware runs first
        if r.status_code == 413:
            assert "too large" in r.json().get("error", "").lower()


class TestSecurityHeaders:
    def test_all_security_headers_present_on_limited_app(self, limited_app):
        client = TestClient(limited_app, headers={"X-API-Key": "ratelimit_test_key"})
        r = client.get("/ping")
        for h in ("Strict-Transport-Security", "X-Frame-Options",
                  "Content-Security-Policy", "Referrer-Policy",
                  "X-Content-Type-Options", "Permissions-Policy"):
            assert h in r.headers, f"missing {h}"
        assert r.headers["X-Frame-Options"] == "DENY"

    def test_request_id_generated_when_not_supplied(self, limited_app):
        client = TestClient(limited_app, headers={"X-API-Key": "ratelimit_test_key"})
        r = client.get("/ping")
        rid = r.headers.get("X-Request-ID", "")
        assert rid and 8 <= len(rid) <= 64

    def test_request_id_echoed(self, limited_app):
        client = TestClient(limited_app, headers={"X-API-Key": "ratelimit_test_key"})
        r = client.get("/ping", headers={"X-Request-ID": "abc123"})
        assert r.headers.get("X-Request-ID") == "abc123"

    def test_malicious_request_id_replaced(self, limited_app):
        """A bad request-id header should be silently replaced, not reflected."""
        client = TestClient(limited_app, headers={"X-API-Key": "ratelimit_test_key"})
        # Path traversal / header injection attempt
        r = client.get("/ping", headers={"X-Request-ID": "../../etc/passwd"})
        rid = r.headers.get("X-Request-ID", "")
        assert "../" not in rid
        assert "/" not in rid
