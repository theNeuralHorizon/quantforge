"""Pin the contract that /docs and /redoc are iframeable from /ui/,
while every other endpoint stays DENY.

This is a regression test for the Swagger iframe that shipped empty
because X-Frame-Options + frame-ancestors were blocking it.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from quantforge.api import auth
    auth.reset_keys()
    auth.register_raw_key("iframe_test_key_12345")
    from quantforge.api.app import create_app
    return TestClient(create_app(), headers={"X-API-Key": "iframe_test_key_12345"})


class TestFrameability:
    def test_api_routes_deny_framing(self, client):
        r = client.get("/healthz")
        assert r.headers["X-Frame-Options"] == "DENY"
        assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]

    def test_v1_routes_deny_framing(self, client):
        r = client.post(
            "/v1/options/price",
            json={"S": 100, "K": 100, "T": 1.0, "sigma": 0.2},
        )
        assert r.status_code == 200
        assert r.headers["X-Frame-Options"] == "DENY"
        assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]

    def test_ui_itself_cannot_be_iframed(self, client):
        # Defence-in-depth: even the dashboard HTML is not meant to be iframed.
        r = client.get("/ui/index.html")
        # Status may be 404 if the static file handler doesn't resolve without
        # a trailing slash, but ANY response MUST carry DENY.
        assert r.headers["X-Frame-Options"] == "DENY"

    def test_docs_is_same_origin_iframeable(self, client):
        r = client.get("/docs")
        assert r.status_code == 200
        # Must NOT be DENY — that would kill the iframe on /ui/
        assert r.headers["X-Frame-Options"] == "SAMEORIGIN"
        # CSP must allow same-origin framing
        csp = r.headers["Content-Security-Policy"]
        assert "frame-ancestors 'self'" in csp
        assert "frame-ancestors 'none'" not in csp
        # Swagger's own bundles
        assert "cdn.jsdelivr.net" in csp

    def test_redoc_is_same_origin_iframeable(self, client):
        r = client.get("/redoc")
        assert r.status_code == 200
        assert r.headers["X-Frame-Options"] == "SAMEORIGIN"
        assert "frame-ancestors 'self'" in r.headers["Content-Security-Policy"]

    def test_openapi_json_stays_strict(self, client):
        # The JSON schema is an API response, not a document. Lock it down.
        r = client.get("/openapi.json")
        assert r.status_code == 200
        assert r.headers["X-Frame-Options"] == "DENY"


class TestCSPStillHardForAPI:
    def test_v1_csp_has_no_unsafe_eval(self, client):
        r = client.get("/healthz")
        assert "'unsafe-eval'" not in r.headers["Content-Security-Policy"]
        assert "cdn.jsdelivr.net" not in r.headers["Content-Security-Policy"]

    def test_ui_csp_has_unsafe_eval_and_cdns(self, client):
        r = client.get("/ui/")
        csp = r.headers["Content-Security-Policy"]
        assert "'unsafe-eval'" in csp
        assert "cdn.jsdelivr.net" in csp
        assert "cdn.tailwindcss.com" in csp

    def test_docs_csp_allows_swagger_cdn(self, client):
        r = client.get("/docs")
        csp = r.headers["Content-Security-Policy"]
        # Swagger UI bundle lives on jsdelivr; has to be allowed.
        assert "cdn.jsdelivr.net" in csp
        # Docs does NOT need unsafe-eval; Swagger UI is a React app that doesn't eval.
        assert "'unsafe-eval'" not in csp
