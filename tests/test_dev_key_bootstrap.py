"""Tests for the /v1/meta/dev-key endpoint + auth.is_dev_mode().

Security property under test: the endpoint must NEVER leak a real API key.
It only returns the auto-generated dev key when no QUANTFORGE_API_KEYS are
configured.
"""
from __future__ import annotations

import hashlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def dev_client():
    """Fresh app with NO configured keys → dev mode active."""
    os.environ.pop("QUANTFORGE_API_KEYS", None)
    from quantforge.api import auth
    auth.reset_keys()
    from quantforge.api.app import create_app
    app = create_app()
    return TestClient(app)


@pytest.fixture
def prod_client():
    """App configured with real keys → dev mode must be inactive."""
    from quantforge.api import auth
    auth.reset_keys()
    real = hashlib.sha256(b"real_production_key_12345").hexdigest()
    os.environ["QUANTFORGE_API_KEYS"] = real
    # reload internal allowlist
    auth._ALLOWED_HASHES.clear()
    auth._ALLOWED_HASHES.update(auth._load_allowed_hashes())
    try:
        from quantforge.api.app import create_app
        app = create_app()
        yield TestClient(app)
    finally:
        os.environ.pop("QUANTFORGE_API_KEYS", None)
        auth.reset_keys()


class TestDevModeEndpoint:
    def test_returns_dev_key_when_in_dev_mode(self, dev_client):
        r = dev_client.get("/v1/meta/dev-key")
        assert r.status_code == 200
        body = r.json()
        assert body["dev_mode"] is True
        assert body["dev_key"] is not None
        assert body["dev_key"].startswith("qf_dev_")
        assert len(body["dev_key"]) > 20

    def test_endpoint_is_public_no_auth_required(self, dev_client):
        # Must work WITHOUT X-API-Key — that's the whole point (bootstrap)
        r = dev_client.get("/v1/meta/dev-key")
        assert r.status_code == 200

    def test_returns_null_key_in_prod_mode(self, prod_client):
        r = prod_client.get("/v1/meta/dev-key")
        assert r.status_code == 200
        body = r.json()
        # Security property: must NEVER leak the real keys
        assert body["dev_mode"] is False
        assert body["dev_key"] is None

    def test_real_keys_still_authenticate_in_prod(self, prod_client):
        r = prod_client.post(
            "/v1/options/price",
            headers={"X-API-Key": "real_production_key_12345"},
            json={"S": 100, "K": 100, "T": 1.0, "sigma": 0.2},
        )
        assert r.status_code == 200
        assert r.json()["black_scholes"] > 0

    def test_dev_key_authenticates_in_dev_mode(self, dev_client):
        dev_key = dev_client.get("/v1/meta/dev-key").json()["dev_key"]
        r = dev_client.post(
            "/v1/options/price",
            headers={"X-API-Key": dev_key},
            json={"S": 100, "K": 100, "T": 1.0, "sigma": 0.2},
        )
        assert r.status_code == 200
