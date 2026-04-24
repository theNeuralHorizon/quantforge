"""API test suite — fastapi TestClient, no network required (uses synthetic data)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # Register a known key BEFORE importing the app so startup auth works.
    from quantforge.api import auth
    auth.reset_keys()
    auth.register_raw_key("test_key_1234567890")
    from quantforge.api.app import create_app
    app = create_app()
    return TestClient(app, headers={"X-API-Key": "test_key_1234567890"})


class TestHealth:
    def test_healthz(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_readyz(self, client):
        r = client.get("/readyz")
        assert r.status_code == 200

    def test_metrics(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "QuantForge" in r.text


class TestSecurity:
    def test_missing_key_rejected(self, client):
        r = client.post("/v1/options/price",
                         headers={"X-API-Key": ""},
                         json={"S": 100, "K": 100, "T": 1.0, "sigma": 0.2})
        assert r.status_code == 401

    def test_wrong_key_rejected(self, client):
        r = client.post("/v1/options/price",
                         headers={"X-API-Key": "wrong_key_xxxxxxxxxxxx"},
                         json={"S": 100, "K": 100, "T": 1.0, "sigma": 0.2})
        assert r.status_code == 401

    def test_security_headers_present(self, client):
        r = client.get("/healthz")
        for h in ("Strict-Transport-Security", "X-Content-Type-Options",
                  "X-Frame-Options", "Referrer-Policy", "Content-Security-Policy"):
            assert h in r.headers, f"missing security header: {h}"

    def test_request_id_echo(self, client):
        r = client.get("/healthz", headers={"X-Request-ID": "abc123"})
        assert r.headers.get("X-Request-ID") == "abc123"

    def test_request_id_auto_assigned(self, client):
        r = client.get("/healthz")
        rid = r.headers.get("X-Request-ID")
        assert rid and len(rid) >= 8

    def test_oversized_body_rejected(self, client):
        big = {"returns": [0.0] * 100_000, "confidence": 0.95, "method": "historical"}
        # this is rejected via Content-Length header
        r = client.post("/v1/risk/var", json=big,
                         headers={"Content-Length": str(10_000_000)})
        # Either body too big (413) or schema rejects first
        assert r.status_code in (413, 422)


class TestOptions:
    def test_price_call(self, client):
        r = client.post("/v1/options/price",
                         json={"S": 100, "K": 100, "T": 1.0, "r": 0.04, "sigma": 0.2})
        assert r.status_code == 200
        body = r.json()
        assert body["black_scholes"] > 0
        assert "delta" in body["greeks"]

    def test_price_put_call_parity(self, client):
        call_r = client.post("/v1/options/price",
                              json={"S": 100, "K": 100, "T": 1.0, "r": 0.05,
                                    "sigma": 0.25, "option": "call"}).json()
        put_r = client.post("/v1/options/price",
                             json={"S": 100, "K": 100, "T": 1.0, "r": 0.05,
                                   "sigma": 0.25, "option": "put"}).json()
        # c - p = S - K*exp(-rT)
        import math
        parity = call_r["black_scholes"] - put_r["black_scholes"] - (100 - 100 * math.exp(-0.05))
        assert abs(parity) < 1e-6

    def test_iv_round_trip(self, client):
        price_r = client.post("/v1/options/price",
                               json={"S": 100, "K": 100, "T": 1.0, "r": 0.04,
                                     "sigma": 0.22}).json()
        iv_r = client.post("/v1/options/iv",
                            json={"price": price_r["black_scholes"],
                                  "S": 100, "K": 100, "T": 1.0, "r": 0.04}).json()
        assert abs(iv_r["implied_vol"] - 0.22) < 1e-5

    def test_iv_no_solution(self, client):
        # Price below intrinsic
        r = client.post("/v1/options/iv",
                         json={"price": 0.01, "S": 200, "K": 100, "T": 1.0, "r": 0.04})
        assert r.status_code in (422, 400)

    def test_invalid_inputs_rejected(self, client):
        # Negative spot
        r = client.post("/v1/options/price",
                         json={"S": -100, "K": 100, "T": 1.0, "sigma": 0.2})
        assert r.status_code == 422
        # Zero time
        r = client.post("/v1/options/price",
                         json={"S": 100, "K": 100, "T": 0, "sigma": 0.2})
        assert r.status_code == 422


class TestRisk:
    def test_var_historical(self, client):
        import numpy as np
        rng = np.random.default_rng(42)
        rets = rng.normal(0, 0.01, 1000).tolist()
        r = client.post("/v1/risk/var",
                         json={"returns": rets, "confidence": 0.95, "method": "historical"})
        assert r.status_code == 200
        body = r.json()
        assert body["var"] > 0  # returned as positive loss
        assert body["cvar"] >= body["var"]

    def test_var_methods_all_work(self, client):
        import numpy as np
        rng = np.random.default_rng(1)
        rets = rng.normal(0.0005, 0.02, 500).tolist()
        for method in ("historical", "parametric", "cornish_fisher", "monte_carlo"):
            r = client.post("/v1/risk/var",
                             json={"returns": rets, "confidence": 0.95, "method": method})
            assert r.status_code == 200, f"method={method} failed"


class TestMeta:
    def test_openapi_schema(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        paths = schema["paths"]
        assert "/v1/options/price" in paths
        assert "/v1/backtest" in paths
        assert "/v1/portfolio/optimize" in paths
        assert "/v1/risk/var" in paths
        assert "/v1/ml/train" in paths
        assert "/healthz" in paths
