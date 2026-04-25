"""Tests for /v1/alerts, /v1/audit, and /v1/backtest/{walk-forward,compare}."""
from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # Use a temporary audit DB so we don't pollute the repo
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    os.environ["QUANTFORGE_AUDIT_DB"] = tmp.name

    from quantforge.api import audit as audit_mod
    from quantforge.api import auth
    from quantforge.api.routes import alerts_routes as alerts_mod
    auth.reset_keys()
    auth.register_raw_key("alerts_audit_key_12345")
    audit_mod.reset_for_tests()
    alerts_mod.reset_for_tests()

    from quantforge.api.app import create_app
    app = create_app()
    return TestClient(app, headers={"X-API-Key": "alerts_audit_key_12345"})


class TestAlertsAPI:
    def test_no_rules_initially(self, client):
        r = client.get("/v1/alerts/rules")
        assert r.status_code == 200
        assert r.json() == []

    def test_create_rule(self, client):
        r = client.post("/v1/alerts/rules", json={
            "name": "high_drawdown", "metric": "drawdown",
            "threshold": -0.10, "direction": "below", "severity": "critical",
        })
        assert r.status_code == 201
        body = r.json()
        assert body["name"] == "high_drawdown"

    def test_duplicate_rule_rejected(self, client):
        r = client.post("/v1/alerts/rules", json={
            "name": "high_drawdown", "metric": "drawdown",
            "threshold": -0.20, "direction": "below", "severity": "warning",
        })
        assert r.status_code == 409

    def test_invalid_metric_rejected(self, client):
        r = client.post("/v1/alerts/rules", json={
            "name": "bad_rule", "metric": "not_a_metric",
            "threshold": 0, "direction": "above", "severity": "info",
        })
        assert r.status_code == 422

    def test_evaluate_fires(self, client):
        # drawdown -0.15 triggers the 'high_drawdown' rule (<= -0.10)
        r = client.post("/v1/alerts/evaluate", json={"context": {"drawdown": -0.15}})
        assert r.status_code == 200
        body = r.json()
        assert len(body["fired"]) >= 1

    def test_evaluate_no_fire(self, client):
        # Dedupe will still suppress the previous fire, and the new value -0.05 is above threshold
        r = client.post("/v1/alerts/evaluate", json={"context": {"drawdown": -0.05}})
        assert r.status_code == 200
        assert len(r.json()["fired"]) == 0

    def test_recent_events(self, client):
        r = client.get("/v1/alerts/events?limit=10")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_delete_rule(self, client):
        r = client.delete("/v1/alerts/rules/high_drawdown")
        assert r.status_code == 200
        # Verify removed
        rules = client.get("/v1/alerts/rules").json()
        assert not any(x["name"] == "high_drawdown" for x in rules)


class TestAuditAPI:
    def test_requires_api_key(self, client):
        # Our client has a key; exercise that at least we get 200 (not 401)
        r = client.get("/v1/audit")
        assert r.status_code == 200

    def test_returns_list(self, client):
        # Generate some traffic
        client.get("/v1/alerts/rules")
        client.get("/v1/alerts/events?limit=5")
        r = client.get("/v1/audit")
        assert r.status_code == 200
        rows = r.json()
        assert isinstance(rows, list)
        assert len(rows) >= 1
        # Each row has expected fields
        assert set(rows[0].keys()) >= {"ts", "owner", "method", "path", "status", "latency_ms", "request_id"}

    def test_anonymous_rejected_in_hardened_mode(self, client, monkeypatch):
        """Without ALLOW_UNAUTH=true, anonymous callers must 401 — this
        is the privacy contract the audit endpoint enforces. Strip the
        client's auth header so it looks anonymous."""
        monkeypatch.delenv("QUANTFORGE_ALLOW_UNAUTH", raising=False)
        from fastapi.testclient import TestClient
        bare = TestClient(client.app)
        r = bare.get("/v1/audit")
        # auth.verify_api_key returns "anonymous" only when ALLOW_UNAUTH
        # is truthy. With it unset the dependency itself 401s before our
        # extra check runs.
        assert r.status_code == 401

    def test_anonymous_in_demo_mode_returns_masked_stream(self, client, monkeypatch):
        """With ALLOW_UNAUTH=true the anon caller should see the recent
        audit stream so the demo UI has something to render — but every
        owner field must be masked so we don't leak real key hashes."""
        monkeypatch.setenv("QUANTFORGE_ALLOW_UNAUTH", "true")
        # Generate traffic via the authenticated client first.
        client.get("/v1/alerts/rules")
        client.get("/v1/alerts/events?limit=5")
        from fastapi.testclient import TestClient
        bare = TestClient(client.app)
        r = bare.get("/v1/audit?limit=5")
        assert r.status_code == 200
        rows = r.json()
        assert isinstance(rows, list)
        assert len(rows) >= 1
        # Every row's owner is the redaction mark, never a hash.
        assert all(row["owner"] == "•••" for row in rows), rows


class TestCompareEndpoint:
    def test_compare_requires_min_two(self, client):
        r = client.post("/v1/backtest/compare", json={"strategies": [
            {"strategy": "buy_and_hold", "tickers": ["SPY"],
             "start": "2023-01-01", "end": "2024-01-01",
             "capital": 100000, "sizing_fraction": 1.0, "rebalance": "bar"}
        ]})
        assert r.status_code == 422  # min_length=2
