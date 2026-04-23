"""Tests for async job queue + /v1/jobs endpoints."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from quantforge.api import auth
    from quantforge.api import jobs as jobs_module
    auth.reset_keys()
    auth.register_raw_key("test_job_key_1234567890")
    jobs_module.reset_manager()
    from quantforge.api.app import create_app
    app = create_app()
    return TestClient(app, headers={"X-API-Key": "test_job_key_1234567890"})


class TestJobLifecycle:
    def test_submit_backtest_returns_job_id(self, client):
        r = client.post("/v1/jobs/backtest", json={
            "strategy": "buy_and_hold",
            "params": {},
            "tickers": ["SPY"],
            "start": "2023-01-01", "end": "2024-01-01",
            "capital": 100_000, "sizing_fraction": 1.0,
            "rebalance": "bar",
        })
        assert r.status_code == 202
        body = r.json()
        assert "job_id" in body
        assert body["status"] in ("queued", "running")

    def test_poll_and_complete(self, client):
        submit = client.post("/v1/jobs/backtest", json={
            "strategy": "buy_and_hold",
            "tickers": ["SPY"], "start": "2023-01-01", "end": "2024-01-01",
            "capital": 100_000, "sizing_fraction": 1.0, "rebalance": "bar",
        })
        job_id = submit.json()["job_id"]

        deadline = time.time() + 60
        final = None
        while time.time() < deadline:
            r = client.get(f"/v1/jobs/{job_id}")
            assert r.status_code == 200
            body = r.json()
            if body["status"] in ("completed", "failed", "cancelled"):
                final = body
                break
            time.sleep(0.2)

        assert final is not None, "job didn't complete in 60s"
        # Either completed OR failed (e.g. yfinance blocked in CI); both are valid
        # lifecycle terminators; we just check the status transitioned.
        assert final["status"] in ("completed", "failed")
        if final["status"] == "completed":
            assert "result" in final
            assert final["result"]["strategy"] == "buy_and_hold"

    def test_get_unknown_job_404(self, client):
        r = client.get("/v1/jobs/nonexistent_xxxxxxxxx")
        assert r.status_code == 404

    def test_cancel_unknown_job_404(self, client):
        r = client.delete("/v1/jobs/ghost")
        assert r.status_code == 404

    def test_list_jobs(self, client):
        r = client.get("/v1/jobs?limit=10")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


class TestIsolation:
    def test_foreign_key_cannot_access_job(self, client):
        # Create job with key 1
        from quantforge.api import auth
        auth.register_raw_key("other_owner_key_xxxxxx")
        sub = client.post("/v1/jobs/backtest", json={
            "strategy": "buy_and_hold",
            "tickers": ["SPY"], "start": "2023-01-01", "end": "2024-01-01",
            "capital": 100_000, "sizing_fraction": 1.0, "rebalance": "bar",
        })
        job_id = sub.json()["job_id"]
        # Try to access with a different key
        r = TestClient(client.app, headers={"X-API-Key": "other_owner_key_xxxxxx"}).get(
            f"/v1/jobs/{job_id}"
        )
        assert r.status_code in (403, 404)
