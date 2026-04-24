"""Tests for quantforge_client — run a real uvicorn server in a thread."""
from __future__ import annotations

import socket
import threading
import time

import pytest
import uvicorn


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server_url():
    from quantforge.api import audit as audit_mod
    from quantforge.api import auth
    from quantforge.api.routes import alerts_routes
    auth.reset_keys()
    auth.register_raw_key("sdk_test_key_12345")
    audit_mod.reset_for_tests()
    alerts_routes.reset_for_tests()

    from quantforge.api.app import create_app
    app = create_app()

    port = _free_port()
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                          access_log=False)
    server = uvicorn.Server(cfg)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for startup
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("uvicorn didn't start in time")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def sdk(server_url):
    from quantforge_client import QuantForge
    c = QuantForge(server_url, api_key="sdk_test_key_12345", timeout=30)
    yield c
    c.close()


class TestSDKBasics:
    def test_health(self, sdk):
        h = sdk.health()
        assert h["status"] == "ok"
        assert "version" in h

    def test_readiness(self, sdk):
        r = sdk.readiness()
        assert r["status"] in ("ready", "not_ready")

    def test_price_option(self, sdk):
        r = sdk.price_option(S=100, K=100, T=1.0, sigma=0.2)
        assert r["black_scholes"] > 0
        assert "delta" in r["greeks"]

    def test_implied_vol_round_trip(self, sdk):
        p = sdk.price_option(S=100, K=100, T=1.0, sigma=0.25)
        iv = sdk.implied_vol(price=p["black_scholes"], S=100, K=100, T=1.0, r=0.04)
        assert abs(iv["implied_vol"] - 0.25) < 1e-4


class TestSDKErrors:
    def test_invalid_raises(self, sdk):
        from quantforge_client import QuantForgeError
        with pytest.raises(QuantForgeError) as exc:
            sdk.price_option(S=-1, K=100, T=1, sigma=0.2)
        assert exc.value.status == 422


class TestSDKAlerts:
    def test_create_list_delete(self, sdk):
        sdk.create_alert_rule("sdk_rule_1", "drawdown", -0.1, "below", "critical")
        rules = sdk.list_alert_rules()
        assert any(r["name"] == "sdk_rule_1" for r in rules)
        ev = sdk.evaluate_alerts({"drawdown": -0.2})
        assert ev["n_rules"] >= 1
        assert len(ev["fired"]) >= 1
        sdk.delete_alert_rule("sdk_rule_1")
        rules = sdk.list_alert_rules()
        assert not any(r["name"] == "sdk_rule_1" for r in rules)


class TestSDKJobs:
    def test_submit_and_wait(self, sdk):
        job = sdk.submit_backtest(
            strategy="buy_and_hold", tickers=["SPY"],
            start="2023-01-01", end="2024-01-01",
            capital=100_000, sizing_fraction=1.0, rebalance="bar",
        )
        assert "job_id" in job
        final = sdk.wait(job["job_id"], poll_interval=0.3, timeout=90)
        assert final["status"] in ("completed", "failed")


class TestSDKMarketAndAudit:
    def test_audit_after_calls(self, sdk):
        # Generate some traffic
        sdk.health()
        sdk.price_option(S=100, K=100, T=1, sigma=0.2)
        rows = sdk.audit(limit=20)
        # health is skipped, but /v1/options/price isn't
        paths = {r["path"] for r in rows}
        assert any(p.startswith("/v1/options") for p in paths)
