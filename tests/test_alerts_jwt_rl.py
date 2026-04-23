"""Tests for JWT auth, alerting engine, and PPO RL strategy."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest


# ---------- JWT ---------------------------------------------------------------
class TestJWT:
    def _setup(self):
        os.environ["QUANTFORGE_JWT_SECRET"] = "test_secret_xxx"
        os.environ["QUANTFORGE_JWT_ISSUER"] = "quantforge-test"
        os.environ["QUANTFORGE_JWT_AUDIENCE"] = "quantforge-test-aud"
        from quantforge.api import jwt_auth
        jwt_auth.reload_config()
        return jwt_auth

    def test_issue_and_decode_round_trip(self):
        jwt_auth = self._setup()
        tok = jwt_auth.issue_token("user-1", scopes={"options:read", "backtest:read"})
        claims = jwt_auth.decode_token(tok)
        assert claims["sub"] == "user-1"
        assert "options:read" in claims["scopes"]

    def test_expired_token_rejected(self):
        jwt_auth = self._setup()
        tok = jwt_auth.issue_token("user-1", ttl_seconds=1)
        time.sleep(1.2)
        with pytest.raises(Exception):
            jwt_auth.decode_token(tok)

    def test_tampered_token_rejected(self):
        jwt_auth = self._setup()
        tok = jwt_auth.issue_token("user-1")
        bad = tok[:-2] + ("A" if tok[-1] != "A" else "B") + "A"
        with pytest.raises(Exception):
            jwt_auth.decode_token(bad)

    def test_wrong_issuer_rejected(self):
        jwt_auth = self._setup()
        tok = jwt_auth.issue_token("user-1")
        # Change config to a different issuer
        os.environ["QUANTFORGE_JWT_ISSUER"] = "other-issuer"
        jwt_auth.reload_config()
        with pytest.raises(Exception):
            jwt_auth.decode_token(tok)
        # restore
        os.environ["QUANTFORGE_JWT_ISSUER"] = "quantforge-test"
        jwt_auth.reload_config()

    def test_jwt_disabled_when_no_secret(self):
        for k in ("QUANTFORGE_JWT_SECRET",):
            os.environ.pop(k, None)
        from quantforge.api import jwt_auth
        jwt_auth.reload_config()
        with pytest.raises(RuntimeError):
            jwt_auth.issue_token("u")


# ---------- Alerts -----------------------------------------------------------
class TestAlerts:
    def test_threshold_rule_fires(self):
        from quantforge.alerts.rules import ThresholdRule, Severity
        rule = ThresholdRule(
            name="drawdown", metric=lambda ctx: ctx["drawdown"],
            threshold=-0.10, direction="below", severity=Severity.CRITICAL,
        )
        event = rule.evaluate({"drawdown": -0.15})
        assert event is not None
        assert event.severity == Severity.CRITICAL

    def test_threshold_rule_no_fire(self):
        from quantforge.alerts.rules import ThresholdRule
        rule = ThresholdRule(
            name="vol", metric=lambda ctx: ctx["vol"],
            threshold=0.5, direction="above",
        )
        assert rule.evaluate({"vol": 0.2}) is None

    def test_engine_dedupes(self):
        from quantforge.alerts import AlertEngine, ThresholdRule
        eng = AlertEngine(dedupe_seconds=60.0)
        eng.add_rule(ThresholdRule(
            name="high_var", metric=lambda ctx: ctx["var"],
            threshold=0.02, direction="above",
        ))
        # First evaluation fires; second (same ctx) is deduped
        fired1 = eng.evaluate({"var": 0.03})
        fired2 = eng.evaluate({"var": 0.04})
        assert len(fired1) == 1
        assert len(fired2) == 0

    def test_engine_fans_out_to_channels(self):
        from quantforge.alerts import AlertEngine, ThresholdRule
        from quantforge.alerts.channels import Channel
        received = []
        class Capture(Channel):
            def send(self, event):
                received.append(event)
                return True
        eng = AlertEngine(channels=[Capture()])
        eng.add_rule(ThresholdRule(
            name="x", metric=lambda c: c["x"], threshold=1, direction="above",
        ))
        eng.evaluate({"x": 2})
        assert len(received) == 1

    def test_null_channel(self):
        from quantforge.alerts.channels import NullChannel
        from quantforge.alerts.rules import AlertEvent, Severity
        assert NullChannel().send(AlertEvent(
            rule_name="t", severity=Severity.INFO, title="t", detail="t",
        )) is True


# ---------- PPO --------------------------------------------------------------
class TestPPO:
    def _data(self):
        from quantforge.data.synthetic import generate_ohlcv
        return {"SPY": generate_ohlcv(n=800, s0=100, mu=0.1, sigma=0.2, seed=42)}

    def test_warmup(self):
        from quantforge.strategies import PPOStrategy
        s = PPOStrategy(train_window=300, window=10)
        assert s.warmup() > 300

    def test_returns_valid_signal(self):
        from quantforge.backtest import BacktestEngine
        from quantforge.strategies import PPOStrategy
        data = self._data()
        s = PPOStrategy(train_window=300, window=10, retrain_every=63, n_epochs=2)
        eng = BacktestEngine(strategy=s, data=data, initial_capital=100_000,
                              sizing_fraction=0.5, rebalance="weekly")
        res = eng.run()
        assert len(res.equity_curve) > 0
        assert np.isfinite(res.equity_curve.iloc[-1])

    def test_no_short_when_disabled(self):
        from quantforge.strategies.rl_ppo import PPOStrategy
        s = PPOStrategy(train_window=300, window=10, allow_short=False)
        # Smoke: policy always produces direction >= 0 after forcing short off
        import pandas as pd
        from quantforge.data.synthetic import generate_ohlcv
        df = generate_ohlcv(n=400, seed=7)
        hist = df.iloc[:350]
        sigs = s.on_bar("SPY", df.iloc[349], hist)
        for sig in sigs:
            assert sig.direction >= 0
