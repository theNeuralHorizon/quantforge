"""Tests for the demo broadcaster (demo-mode signal/alert auto-emit)."""
from __future__ import annotations

import asyncio

import pytest

from quantforge.api import demo_broadcaster as db


def _run(coro):
    """Tiny shim so we don't have to depend on pytest-asyncio."""
    return asyncio.run(coro)


class TestIsEnabled:
    def test_default_is_disabled(self, monkeypatch):
        monkeypatch.delenv("QUANTFORGE_ALLOW_UNAUTH", raising=False)
        assert db.is_enabled() is False

    @pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "True"])
    def test_truthy_values_enable(self, monkeypatch, value):
        monkeypatch.setenv("QUANTFORGE_ALLOW_UNAUTH", value)
        assert db.is_enabled() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "", "off"])
    def test_falsy_values_disable(self, monkeypatch, value):
        monkeypatch.setenv("QUANTFORGE_ALLOW_UNAUTH", value)
        assert db.is_enabled() is False


class TestSignalsLoop:
    def test_emits_signals_periodically(self, monkeypatch):
        """The loop should call broadcast_signal at least once on a short
        interval. Swap the broadcaster for a counter and let it tick."""
        calls: list[tuple] = []

        async def fake_broadcast(symbol, direction, strategy, extra=None):
            calls.append((symbol, direction, strategy, extra))
            return 0

        monkeypatch.setattr(db, "broadcast_signal", fake_broadcast)

        async def drive():
            task = asyncio.create_task(db._signals_loop(interval=0.05))
            try:
                # First signal lands after interval + jitter (max ~0.65s);
                # wait long enough to see at least one cycle even on a
                # slow CI runner.
                await asyncio.sleep(1.5)
            finally:
                task.cancel()
                with __import__("contextlib").suppress(asyncio.CancelledError):
                    await task

        _run(drive())

        assert len(calls) >= 1, "broadcaster did not emit any signals"
        for sym, direction, strat, extra in calls:
            assert sym in db._TICKERS
            assert direction in (-1, 1)
            assert strat in db._STRATEGIES
            assert extra and extra.get("synthetic") is True
            assert 0.0 <= extra["score"] <= 1.0


class TestAlertsLoop:
    def test_alerts_loop_runs_cleanly(self, monkeypatch):
        """The alerts loop has a wide jitter (-3 to +6s) so it may not
        actually emit in a tight test window. The important contract is
        that it doesn't crash and that emitted alerts have the right shape."""
        calls: list[tuple] = []

        async def fake_broadcast(severity, title, detail, extra=None):
            calls.append((severity, title, detail, extra))
            return 0

        monkeypatch.setattr(db, "broadcast_alert", fake_broadcast)

        async def drive():
            task = asyncio.create_task(db._alerts_loop(interval=0.001))
            try:
                # The loop's negative jitter floor + 0 interval means
                # asyncio.sleep often clamps to 0 — but the upper jitter
                # bound (+6s) means we can't guarantee an emission. We
                # just verify the loop doesn't crash.
                await asyncio.sleep(0.05)
            finally:
                task.cancel()
                with __import__("contextlib").suppress(asyncio.CancelledError):
                    await task

        _run(drive())

        for severity, title, _detail, extra in calls:
            assert severity in {"info", "warning", "critical"}
            assert title
            assert extra and extra.get("synthetic") is True


class TestPulseLoop:
    def test_pulse_only_publishes_when_subscribed(self, monkeypatch):
        """The pulse loop should noop unless someone is listening on
        the `pulse` topic."""
        published: list[dict] = []

        async def fake_publish(topic, msg):
            published.append({"topic": topic, **msg})
            return 0

        monkeypatch.setattr(db.hub, "publish", fake_publish)
        monkeypatch.setattr(db.hub, "clients", {})

        async def drive():
            task = asyncio.create_task(db._pulse_loop(interval=0.05))
            try:
                await asyncio.sleep(0.2)
            finally:
                task.cancel()
                with __import__("contextlib").suppress(asyncio.CancelledError):
                    await task

        _run(drive())
        assert published == []

    def test_pulse_publishes_when_subscribed(self, monkeypatch):
        published: list[dict] = []

        async def fake_publish(topic, msg):
            published.append({"topic": topic, **msg})
            return 1

        monkeypatch.setattr(db.hub, "publish", fake_publish)
        monkeypatch.setattr(db.hub, "clients", {"pulse": {"fake_socket"}})

        async def drive():
            task = asyncio.create_task(db._pulse_loop(interval=0.05))
            try:
                await asyncio.sleep(0.3)
            finally:
                task.cancel()
                with __import__("contextlib").suppress(asyncio.CancelledError):
                    await task

        _run(drive())
        assert len(published) >= 1
        assert all(p["topic"] == "pulse" and "ts" in p for p in published)


class TestStartIntegration:
    def test_start_noop_when_disabled(self, monkeypatch):
        """When ALLOW_UNAUTH=false, start() must not register any tasks."""
        from fastapi import FastAPI
        monkeypatch.delenv("QUANTFORGE_ALLOW_UNAUTH", raising=False)
        app = FastAPI()
        db.start(app)
        assert getattr(app.state, "demo_tasks", None) is None
