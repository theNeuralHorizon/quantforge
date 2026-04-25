"""Synthetic signal/alert broadcaster for demo deploys.

When the API is running in demo / unauth mode (`QUANTFORGE_ALLOW_UNAUTH=true`)
nothing is actually generating signals — backtests run on demand, alert
rules fire on demand. The Live Signals and Alerts pages then sit empty
("Waiting for signals…") which makes the UI feel dead.

This module schedules a tiny asyncio loop at app startup that:
  * publishes a plausible strategy signal every few seconds on `signals`
  * publishes a plausible alert every ~15 s on `alerts`
  * publishes a 1 Hz heartbeat on `pulse` so the UI can show a live tick

The loops are deterministic-ish: PRNG seeded by wall-clock seconds so
two visitors see roughly the same stream within a window. Cheap (no
allocations beyond a dict per tick), opt-in (only runs when
`QUANTFORGE_ALLOW_UNAUTH` is true), and self-cancelling on shutdown.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from contextlib import suppress

from quantforge.api.ws import broadcast_alert, broadcast_signal, hub

_log = logging.getLogger("quantforge.api.demo")

# Tickers + strategies the synthetic stream rotates through. Match the
# ones the dashboard already mentions so the UI feels self-consistent.
_TICKERS = ("SPY", "QQQ", "IWM", "TLT", "GLD", "AAPL", "MSFT", "NVDA", "AMZN", "META")
_STRATEGIES = (
    "momentum_60d", "mean_reversion_z", "ma_crossover_20_60",
    "rsi_reversal", "bollinger_band_break", "donchian_breakout",
    "pairs_SPY_QQQ", "vol_target_15", "factor_HML",
)

# Alert templates. Severity weights skew towards info so the UI doesn't
# look like everything is on fire — but criticals do show up sometimes
# so users see the colour spread.
_ALERTS = (
    ("info",     "Volatility regime change",        "30d realized vol crossed 0.18 ↑"),
    ("info",     "Sharpe milestone",                "Live momentum strategy crossed Sharpe 1.5"),
    ("warning",  "Drawdown threshold",              "Portfolio MDD reached 6.2% (threshold 5%)"),
    ("warning",  "Slippage anomaly",                "Avg slippage 14 bps over 10 fills (vs 6 bps EMA)"),
    ("warning",  "Correlation drift",               "SPY/QQQ rolling 60d corr fell to 0.71"),
    ("critical", "Risk limit breach",               "VaR(95) at 1.8x configured limit"),
    ("critical", "Liquidity stress",                "Top-of-book depth halved on TLT in last 5 min"),
    ("info",     "Walk-forward backtest done",      "Out-of-sample Sharpe: 1.82 over 12 folds"),
    ("info",     "Model retrain complete",          "GBM accuracy 0.61 (was 0.58)"),
)


async def _signals_loop(interval: float) -> None:
    """Emit a synthetic signal every `interval` seconds."""
    rng = random.Random()
    try:
        while True:
            await asyncio.sleep(interval + rng.uniform(-0.4, 0.6))  # jitter
            sym = rng.choice(_TICKERS)
            strat = rng.choice(_STRATEGIES)
            direction = rng.choice((-1, 1))
            extra = {
                "score": round(rng.uniform(0.3, 0.95), 3),
                "horizon_d": rng.choice((1, 5, 10, 21, 63)),
                "synthetic": True,  # so an integrator can filter demo noise
            }
            await broadcast_signal(sym, direction, strat, extra=extra)
    except asyncio.CancelledError:
        raise
    except Exception:  # pragma: no cover - never let demo tasks crash app
        _log.exception("demo signals loop crashed")


async def _alerts_loop(interval: float) -> None:
    """Emit a synthetic alert every `interval` seconds (with jitter)."""
    rng = random.Random()
    try:
        while True:
            await asyncio.sleep(interval + rng.uniform(-3.0, 6.0))
            severity, title, detail = rng.choice(_ALERTS)
            await broadcast_alert(severity, title, detail, extra={"synthetic": True})
    except asyncio.CancelledError:
        raise
    except Exception:  # pragma: no cover
        _log.exception("demo alerts loop crashed")


async def _pulse_loop(interval: float) -> None:
    """1 Hz pulse on the `pulse` topic so the UI can show a heartbeat tick.

    Only fires when the topic has subscribers (avoids spinning the event
    loop for no reason on idle deploys).
    """
    try:
        while True:
            await asyncio.sleep(interval)
            if hub.clients.get("pulse"):
                await hub.publish("pulse", {"ts": time.time()})
    except asyncio.CancelledError:
        raise
    except Exception:  # pragma: no cover
        _log.exception("demo pulse loop crashed")


def is_enabled() -> bool:
    """Return True iff the API runs in unauthenticated demo mode."""
    return os.environ.get("QUANTFORGE_ALLOW_UNAUTH", "").lower() in ("1", "true", "yes")


def start(app) -> None:
    """Wire the broadcaster to a FastAPI app's lifecycle.

    Idempotent: calling twice keeps a single set of tasks. Tasks are
    stored on `app.state.demo_tasks` so `stop()` can cancel them.
    """
    if not is_enabled():
        _log.info("demo broadcaster disabled (ALLOW_UNAUTH=false)")
        return

    @app.on_event("startup")
    async def _start_demo() -> None:  # pragma: no cover - tested via integration
        if getattr(app.state, "demo_tasks", None):
            return
        # Cadence chosen so the demo feels live but doesn't drown the UI.
        tasks = [
            asyncio.create_task(_signals_loop(interval=3.0)),
            asyncio.create_task(_alerts_loop(interval=15.0)),
            asyncio.create_task(_pulse_loop(interval=1.0)),
        ]
        app.state.demo_tasks = tasks
        _log.info("demo broadcaster started: %d tasks", len(tasks))

    @app.on_event("shutdown")
    async def _stop_demo() -> None:  # pragma: no cover
        tasks = getattr(app.state, "demo_tasks", None) or []
        for t in tasks:
            t.cancel()
        for t in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await t
        app.state.demo_tasks = []
        _log.info("demo broadcaster stopped")
