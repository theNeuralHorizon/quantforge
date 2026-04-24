"""Classic execution algorithms.

All algorithms take an OHLCV frame covering the execution window and
return an ExecutionReport with per-bar child orders, cumulative fill,
average execution price, and slippage vs. arrival / VWAP benchmarks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExecutionReport:
    schedule: pd.DataFrame         # per-bar: qty, px, notional, pct_of_bar_volume
    total_qty: float               # signed total executed
    avg_price: float               # VWAP of our own fills
    arrival_price: float           # first bar's open (benchmark)
    slippage_bps_vs_arrival: float # (avg_price - arrival)/arrival * 1e4 * sign
    slippage_bps_vs_vwap: float    # (avg_price - market_vwap)/market_vwap * 1e4 * sign
    market_vwap: float
    horizon_bars: int

    def summary(self) -> str:
        return (
            f"qty={self.total_qty:>+.0f}  avg_px={self.avg_price:.4f}  "
            f"arrival={self.arrival_price:.4f}  "
            f"vs_arrival={self.slippage_bps_vs_arrival:+.2f}bp  "
            f"vs_vwap={self.slippage_bps_vs_vwap:+.2f}bp"
        )


def _bars_dataframe(ohlcv: pd.DataFrame) -> pd.DataFrame:
    req = {"open", "high", "low", "close", "volume"}
    if not req.issubset(ohlcv.columns):
        raise ValueError(f"ohlcv needs columns {req}, got {set(ohlcv.columns)}")
    if ohlcv.empty:
        raise ValueError("empty ohlcv")
    return ohlcv


def _build_report(bars: pd.DataFrame, qty_per_bar: np.ndarray,
                   px_per_bar: np.ndarray, signed: bool) -> ExecutionReport:
    sign = 1 if signed else -1
    # per-bar execution prices — we model a bar as its typical (h+l+c)/3 unless caller overrides
    # (callers always pass explicit px_per_bar for realism)
    notional = qty_per_bar * px_per_bar
    schedule = pd.DataFrame({
        "qty": qty_per_bar * sign,
        "px": px_per_bar,
        "notional": notional * sign,
        "pct_of_bar_volume": np.where(bars["volume"].values > 0,
                                       qty_per_bar / bars["volume"].values, 0.0),
    }, index=bars.index)

    total = float(qty_per_bar.sum())
    if total <= 0:
        return ExecutionReport(
            schedule=schedule, total_qty=0.0, avg_price=float("nan"),
            arrival_price=float(bars["open"].iloc[0]),
            slippage_bps_vs_arrival=0.0, slippage_bps_vs_vwap=0.0,
            market_vwap=float("nan"), horizon_bars=len(bars),
        )

    avg_px = float((qty_per_bar * px_per_bar).sum() / total)
    arrival = float(bars["open"].iloc[0])
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    market_vwap = float((typical * bars["volume"]).sum() / bars["volume"].sum())

    def bps(a, b):
        return 0.0 if b == 0 else (a - b) / b * 1e4

    return ExecutionReport(
        schedule=schedule,
        total_qty=total * sign,
        avg_price=avg_px,
        arrival_price=arrival,
        slippage_bps_vs_arrival=sign * bps(avg_px, arrival),
        slippage_bps_vs_vwap=sign * bps(avg_px, market_vwap),
        market_vwap=market_vwap,
        horizon_bars=len(bars),
    )


def twap(ohlcv: pd.DataFrame, quantity: float, side: str = "buy") -> ExecutionReport:
    """Time-Weighted Average Price: split `quantity` equally across bars.

    Execution price for each slice is modelled as the bar's typical price
    (h+l+c)/3 (reasonable proxy for a mid-bar fill).
    """
    bars = _bars_dataframe(ohlcv)
    n = len(bars)
    qty_per_bar = np.full(n, abs(quantity) / n)
    px_per_bar = ((bars["high"] + bars["low"] + bars["close"]) / 3).values
    return _build_report(bars, qty_per_bar, px_per_bar, signed=(side == "buy"))


def vwap(ohlcv: pd.DataFrame, quantity: float, side: str = "buy",
          volume_curve: np.ndarray | None = None) -> ExecutionReport:
    """Volume-Weighted Average Price: allocate proportional to bar volume.

    `volume_curve` lets the caller pass a historical intraday volume profile
    (same length as ohlcv). Default uses the realized bar volumes.
    """
    bars = _bars_dataframe(ohlcv)
    if volume_curve is not None:
        weights = np.asarray(volume_curve, dtype=float)
        if len(weights) != len(bars):
            raise ValueError("volume_curve must match ohlcv length")
    else:
        weights = bars["volume"].values.astype(float)
    total_w = weights.sum()
    if total_w <= 0:
        # degenerate; fall back to TWAP
        return twap(ohlcv, quantity, side)
    qty_per_bar = abs(quantity) * weights / total_w
    px_per_bar = ((bars["high"] + bars["low"] + bars["close"]) / 3).values
    return _build_report(bars, qty_per_bar, px_per_bar, signed=(side == "buy"))


def pov(ohlcv: pd.DataFrame, participation_rate: float, side: str = "buy",
         max_quantity: float | None = None) -> ExecutionReport:
    """Percent-of-Volume (Participation) algo.

    Buys/sells `participation_rate` * bar volume on each bar until the
    `max_quantity` cap is hit (or horizon ends). Good for liquidity-sensitive
    execution.
    """
    if not (0 < participation_rate <= 1.0):
        raise ValueError("participation_rate must be in (0, 1]")
    bars = _bars_dataframe(ohlcv)
    volumes = bars["volume"].values
    qty_per_bar = volumes * participation_rate
    if max_quantity is not None and max_quantity > 0:
        cum = np.cumsum(qty_per_bar)
        over = cum > max_quantity
        if over.any():
            first_over = int(np.argmax(over))
            qty_per_bar = qty_per_bar.copy()
            remaining = max_quantity - (cum[first_over - 1] if first_over > 0 else 0)
            qty_per_bar[first_over] = max(0.0, remaining)
            qty_per_bar[first_over + 1:] = 0.0
    px_per_bar = ((bars["high"] + bars["low"] + bars["close"]) / 3).values
    return _build_report(bars, qty_per_bar, px_per_bar, signed=(side == "buy"))


def implementation_shortfall(
    ohlcv: pd.DataFrame,
    quantity: float,
    side: str = "buy",
    risk_aversion: float = 1e-6,
    volatility: float | None = None,
    temporary_impact_coef: float = 1e-5,
    permanent_impact_coef: float = 1e-7,
) -> ExecutionReport:
    """Almgren-Chriss implementation shortfall.

    Optimally trades off market-impact cost against timing (variance) risk.
    Returns the IS-optimal schedule using the classic exponential decay form.

    Minimizes:   E[cost] + risk_aversion * Var[cost]
    where cost = temporary_impact * (q_i / t)^2 + permanent_impact * q_i
    and risk = sigma^2 * sum( (X_i)^2 * t_i ) with X_i = shares remaining.

    We use the closed-form optimal trajectory:
        x_j = X * sinh(kappa (T - t_j)) / sinh(kappa T)
    with kappa = sqrt(risk_aversion * sigma^2 / temporary_impact).
    """
    bars = _bars_dataframe(ohlcv)
    n = len(bars)
    X = abs(quantity)
    if volatility is None:
        r = bars["close"].pct_change().dropna()
        volatility = float(r.std()) if len(r) > 1 else 0.01
    sigma = max(volatility, 1e-6)

    if temporary_impact_coef <= 0 or risk_aversion <= 0:
        return twap(ohlcv, quantity, side)

    kappa_sq = risk_aversion * sigma * sigma / temporary_impact_coef
    kappa = float(np.sqrt(kappa_sq))
    T_bars = float(n)

    # trajectory of remaining shares at end of each bar
    t_end = np.arange(1, n + 1, dtype=float)
    denom = np.sinh(kappa * T_bars) if kappa * T_bars > 1e-9 else 1.0
    remaining = X * np.sinh(kappa * (T_bars - t_end)) / denom
    remaining = np.clip(remaining, 0.0, X)

    # slices = decrement of remaining
    qty_per_bar = np.empty(n)
    qty_per_bar[0] = X - remaining[0]
    for i in range(1, n):
        qty_per_bar[i] = remaining[i - 1] - remaining[i]
    qty_per_bar = np.maximum(qty_per_bar, 0.0)
    # fix any rounding that leaves unsold quantity
    diff = X - qty_per_bar.sum()
    if abs(diff) > 1e-9:
        qty_per_bar[-1] += diff

    # Execution price model includes permanent + temporary impact
    typical = ((bars["high"] + bars["low"] + bars["close"]) / 3).values
    side_sign = 1 if side == "buy" else -1
    cum_exec = np.cumsum(qty_per_bar)
    permanent = permanent_impact_coef * cum_exec
    temporary = temporary_impact_coef * qty_per_bar  # per-bar temp kick
    px_per_bar = typical * (1 + side_sign * (permanent + temporary))

    return _build_report(bars, qty_per_bar, px_per_bar, signed=(side == "buy"))
