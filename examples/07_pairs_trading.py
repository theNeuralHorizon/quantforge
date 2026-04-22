"""Example 07: Pairs trading on two cointegrated synthetic series."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.analytics.tearsheet import tearsheet_text
from quantforge.backtest import BacktestEngine
from quantforge.data.synthetic import generate_ohlcv
from quantforge.indicators.statistical import adf_test, half_life
from quantforge.strategies.pairs_trading import PairsTradingStrategy


def _make_cointegrated_pair(n: int = 800, seed: int = 42):
    """Build two series where A = 0.8*B + stationary noise."""
    rng = np.random.default_rng(seed)
    b_rets = rng.normal(0.0003, 0.012, n)
    B = 100 * np.exp(np.cumsum(b_rets))
    noise = np.zeros(n)
    phi = 0.85
    for t in range(1, n):
        noise[t] = phi * noise[t - 1] + rng.normal(0, 0.4)
    A = 0.8 * B + noise + 20
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    dfa = pd.DataFrame({"open": A, "high": A, "low": A, "close": A, "volume": 1e6}, index=idx)
    dfb = pd.DataFrame({"open": B, "high": B, "low": B, "close": B, "volume": 1e6}, index=idx)
    return {"A": dfa, "B": dfb}


def main() -> None:
    panel = _make_cointegrated_pair(n=800, seed=42)
    spread = panel["A"]["close"] - 0.8 * panel["B"]["close"]

    stat, p = adf_test(spread)
    hl = half_life(spread)
    print(f"Spread ADF stat={stat:.3f} pvalue={p:.3f} (low p -> stationary)")
    print(f"Half-life: {hl:.1f} bars\n")

    strat = PairsTradingStrategy(asset_a="A", asset_b="B", window=60, entry_z=2.0, exit_z=0.5)
    engine = BacktestEngine(strategy=strat, data=panel, initial_capital=100_000, sizing_fraction=0.3)
    res = engine.run()
    print(tearsheet_text(res.equity_curve, "Pairs A-B"))


if __name__ == "__main__":
    main()
