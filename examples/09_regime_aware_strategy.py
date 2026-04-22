"""Example 09: Regime-aware strategy that switches between momentum and mean reversion."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from quantforge.analytics.performance import summary_stats
from quantforge.analytics.tearsheet import tearsheet_text
from quantforge.backtest import BacktestEngine
from quantforge.core.event import SignalEvent
from quantforge.data.synthetic import generate_ohlcv
from quantforge.indicators.statistical import hurst_exponent
from quantforge.ml.regime import bull_bear_regime, vol_regime
from quantforge.strategies.base import Strategy
from quantforge.strategies.ma_crossover import MACrossoverStrategy
from quantforge.strategies.mean_reversion import MeanReversionStrategy


@dataclass
class RegimeAwareStrategy(Strategy):
    """Uses momentum when Hurst > 0.55, mean reversion when Hurst < 0.45."""
    hurst_window: int = 120
    name: str = "regime_aware"
    momentum: Strategy = field(default_factory=lambda: MACrossoverStrategy(20, 100))
    reversion: Strategy = field(default_factory=lambda: MeanReversionStrategy(20, 2.0, 0.5))
    _last_regime: str = "neutral"

    def warmup(self) -> int:
        return max(self.hurst_window, self.momentum.warmup(), self.reversion.warmup())

    def _classify(self, history: pd.DataFrame) -> str:
        if len(history) < self.hurst_window:
            return "neutral"
        h = hurst_exponent(history["close"].iloc[-self.hurst_window:])
        if np.isnan(h):
            return "neutral"
        if h > 0.55:
            return "trend"
        if h < 0.45:
            return "revert"
        return "neutral"

    def on_bar(self, symbol: str, bar: pd.Series, history: pd.DataFrame) -> List[SignalEvent]:
        regime = self._classify(history)
        self._last_regime = regime
        if regime == "trend":
            return self.momentum.on_bar(symbol, bar, history)
        if regime == "revert":
            return self.reversion.on_bar(symbol, bar, history)
        return []


def main() -> None:
    # Create a dataset that alternates trending / mean-reverting regimes
    df = generate_ohlcv(1000, s0=100, mu=0.12, sigma=0.18, seed=17)
    panel = {"ASSET": df}

    print("Regime classification snapshot:")
    bb = bull_bear_regime(df["close"]).iloc[-20:]
    vr = vol_regime(df["close"].pct_change(), 63).iloc[-20:]
    print(pd.DataFrame({"bull_bear": bb, "vol_regime": vr}).tail(10))

    strat = RegimeAwareStrategy(hurst_window=120)
    engine = BacktestEngine(strategy=strat, data=panel, initial_capital=100_000, sizing_fraction=0.5)
    res = engine.run()

    print("\n" + tearsheet_text(res.equity_curve, "Regime-Aware"))
    print(f"\nTrades: {len(res.trades)}")

    # Compare with pure momentum + pure reversion baselines
    print("\n" + "=" * 60)
    print("Baselines for comparison")
    print("=" * 60)
    for name, s in [("Pure MA", MACrossoverStrategy(20, 100)),
                    ("Pure MR", MeanReversionStrategy(20, 2.0, 0.5))]:
        eng = BacktestEngine(strategy=s, data=panel, initial_capital=100_000, sizing_fraction=0.5)
        r = eng.run()
        stats = summary_stats(r.equity_curve)
        print(f"  {name:>10}  ret: {stats['total_return']*100:+6.2f}%  sharpe: {stats['sharpe']:+.2f}  dd: {stats['max_drawdown']*100:+6.2f}%")


if __name__ == "__main__":
    main()
