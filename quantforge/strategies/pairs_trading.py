"""Pairs trading: trade the spread of two cointegrated assets."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantforge.core.event import SignalEvent
from quantforge.indicators.statistical import rolling_beta, rolling_zscore
from quantforge.strategies.base import Strategy


@dataclass
class PairsTradingStrategy(Strategy):
    """Standard stat-arb: spread = asset_a - beta * asset_b, trade on z-score."""
    asset_a: str = "A"
    asset_b: str = "B"
    window: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    name: str = "pairs"
    _last_prices: dict[str, float] = field(default_factory=dict)
    _last_history: dict[str, pd.DataFrame] = field(default_factory=dict)
    _pos_dir: int = 0

    def warmup(self) -> int:
        return self.window + 5

    def on_bar(self, symbol: str, bar: pd.Series, history: pd.DataFrame) -> list[SignalEvent]:
        self._last_prices[symbol] = float(bar["close"])
        self._last_history[symbol] = history
        if self.asset_a not in self._last_prices or self.asset_b not in self._last_prices:
            return []
        if symbol != self.asset_b:  # act only after seeing the 2nd leg of the pair this bar
            return []
        ha = self._last_history[self.asset_a]["close"]
        hb = self._last_history[self.asset_b]["close"]
        ts = min(len(ha), len(hb))
        if ts < self.window + 1:
            return []
        ha = ha.iloc[-ts:]
        hb = hb.iloc[-ts:]
        beta = rolling_beta(ha, hb, self.window).iloc[-1]
        if np.isnan(beta):
            return []
        spread = ha - beta * hb
        z = rolling_zscore(spread, self.window).iloc[-1]
        if np.isnan(z):
            return []
        signals = []
        if z > self.entry_z and self._pos_dir != -1:
            signals = [
                self._signal(bar.name, self.asset_a, -1, 1.0, self.name),
                self._signal(bar.name, self.asset_b, +1, float(min(1.0, abs(beta))), self.name),
            ]
            self._pos_dir = -1
        elif z < -self.entry_z and self._pos_dir != 1:
            signals = [
                self._signal(bar.name, self.asset_a, +1, 1.0, self.name),
                self._signal(bar.name, self.asset_b, -1, float(min(1.0, abs(beta))), self.name),
            ]
            self._pos_dir = 1
        elif abs(z) < self.exit_z and self._pos_dir != 0:
            signals = [
                self._signal(bar.name, self.asset_a, 0, 1.0, self.name),
                self._signal(bar.name, self.asset_b, 0, 1.0, self.name),
            ]
            self._pos_dir = 0
        return signals
