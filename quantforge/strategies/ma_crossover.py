"""Classic moving-average crossover."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantforge.core.event import SignalEvent
from quantforge.indicators.technical import ema
from quantforge.strategies.base import Strategy


@dataclass
class MACrossoverStrategy(Strategy):
    fast: int = 10
    slow: int = 30
    allow_short: bool = False
    name: str = "ma_crossover"

    def warmup(self) -> int:
        return self.slow + 2

    def on_bar(self, symbol: str, bar: pd.Series, history: pd.DataFrame) -> list[SignalEvent]:
        if len(history) < self.slow + 2:
            return []
        close = history["close"]
        fast = ema(close, self.fast)
        slow = ema(close, self.slow)
        diff = fast - slow
        if diff.iloc[-2] <= 0 < diff.iloc[-1]:
            return [self._signal(bar.name, symbol, 1, 1.0, self.name)]
        if diff.iloc[-2] >= 0 > diff.iloc[-1]:
            if self.allow_short:
                return [self._signal(bar.name, symbol, -1, 1.0, self.name)]
            return [self._signal(bar.name, symbol, 0, 1.0, self.name)]
        return []
