"""RSI mean-reversion / reversal."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantforge.core.event import SignalEvent
from quantforge.indicators.technical import rsi
from quantforge.strategies.base import Strategy


@dataclass
class RSIReversalStrategy(Strategy):
    window: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    allow_short: bool = True
    name: str = "rsi_reversal"

    def warmup(self) -> int:
        return self.window * 2

    def on_bar(self, symbol: str, bar: pd.Series, history: pd.DataFrame) -> list[SignalEvent]:
        if len(history) < self.window * 2:
            return []
        r = rsi(history["close"], self.window).iloc[-1]
        if pd.isna(r):
            return []
        if r < self.oversold:
            return [self._signal(bar.name, symbol, 1, 1.0, self.name)]
        if r > self.overbought and self.allow_short:
            return [self._signal(bar.name, symbol, -1, 1.0, self.name)]
        if 45 < r < 55:
            return [self._signal(bar.name, symbol, 0, 1.0, self.name)]
        return []
