"""Mean reversion strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from quantforge.core.event import SignalEvent
from quantforge.indicators.technical import bollinger_bands
from quantforge.indicators.statistical import rolling_zscore
from quantforge.strategies.base import Strategy


@dataclass
class MeanReversionStrategy(Strategy):
    """Z-score mean reversion: long when z < -entry, short when z > entry."""
    lookback: int = 20
    entry_z: float = 2.0
    exit_z: float = 0.5
    allow_short: bool = True
    name: str = "mean_reversion_zscore"

    def warmup(self) -> int:
        return self.lookback + 1

    def on_bar(self, symbol: str, bar: pd.Series, history: pd.DataFrame) -> List[SignalEvent]:
        if len(history) < self.lookback + 1:
            return []
        z = rolling_zscore(history["close"], self.lookback).iloc[-1]
        if np.isnan(z):
            return []
        direction = 0
        if z < -self.entry_z:
            direction = 1
        elif z > self.entry_z and self.allow_short:
            direction = -1
        elif abs(z) < self.exit_z:
            direction = 0
        else:
            return []
        return [self._signal(bar.name, symbol, direction, min(1.0, abs(z) / self.entry_z), self.name)]


@dataclass
class BollingerMeanReversion(Strategy):
    """Bollinger Band mean reversion."""
    window: int = 20
    k: float = 2.0
    name: str = "bollinger_mr"

    def warmup(self) -> int:
        return self.window + 1

    def on_bar(self, symbol: str, bar: pd.Series, history: pd.DataFrame) -> List[SignalEvent]:
        if len(history) < self.window + 1:
            return []
        bb = bollinger_bands(history["close"], self.window, self.k).iloc[-1]
        px = history["close"].iloc[-1]
        if px < bb["lower"]:
            return [self._signal(bar.name, symbol, 1, 1.0, self.name)]
        if px > bb["upper"]:
            return [self._signal(bar.name, symbol, -1, 1.0, self.name)]
        if bb["lower"] <= px <= bb["upper"]:
            return [self._signal(bar.name, symbol, 0, 1.0, self.name)]
        return []
