"""Cross-sectional / time-series momentum."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from quantforge.core.event import SignalEvent
from quantforge.strategies.base import Strategy


@dataclass
class MomentumStrategy(Strategy):
    """Time-series momentum: long when lookback return positive, flat/short otherwise."""
    lookback: int = 60
    threshold: float = 0.0
    allow_short: bool = False
    name: str = "momentum"

    def warmup(self) -> int:
        return self.lookback + 1

    def on_bar(self, symbol: str, bar: pd.Series, history: pd.DataFrame) -> List[SignalEvent]:
        if len(history) < self.lookback + 1:
            return []
        past = history["close"].iloc[-(self.lookback + 1)]
        now = history["close"].iloc[-1]
        ret = now / past - 1.0
        direction = 0
        if ret > self.threshold:
            direction = 1
        elif ret < -self.threshold and self.allow_short:
            direction = -1
        strength = min(1.0, abs(ret) / max(self.threshold + 1e-6, 0.05))
        return [self._signal(bar.name, symbol, direction, strength, self.name)]
