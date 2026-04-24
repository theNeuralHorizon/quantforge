"""Donchian channel breakout (turtle-style)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantforge.core.event import SignalEvent
from quantforge.indicators.technical import donchian_channel
from quantforge.strategies.base import Strategy


@dataclass
class DonchianBreakout(Strategy):
    entry_window: int = 20
    exit_window: int = 10
    allow_short: bool = True
    name: str = "donchian"

    def warmup(self) -> int:
        return self.entry_window + 1

    def on_bar(self, symbol: str, bar: pd.Series, history: pd.DataFrame) -> list[SignalEvent]:
        if len(history) < self.entry_window + 1:
            return []
        entry = donchian_channel(history, self.entry_window).iloc[-2]  # signal on prior bar to avoid lookahead
        exit_ = donchian_channel(history, self.exit_window).iloc[-2]
        px = history["close"].iloc[-1]
        if px > entry["upper"]:
            return [self._signal(bar.name, symbol, 1, 1.0, self.name)]
        if px < entry["lower"] and self.allow_short:
            return [self._signal(bar.name, symbol, -1, 1.0, self.name)]
        if px < exit_["lower"] or px > exit_["upper"]:
            return [self._signal(bar.name, symbol, 0, 1.0, self.name)]
        return []
