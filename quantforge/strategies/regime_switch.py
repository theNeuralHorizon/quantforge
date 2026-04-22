"""Macro regime switcher — risk-on/risk-off rotation.

Uses a simple but well-documented signal: when SPY is above its 200-day SMA,
we are in risk-on (long equity); otherwise risk-off (long bonds).

This is a famous defensive strategy that historically reduces drawdowns by ~50%.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from quantforge.core.event import SignalEvent
from quantforge.strategies.base import Strategy


@dataclass
class RegimeSwitch(Strategy):
    risk_on_asset: str = "SPY"
    risk_off_asset: str = "TLT"
    signal_asset: str = "SPY"  # asset whose regime decides allocation
    fast_sma: int = 50
    slow_sma: int = 200
    name: str = "regime_switch"
    _panel: Dict[str, pd.DataFrame] = field(default_factory=dict)
    _last_ts: object = None

    def warmup(self) -> int:
        return self.slow_sma + 2

    def on_bar(self, symbol: str, bar: pd.Series, history: pd.DataFrame) -> List[SignalEvent]:
        self._panel[symbol] = history
        if bar.name == self._last_ts:
            return []
        self._last_ts = bar.name

        signal_hist = self._panel.get(self.signal_asset)
        if signal_hist is None or len(signal_hist) < self.slow_sma + 1:
            return []

        fast = signal_hist["close"].rolling(self.fast_sma).mean().iloc[-1]
        slow = signal_hist["close"].rolling(self.slow_sma).mean().iloc[-1]
        px = signal_hist["close"].iloc[-1]
        risk_on = (px > slow) and (fast > slow)

        target = self.risk_on_asset if risk_on else self.risk_off_asset
        signals = []
        for s in self._panel:
            if s == target:
                signals.append(self._signal(bar.name, s, 1, 1.0, self.name))
            else:
                signals.append(self._signal(bar.name, s, 0, 1.0, self.name))
        return signals
