"""Dual Momentum (Gary Antonacci, 2014).

Absolute momentum: keep asset only if 12m return > risk-free (else go to cash/bonds).
Relative momentum: among qualifying risky assets, pick the single best one.

This is one of the simplest strategies with documented, persistent out-of-sample alpha.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from quantforge.core.event import SignalEvent
from quantforge.strategies.base import Strategy


@dataclass
class DualMomentum(Strategy):
    lookback: int = 252
    cash_asset: str = "TLT"  # fall-back "cash" — long-duration bonds
    risk_free_daily: float = 0.0  # absolute momentum hurdle (daily return)
    name: str = "dual_momentum"
    _panel: Dict[str, pd.DataFrame] = field(default_factory=dict)
    _last_ts: object = None

    def warmup(self) -> int:
        return self.lookback + 5

    def _return(self, history: pd.DataFrame) -> float:
        close = history["close"]
        if len(close) < self.lookback + 1:
            return np.nan
        return float(close.iloc[-1] / close.iloc[-self.lookback - 1] - 1)

    def on_bar(self, symbol: str, bar: pd.Series, history: pd.DataFrame) -> List[SignalEvent]:
        self._panel[symbol] = history
        if bar.name == self._last_ts:
            return []
        self._last_ts = bar.name

        rets = {s: self._return(h) for s, h in self._panel.items()}
        rets = {k: v for k, v in rets.items() if not np.isnan(v)}
        if not rets:
            return []

        # Risk-on universe = all except the cash proxy
        risky = {k: v for k, v in rets.items() if k != self.cash_asset}
        # Absolute momentum filter: keep only those beating risk-free over the window
        rf_cum = (1 + self.risk_free_daily) ** self.lookback - 1
        qualified = {k: v for k, v in risky.items() if v > rf_cum}

        if qualified:
            winner = max(qualified, key=qualified.get)
            target = winner
        else:
            target = self.cash_asset if self.cash_asset in self._panel else None

        signals = []
        for s in self._panel:
            if s == target:
                signals.append(self._signal(bar.name, s, 1, 1.0, self.name))
            else:
                signals.append(self._signal(bar.name, s, 0, 1.0, self.name))
        return signals
