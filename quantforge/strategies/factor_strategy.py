"""Simple multi-factor scoring strategy (momentum + low-vol + value proxy)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from quantforge.core.event import SignalEvent
from quantforge.strategies.base import Strategy


@dataclass
class FactorStrategy(Strategy):
    """Longs top quintile of a composite factor score."""
    lookback_mom: int = 120
    lookback_vol: int = 60
    top_q: float = 0.2
    name: str = "factor_composite"
    _panel: Dict[str, pd.DataFrame] = field(default_factory=dict)
    _last_ts: object = None

    def warmup(self) -> int:
        return max(self.lookback_mom, self.lookback_vol) + 5

    def _score(self, hist: pd.DataFrame) -> float:
        close = hist["close"]
        if len(close) < self.warmup():
            return np.nan
        mom = close.iloc[-1] / close.iloc[-self.lookback_mom] - 1
        vol = close.pct_change().rolling(self.lookback_vol).std().iloc[-1]
        inv_vol = -vol  # prefer low vol
        z_mom = (mom - 0) / max(0.01, close.pct_change().std() * np.sqrt(self.lookback_mom))
        z_vol = (inv_vol - 0) / max(0.01, abs(vol))
        return float(0.5 * z_mom + 0.5 * z_vol)

    def on_bar(self, symbol: str, bar: pd.Series, history: pd.DataFrame) -> List[SignalEvent]:
        self._panel[symbol] = history
        if bar.name == self._last_ts:
            return []
        self._last_ts = bar.name

        scores = {s: self._score(h) for s, h in self._panel.items()}
        valid = {k: v for k, v in scores.items() if not np.isnan(v)}
        if len(valid) < 3:
            return []
        ranked = sorted(valid.items(), key=lambda kv: kv[1], reverse=True)
        n_long = max(1, int(len(ranked) * self.top_q))
        longs = {k for k, _ in ranked[:n_long]}
        signals = []
        weight = 1.0 / n_long
        for s in self._panel:
            if s in longs:
                signals.append(self._signal(bar.name, s, 1, weight, self.name))
            else:
                signals.append(self._signal(bar.name, s, 0, 1.0, self.name))
        return signals
