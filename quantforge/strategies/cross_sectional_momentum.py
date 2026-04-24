"""Cross-sectional momentum (Jegadeesh-Titman 1993).

At each rebalance, rank assets by 12-month (or configurable) return excluding the
most-recent month (to avoid short-term reversal). Long the top quantile, optionally
short the bottom quantile. This is one of the most documented anomalies in finance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantforge.core.event import SignalEvent
from quantforge.strategies.base import Strategy


@dataclass
class CrossSectionalMomentum(Strategy):
    lookback: int = 252        # 12 months of daily bars
    skip: int = 21             # skip the last 1 month (short-term reversal)
    top_q: float = 0.4         # top 40% = long
    bottom_q: float = 0.0      # 0 = long only; 0.4 would short bottom 40%
    name: str = "cross_sectional_momentum"
    _panel: dict[str, pd.DataFrame] = field(default_factory=dict)
    _last_ts: object = None

    def warmup(self) -> int:
        return self.lookback + self.skip + 5

    def _score(self, history: pd.DataFrame) -> float:
        close = history["close"]
        if len(close) < self.lookback + self.skip + 1:
            return np.nan
        end_idx = -self.skip - 1 if self.skip > 0 else -1
        start_idx = end_idx - self.lookback
        return float(close.iloc[end_idx] / close.iloc[start_idx] - 1)

    def on_bar(self, symbol: str, bar: pd.Series, history: pd.DataFrame) -> list[SignalEvent]:
        self._panel[symbol] = history
        if bar.name == self._last_ts:
            return []
        self._last_ts = bar.name

        scores = {s: self._score(h) for s, h in self._panel.items()}
        valid = {k: v for k, v in scores.items() if not np.isnan(v)}
        if len(valid) < 2:
            return []

        ranked = sorted(valid.items(), key=lambda kv: kv[1], reverse=True)
        n = len(ranked)
        n_long = max(1, int(round(n * self.top_q)))
        n_short = int(round(n * self.bottom_q))
        longs = {k for k, _ in ranked[:n_long]}
        shorts = {k for k, _ in ranked[-n_short:]} if n_short > 0 else set()

        long_w = 1.0 / n_long if n_long > 0 else 0.0
        short_w = 1.0 / n_short if n_short > 0 else 0.0

        signals = []
        for s in self._panel:
            if s in longs:
                signals.append(self._signal(bar.name, s, 1, long_w, self.name))
            elif s in shorts:
                signals.append(self._signal(bar.name, s, -1, short_w, self.name))
            else:
                signals.append(self._signal(bar.name, s, 0, 1.0, self.name))
        return signals
