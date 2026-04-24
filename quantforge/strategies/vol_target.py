"""Volatility-targeting wrapper.

Wraps another Strategy: scales every signal's `strength` so the resulting
portfolio targets a fixed annualized volatility. Standard trick used by risk-parity
funds and AQR-style portable alpha strategies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantforge.core.constants import TRADING_DAYS_YEAR
from quantforge.core.event import SignalEvent
from quantforge.strategies.base import Strategy


@dataclass
class VolTarget(Strategy):
    """Wraps a base strategy and rescales signal strength by realized vol."""
    base: Strategy = None
    target_vol: float = 0.10        # 10% annualized
    vol_lookback: int = 63          # 3 months
    max_leverage: float = 2.0
    min_leverage: float = 0.1
    name: str = "vol_target"
    _last_signal_ts: object = None
    _asset_vols: dict[str, float] = field(default_factory=dict)

    def warmup(self) -> int:
        base_warm = self.base.warmup() if self.base is not None else 0
        return max(base_warm, self.vol_lookback + 2)

    def _asset_vol(self, history: pd.DataFrame) -> float:
        r = history["close"].pct_change().dropna()
        if len(r) < self.vol_lookback:
            return np.nan
        return float(r.iloc[-self.vol_lookback:].std() * np.sqrt(TRADING_DAYS_YEAR))

    def on_bar(self, symbol: str, bar: pd.Series, history: pd.DataFrame) -> list[SignalEvent]:
        if self.base is None:
            return []
        vol = self._asset_vol(history)
        if not np.isnan(vol):
            self._asset_vols[symbol] = vol

        base_signals = self.base.on_bar(symbol, bar, history)
        if not base_signals:
            return []

        scaled = []
        for sig in base_signals:
            asset_vol = self._asset_vols.get(sig.symbol, np.nan)
            if np.isnan(asset_vol) or asset_vol <= 0:
                scale = 1.0
            else:
                scale = self.target_vol / asset_vol
                scale = float(np.clip(scale, self.min_leverage, self.max_leverage))
            new_strength = float(sig.strength) * scale
            scaled.append(self._signal(sig.timestamp, sig.symbol, sig.direction, new_strength, self.name))
        return scaled
