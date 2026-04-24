"""Strategy base class."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from quantforge.core.event import EventType, SignalEvent


class Strategy(ABC):
    name: str = "base"

    def warmup(self) -> int:
        return 20

    @abstractmethod
    def on_bar(self, symbol: str, bar: pd.Series, history: pd.DataFrame) -> list[SignalEvent]:
        """Produce signals for this symbol given current bar and full history."""

    @staticmethod
    def _signal(ts, symbol: str, direction: int, strength: float = 1.0, strategy_id: str = "default") -> SignalEvent:
        return SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=ts,
            symbol=symbol,
            direction=direction,
            strength=strength,
            strategy_id=strategy_id,
        )
