"""Buy-and-hold benchmark strategy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set

import pandas as pd

from quantforge.core.event import SignalEvent
from quantforge.strategies.base import Strategy


@dataclass
class BuyAndHoldStrategy(Strategy):
    """Enters a long position on the first warm bar for each symbol and holds forever.

    Acts as the reference benchmark for any other strategy.
    """
    name: str = "buy_and_hold"
    _entered: Set[str] = field(default_factory=set)

    def warmup(self) -> int:
        return 1

    def on_bar(self, symbol: str, bar: pd.Series, history: pd.DataFrame) -> List[SignalEvent]:
        if symbol in self._entered:
            return []
        self._entered.add(symbol)
        return [self._signal(bar.name, symbol, 1, 1.0, self.name)]
