"""Portfolio aggregates positions, tracks cash and equity."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Mapping, Tuple

from quantforge.core.position import Position


@dataclass
class Portfolio:
    initial_capital: float = 100_000.0
    cash: float = field(init=False)
    positions: Dict[str, Position] = field(default_factory=dict)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.initial_capital

    def position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def apply_fill(self, symbol: str, signed_qty: float, price: float, commission: float = 0.0) -> None:
        pos = self.position(symbol)
        cash_delta = pos.apply_fill(signed_qty, price, commission)
        self.cash += cash_delta

    def mark_to_market(self, prices: Mapping[str, float]) -> None:
        for sym, p in prices.items():
            if sym in self.positions:
                self.positions[sym].mark_to_market(p)

    @property
    def market_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def equity(self) -> float:
        return self.cash + self.market_value

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions.values())

    @property
    def net_exposure(self) -> float:
        return self.market_value

    @property
    def leverage(self) -> float:
        eq = self.equity
        return self.gross_exposure / eq if eq > 0 else 0.0

    def record_equity(self, ts: datetime) -> None:
        self.equity_curve.append((ts, self.equity))

    def weights(self) -> Dict[str, float]:
        eq = self.equity
        if eq <= 0:
            return {}
        return {s: p.market_value / eq for s, p in self.positions.items()}

    def snapshot(self) -> dict:
        return {
            "cash": self.cash,
            "equity": self.equity,
            "market_value": self.market_value,
            "leverage": self.leverage,
            "positions": {s: {"qty": p.quantity, "avg_cost": p.avg_cost, "mv": p.market_value, "upnl": p.unrealized_pnl} for s, p in self.positions.items()},
        }
