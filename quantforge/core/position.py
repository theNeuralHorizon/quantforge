"""Position holds quantity, average cost, and realized/unrealized P&L for one symbol."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0
    last_price: float = 0.0
    total_commission: float = 0.0

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < 1e-9

    @property
    def is_long(self) -> bool:
        return self.quantity > 1e-9

    @property
    def is_short(self) -> bool:
        return self.quantity < -1e-9

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.last_price - self.avg_cost) * self.quantity

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl - self.total_commission

    def apply_fill(self, qty: float, price: float, commission: float = 0.0) -> float:
        """Apply a fill (+qty = buy, -qty = sell) and return cash delta."""
        self.total_commission += commission
        cash_delta = -qty * price - commission

        if self.is_flat or (self.quantity > 0 and qty > 0) or (self.quantity < 0 and qty < 0):
            total_qty = self.quantity + qty
            if abs(total_qty) > 1e-12:
                self.avg_cost = (self.avg_cost * self.quantity + price * qty) / total_qty
            self.quantity = total_qty
        else:
            closing = min(abs(qty), abs(self.quantity))
            sign = 1.0 if self.quantity > 0 else -1.0
            self.realized_pnl += sign * closing * (price - self.avg_cost)
            new_qty = self.quantity + qty
            if (self.quantity > 0 and new_qty < 0) or (self.quantity < 0 and new_qty > 0):
                self.avg_cost = price
            self.quantity = new_qty

        self.last_price = price
        return cash_delta

    def mark_to_market(self, price: float) -> None:
        self.last_price = price
