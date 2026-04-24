"""Order primitives."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    symbol: str
    quantity: float
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    created_at: datetime | None = None
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0

    @property
    def signed_qty(self) -> float:
        sign = 1.0 if self.side == OrderSide.BUY else -1.0
        return sign * self.quantity

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.PARTIAL)

    def fill(self, qty: float, price: float) -> None:
        total = self.filled_qty + qty
        if total <= 0:
            return
        self.avg_fill_price = (self.avg_fill_price * self.filled_qty + price * qty) / total
        self.filled_qty = total
        self.status = OrderStatus.FILLED if total >= self.quantity else OrderStatus.PARTIAL
