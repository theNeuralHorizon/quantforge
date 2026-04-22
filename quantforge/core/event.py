"""Event types flowing through the backtest engine."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


@dataclass(frozen=True)
class Event:
    event_type: EventType
    timestamp: datetime


@dataclass(frozen=True)
class MarketEvent(Event):
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def make(cls, ts: datetime, symbol: str, o: float, h: float, l: float, c: float, v: float) -> "MarketEvent":
        return cls(EventType.MARKET, ts, symbol, o, h, l, c, v)


@dataclass(frozen=True)
class SignalEvent(Event):
    symbol: str
    direction: int  # +1 long, -1 short, 0 flat
    strength: float = 1.0  # 0..1 sizing hint
    strategy_id: str = "default"


@dataclass(frozen=True)
class OrderEvent(Event):
    symbol: str
    order_type: str
    quantity: float
    direction: int
    limit_price: Optional[float] = None


@dataclass(frozen=True)
class FillEvent(Event):
    symbol: str
    quantity: float
    direction: int
    fill_price: float
    commission: float
    slippage: float = 0.0
