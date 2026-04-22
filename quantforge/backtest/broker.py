"""SimulatedBroker: executes orders against synthetic / historical market data."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

from quantforge.backtest.commission import CommissionModel, FixedBpsCommission
from quantforge.backtest.slippage import SlippageModel, FixedBpsSlippage
from quantforge.core.event import FillEvent, OrderEvent, EventType
from quantforge.core.order import Order, OrderSide, OrderStatus, OrderType


@dataclass
class SimulatedBroker:
    slippage: SlippageModel = field(default_factory=FixedBpsSlippage)
    commission: CommissionModel = field(default_factory=FixedBpsCommission)
    pending: List[Order] = field(default_factory=list)
    fills: List[FillEvent] = field(default_factory=list)

    def submit(self, order: Order) -> None:
        self.pending.append(order)

    def on_bar(self, ts: datetime, symbol: str, bar: Dict[str, float]) -> List[FillEvent]:
        """Process pending orders for this symbol using the bar (open/high/low/close/volume)."""
        filled_now: List[FillEvent] = []
        still_pending: List[Order] = []

        for order in self.pending:
            if order.symbol != symbol or not order.is_active:
                still_pending.append(order)
                continue

            side = 1 if order.side == OrderSide.BUY else -1
            fill_ref = bar["open"]

            if order.order_type == OrderType.MARKET:
                px = self.slippage.adjust(fill_ref, order.quantity, bar["volume"], side)
            elif order.order_type == OrderType.LIMIT:
                lo, hi = bar["low"], bar["high"]
                if order.side == OrderSide.BUY and lo <= order.limit_price:
                    px = min(order.limit_price, fill_ref)
                elif order.side == OrderSide.SELL and hi >= order.limit_price:
                    px = max(order.limit_price, fill_ref)
                else:
                    still_pending.append(order)
                    continue
            else:
                px = self.slippage.adjust(fill_ref, order.quantity, bar["volume"], side)

            qty = order.quantity
            order.fill(qty, px)
            order.status = OrderStatus.FILLED
            comm = self.commission.charge(px, qty)
            fill = FillEvent(
                event_type=EventType.FILL,
                timestamp=ts,
                symbol=symbol,
                quantity=qty,
                direction=side,
                fill_price=px,
                commission=comm,
                slippage=abs(px - fill_ref),
            )
            filled_now.append(fill)
            self.fills.append(fill)

        self.pending = still_pending
        return filled_now
