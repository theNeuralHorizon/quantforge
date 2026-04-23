"""SimulatedBroker: executes orders against synthetic / historical market data.

Supports MARKET, LIMIT, STOP, and STOP_LIMIT order types with realistic
intrabar fill logic (bars are treated as a [low, high] price path; we
assume the path hits stops before limits when both trigger in the same bar).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from quantforge.backtest.commission import CommissionModel, FixedBpsCommission
from quantforge.backtest.slippage import SlippageModel, FixedBpsSlippage
from quantforge.core.event import EventType, FillEvent
from quantforge.core.order import Order, OrderSide, OrderStatus, OrderType


def _limit_fill_price(order: Order, bar: Dict[str, float]) -> Optional[float]:
    """Return fill price for a LIMIT if triggered this bar, else None."""
    lo, hi, op = bar["low"], bar["high"], bar["open"]
    if order.side == OrderSide.BUY:
        if lo <= order.limit_price:
            # Conservative: if open is below limit, we'd have filled at open;
            # otherwise filled exactly at the limit.
            return min(order.limit_price, op)
        return None
    else:  # SELL
        if hi >= order.limit_price:
            return max(order.limit_price, op)
        return None


def _stop_triggered(order: Order, bar: Dict[str, float]) -> Optional[float]:
    """For a STOP order, return the trigger price if hit during the bar, else None.

    Stop buy triggers at stop_price when high >= stop_price.
    Stop sell triggers at stop_price when low <= stop_price.
    """
    lo, hi, op = bar["low"], bar["high"], bar["open"]
    sp = order.stop_price
    if sp is None:
        return None
    if order.side == OrderSide.BUY:
        if op >= sp:
            # gap-up through the stop: fill at open
            return op
        if hi >= sp:
            return sp
        return None
    else:
        if op <= sp:
            return op
        if lo <= sp:
            return sp
        return None


@dataclass
class SimulatedBroker:
    slippage: SlippageModel = field(default_factory=FixedBpsSlippage)
    commission: CommissionModel = field(default_factory=FixedBpsCommission)
    pending: List[Order] = field(default_factory=list)
    fills: List[FillEvent] = field(default_factory=list)

    def submit(self, order: Order) -> None:
        self.pending.append(order)

    def cancel(self, order_id: str) -> bool:
        for o in self.pending:
            if o.order_id == order_id and o.is_active:
                o.status = OrderStatus.CANCELLED
                return True
        return False

    def on_bar(self, ts: datetime, symbol: str, bar: Dict[str, float]) -> List[FillEvent]:
        """Process pending orders for this symbol using the bar."""
        filled_now: List[FillEvent] = []
        still_pending: List[Order] = []

        for order in self.pending:
            if order.symbol != symbol or not order.is_active:
                still_pending.append(order)
                continue

            side = 1 if order.side == OrderSide.BUY else -1
            fill_ref = bar["open"]
            px: Optional[float] = None

            if order.order_type == OrderType.MARKET:
                px = self.slippage.adjust(fill_ref, order.quantity, bar["volume"], side)

            elif order.order_type == OrderType.LIMIT:
                px = _limit_fill_price(order, bar)

            elif order.order_type == OrderType.STOP:
                # On stop trigger, fills at the trigger price (plus slippage)
                trig = _stop_triggered(order, bar)
                if trig is not None:
                    px = self.slippage.adjust(trig, order.quantity, bar["volume"], side)

            elif order.order_type == OrderType.STOP_LIMIT:
                # Once stop triggers, the order becomes a limit with limit_price
                trig = _stop_triggered(order, bar)
                if trig is not None:
                    # After trigger, check if the limit would fill within the bar too
                    px = _limit_fill_price(order, bar)
                    # If not immediately filled, keep it pending as a live LIMIT order
                    if px is None:
                        order.order_type = OrderType.LIMIT
                        still_pending.append(order)
                        continue

            if px is None:
                still_pending.append(order)
                continue

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
