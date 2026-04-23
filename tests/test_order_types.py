"""Tests for LIMIT / STOP / STOP_LIMIT order behaviour in the simulated broker."""
from __future__ import annotations

from datetime import datetime

import pytest

from quantforge.backtest.broker import SimulatedBroker
from quantforge.backtest.commission import NoCommission
from quantforge.backtest.slippage import NoSlippage
from quantforge.core.order import Order, OrderSide, OrderStatus, OrderType


@pytest.fixture
def broker():
    return SimulatedBroker(slippage=NoSlippage(), commission=NoCommission())


def _bar(o=100, h=102, l=98, c=101, v=1_000_000):
    return {"open": o, "high": h, "low": l, "close": c, "volume": v}


class TestLimit:
    def test_buy_limit_fills_when_low_hits_limit(self, broker):
        order = Order(symbol="X", quantity=10, side=OrderSide.BUY,
                       order_type=OrderType.LIMIT, limit_price=99.0)
        broker.submit(order)
        fills = broker.on_bar(datetime.now(), "X", _bar(100, 102, 97, 101))
        assert len(fills) == 1
        # Fill should be at min(limit, open) = 99 (limit, since open=100 > 99)
        assert fills[0].fill_price == pytest.approx(99.0)
        assert order.status == OrderStatus.FILLED

    def test_buy_limit_fills_at_open_if_gapped_below(self, broker):
        order = Order(symbol="X", quantity=10, side=OrderSide.BUY,
                       order_type=OrderType.LIMIT, limit_price=100.0)
        broker.submit(order)
        # Open below limit → fill at open (favorable fill)
        fills = broker.on_bar(datetime.now(), "X", _bar(o=95, h=101, l=94, c=98))
        assert len(fills) == 1
        assert fills[0].fill_price == pytest.approx(95.0)

    def test_buy_limit_stays_pending_if_not_hit(self, broker):
        order = Order(symbol="X", quantity=10, side=OrderSide.BUY,
                       order_type=OrderType.LIMIT, limit_price=90.0)
        broker.submit(order)
        fills = broker.on_bar(datetime.now(), "X", _bar(100, 102, 95, 101))
        assert len(fills) == 0
        assert order.is_active

    def test_sell_limit_fills_when_high_reaches(self, broker):
        order = Order(symbol="X", quantity=10, side=OrderSide.SELL,
                       order_type=OrderType.LIMIT, limit_price=102.0)
        broker.submit(order)
        fills = broker.on_bar(datetime.now(), "X", _bar(100, 103, 98, 101))
        assert len(fills) == 1
        assert fills[0].fill_price == pytest.approx(102.0)


class TestStop:
    def test_buy_stop_triggers_when_high_reaches(self, broker):
        order = Order(symbol="X", quantity=10, side=OrderSide.BUY,
                       order_type=OrderType.STOP, stop_price=105.0)
        broker.submit(order)
        fills = broker.on_bar(datetime.now(), "X", _bar(100, 106, 99, 104))
        assert len(fills) == 1
        assert fills[0].fill_price == pytest.approx(105.0)

    def test_buy_stop_gap_fills_at_open(self, broker):
        order = Order(symbol="X", quantity=10, side=OrderSide.BUY,
                       order_type=OrderType.STOP, stop_price=105.0)
        broker.submit(order)
        # Gap up through stop at the open — fill at open (worse than stop)
        fills = broker.on_bar(datetime.now(), "X", _bar(o=108, h=110, l=107, c=109))
        assert len(fills) == 1
        assert fills[0].fill_price == pytest.approx(108.0)

    def test_sell_stop_triggers_when_low_reaches(self, broker):
        order = Order(symbol="X", quantity=10, side=OrderSide.SELL,
                       order_type=OrderType.STOP, stop_price=95.0)
        broker.submit(order)
        fills = broker.on_bar(datetime.now(), "X", _bar(100, 101, 93, 94))
        assert len(fills) == 1
        assert fills[0].fill_price == pytest.approx(95.0)

    def test_stop_stays_pending_if_not_hit(self, broker):
        order = Order(symbol="X", quantity=10, side=OrderSide.BUY,
                       order_type=OrderType.STOP, stop_price=110.0)
        broker.submit(order)
        fills = broker.on_bar(datetime.now(), "X", _bar(100, 102, 98, 101))
        assert len(fills) == 0
        assert order.is_active


class TestStopLimit:
    def test_stop_limit_triggers_and_fills_same_bar(self, broker):
        order = Order(symbol="X", quantity=10, side=OrderSide.BUY,
                       order_type=OrderType.STOP_LIMIT,
                       stop_price=102.0, limit_price=103.5)
        broker.submit(order)
        # Bar goes from 100 up to 104 — stop triggers at 102, limit of 103.5 reachable
        fills = broker.on_bar(datetime.now(), "X", _bar(100, 104, 99, 103))
        assert len(fills) == 1
        # Fill at min(limit, open) = min(103.5, 100) → but stop-limit post-trigger
        # behaves as a limit; limit 103.5 >= open 100 so open-favorable: 100
        assert fills[0].fill_price == pytest.approx(100.0)

    def test_stop_limit_pending_if_stop_triggered_but_limit_not_reached(self, broker):
        # Stop at 102, limit at 101 (tight spread above stop) — if price goes
        # to 102.01 then keeps going to 105, the limit buy at 101 won't fill
        # within this bar because price > 101 after trigger.
        order = Order(symbol="X", quantity=10, side=OrderSide.BUY,
                       order_type=OrderType.STOP_LIMIT,
                       stop_price=102.0, limit_price=101.0)
        broker.submit(order)
        fills = broker.on_bar(datetime.now(), "X", _bar(o=103, h=105, l=102.5, c=104))
        # Open above limit AND low above limit → no fill; order remains (converted to LIMIT)
        assert len(fills) == 0
        assert order.is_active
        assert order.order_type == OrderType.LIMIT


class TestCancel:
    def test_cancel_removes_from_pending(self, broker):
        order = Order(symbol="X", quantity=10, side=OrderSide.BUY,
                       order_type=OrderType.LIMIT, limit_price=90.0)
        broker.submit(order)
        assert broker.cancel(order.order_id) is True
        # Cancelled orders should not fill on subsequent bars
        fills = broker.on_bar(datetime.now(), "X", _bar(100, 102, 89, 90))
        assert len(fills) == 0
        assert order.status == OrderStatus.CANCELLED

    def test_cancel_unknown_returns_false(self, broker):
        assert broker.cancel("nope") is False
