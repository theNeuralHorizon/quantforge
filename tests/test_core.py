"""Tests for quantforge.core: Position fill math and Portfolio accounting."""
import math
import pytest
from datetime import datetime

from quantforge.core.position import Position
from quantforge.core.portfolio import Portfolio


# ---------------------------------------------------------------------------
# Position fill math
# ---------------------------------------------------------------------------

class TestPositionFillLong:
    def test_buy_from_flat_sets_quantity_and_avg_cost(self):
        pos = Position("AAPL")
        pos.apply_fill(100, 50.0)
        assert pos.quantity == pytest.approx(100.0)
        assert pos.avg_cost == pytest.approx(50.0)

    def test_buy_returns_correct_cash_delta(self):
        pos = Position("AAPL")
        delta = pos.apply_fill(100, 50.0)
        # cash_delta = -qty * price - commission
        assert delta == pytest.approx(-5000.0)

    def test_buy_with_commission_reduces_cash_more(self):
        pos = Position("AAPL")
        delta = pos.apply_fill(100, 50.0, commission=10.0)
        assert delta == pytest.approx(-5010.0)
        assert pos.total_commission == pytest.approx(10.0)

    def test_adding_to_long_updates_avg_cost(self):
        pos = Position("AAPL")
        pos.apply_fill(100, 50.0)
        pos.apply_fill(100, 60.0)
        assert pos.quantity == pytest.approx(200.0)
        assert pos.avg_cost == pytest.approx(55.0)

    def test_partial_close_long_realizes_pnl(self):
        pos = Position("AAPL")
        pos.apply_fill(100, 50.0)   # buy 100 @ 50
        pos.apply_fill(-50, 60.0)   # sell 50 @ 60 => realized = 50 * (60-50) = 500
        assert pos.quantity == pytest.approx(50.0)
        assert pos.realized_pnl == pytest.approx(500.0)

    def test_full_close_long_flattens_position(self):
        pos = Position("AAPL")
        pos.apply_fill(100, 50.0)
        pos.apply_fill(-100, 55.0)
        assert pos.is_flat
        assert pos.realized_pnl == pytest.approx(500.0)

    def test_unrealized_pnl_marks_correctly(self):
        pos = Position("AAPL")
        pos.apply_fill(100, 50.0)
        pos.mark_to_market(55.0)
        assert pos.unrealized_pnl == pytest.approx(500.0)

    def test_market_value_is_qty_times_last_price(self):
        pos = Position("AAPL")
        pos.apply_fill(100, 50.0)
        pos.mark_to_market(60.0)
        assert pos.market_value == pytest.approx(6000.0)


class TestPositionFillShort:
    def test_sell_short_from_flat(self):
        pos = Position("TSLA")
        delta = pos.apply_fill(-100, 200.0)
        assert pos.quantity == pytest.approx(-100.0)
        assert pos.avg_cost == pytest.approx(200.0)
        # cash_delta = -(-100) * 200 = +20000
        assert delta == pytest.approx(20000.0)

    def test_cover_short_realizes_pnl(self):
        pos = Position("TSLA")
        pos.apply_fill(-100, 200.0)   # short 100 @ 200
        pos.apply_fill(100, 180.0)    # cover @ 180 => profit = 100 * (200-180) = 2000
        assert pos.is_flat
        assert pos.realized_pnl == pytest.approx(2000.0)

    def test_short_unrealized_pnl(self):
        pos = Position("TSLA")
        pos.apply_fill(-100, 200.0)
        pos.mark_to_market(180.0)
        # unrealized = (last - avg_cost) * qty = (180 - 200) * -100 = 2000
        assert pos.unrealized_pnl == pytest.approx(2000.0)


class TestPositionFlip:
    def test_flip_long_to_short_sets_new_avg_cost(self):
        pos = Position("SPY")
        pos.apply_fill(100, 400.0)    # long 100
        pos.apply_fill(-150, 410.0)   # sell 150: close 100, short 50
        assert pos.quantity == pytest.approx(-50.0)
        # after flip, avg_cost = price of the flip fill
        assert pos.avg_cost == pytest.approx(410.0)
        # realized pnl from closing the 100 long
        assert pos.realized_pnl == pytest.approx(100 * (410.0 - 400.0))

    def test_flip_short_to_long(self):
        pos = Position("SPY")
        pos.apply_fill(-100, 400.0)   # short 100
        pos.apply_fill(150, 390.0)    # buy 150: cover 100, long 50
        assert pos.quantity == pytest.approx(50.0)
        assert pos.avg_cost == pytest.approx(390.0)
        # realized pnl: 100 * (400 - 390) = 1000
        assert pos.realized_pnl == pytest.approx(1000.0)


class TestPositionProperties:
    def test_is_long(self):
        pos = Position("X")
        pos.apply_fill(1, 10.0)
        assert pos.is_long
        assert not pos.is_short
        assert not pos.is_flat

    def test_is_short(self):
        pos = Position("X")
        pos.apply_fill(-1, 10.0)
        assert pos.is_short
        assert not pos.is_long
        assert not pos.is_flat

    def test_is_flat_at_start(self):
        pos = Position("X")
        assert pos.is_flat

    def test_total_pnl_subtracts_commission(self):
        pos = Position("X")
        pos.apply_fill(100, 10.0, commission=5.0)
        pos.mark_to_market(10.0)
        # unrealized = 0, realized = 0, commission = 5 => total_pnl = -5
        assert pos.total_pnl == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# Portfolio cash accounting and equity
# ---------------------------------------------------------------------------

class TestPortfolioCash:
    def test_initial_cash_equals_capital(self):
        p = Portfolio(initial_capital=100_000)
        assert p.cash == pytest.approx(100_000.0)

    def test_buy_reduces_cash(self):
        p = Portfolio(initial_capital=100_000)
        p.apply_fill("AAPL", 100, 500.0)
        assert p.cash == pytest.approx(50_000.0)

    def test_sell_increases_cash(self):
        p = Portfolio(initial_capital=100_000)
        p.apply_fill("AAPL", 100, 500.0)
        p.apply_fill("AAPL", -100, 550.0)
        # started with 100k, bought for 50k, sold for 55k => 105k
        assert p.cash == pytest.approx(105_000.0)

    def test_equity_equals_cash_plus_market_value(self):
        p = Portfolio(initial_capital=100_000)
        p.apply_fill("AAPL", 100, 400.0)
        p.mark_to_market({"AAPL": 420.0})
        expected_mv = 100 * 420.0
        expected_cash = 100_000 - 100 * 400.0
        assert p.equity == pytest.approx(expected_cash + expected_mv)

    def test_equity_unchanged_when_price_at_cost(self):
        p = Portfolio(initial_capital=100_000)
        p.apply_fill("AAPL", 100, 500.0)
        p.mark_to_market({"AAPL": 500.0})
        assert p.equity == pytest.approx(100_000.0)

    def test_record_equity_appends_to_curve(self):
        p = Portfolio(initial_capital=50_000)
        ts = datetime(2024, 1, 1)
        p.record_equity(ts)
        assert len(p.equity_curve) == 1
        assert p.equity_curve[0][1] == pytest.approx(50_000.0)

    def test_multiple_positions_market_value(self):
        p = Portfolio(initial_capital=200_000)
        p.apply_fill("AAPL", 100, 150.0)
        p.apply_fill("GOOG", 10, 2500.0)
        p.mark_to_market({"AAPL": 160.0, "GOOG": 2600.0})
        mv = 100 * 160.0 + 10 * 2600.0
        assert p.market_value == pytest.approx(mv)

    def test_gross_exposure_counts_short_absolute(self):
        p = Portfolio(initial_capital=100_000)
        p.apply_fill("A", 100, 10.0)   # long
        p.apply_fill("B", -50, 20.0)   # short
        p.mark_to_market({"A": 10.0, "B": 20.0})
        # gross = |100*10| + |-50*20| = 1000 + 1000 = 2000
        assert p.gross_exposure == pytest.approx(2000.0)

    def test_position_created_on_demand(self):
        p = Portfolio()
        pos = p.position("NEW")
        assert pos.symbol == "NEW"
        assert pos.is_flat
