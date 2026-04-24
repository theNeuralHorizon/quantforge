"""Tests for multi-leg options strategies."""
from __future__ import annotations

import math

import numpy as np
import pytest

from quantforge.options.multi_leg import (
    bear_put_spread,
    bull_call_spread,
    butterfly,
    calendar_spread,
    collar,
    iron_condor,
    straddle,
    strangle,
)


class TestStraddle:
    def test_long_straddle_delta_near_zero_atm(self):
        # With r > 0 and q = 0 the risk-neutral drift pushes the call slightly
        # ITM, so ATM straddle delta isn't exactly zero — we just want it small.
        q = straddle(100, 100, 1.0, 0.04, 0.25, direction="long")
        assert abs(q.greeks["delta"]) < 0.3
        assert q.greeks["vega"] > 0  # net long vol

    def test_short_straddle_greeks_flipped(self):
        long_ = straddle(100, 100, 1.0, 0.04, 0.25, direction="long")
        short_ = straddle(100, 100, 1.0, 0.04, 0.25, direction="short")
        for g in ("delta", "gamma", "vega", "theta", "rho"):
            assert math.isclose(long_.greeks[g], -short_.greeks[g], rel_tol=1e-9, abs_tol=1e-9)

    def test_break_evens(self):
        q = straddle(100, 100, 1.0, 0.04, 0.25, direction="long")
        # Long straddle has two break-evens at K ± premium
        assert len(q.break_evens) == 2
        lo, hi = sorted(q.break_evens)
        assert lo < 100 < hi


class TestSpreads:
    def test_bull_call_max_profit(self):
        q = bull_call_spread(100, 95, 105, 1.0, 0.04, 0.2)
        # max profit = (upper - lower) - net debit
        assert q.max_profit is not None and q.max_profit > 0
        # max loss = net debit (negative pnl)
        assert q.max_loss is not None and q.max_loss < 0
        # constraint: max_profit + abs(max_loss) == width
        assert abs(q.max_profit - q.max_loss - 10.0) < 1e-9

    def test_bear_put_mirrors_bull_call_parity_style(self):
        bull = bull_call_spread(100, 95, 105, 1.0, 0.04, 0.2)
        bear = bear_put_spread(100, 95, 105, 1.0, 0.04, 0.2)
        # both capped at width
        assert bull.max_profit - bull.max_loss == pytest.approx(10.0)
        assert bear.max_profit - bear.max_loss == pytest.approx(10.0)

    def test_invalid_strikes(self):
        with pytest.raises(ValueError):
            bull_call_spread(100, 105, 95, 1.0, 0.04, 0.2)


class TestIronCondor:
    def test_iron_condor_net_credit(self):
        q = iron_condor(100, 85, 95, 105, 115, 1.0, 0.04, 0.25)
        # Selling the inner spreads → net credit → negative premium
        assert q.net_premium < 0
        # max_profit = credit received
        assert q.max_profit is not None and q.max_profit > 0
        # max_loss = wing_width - max_profit (negative)
        assert q.max_loss is not None and q.max_loss < 0

    def test_iron_condor_strike_order(self):
        with pytest.raises(ValueError):
            iron_condor(100, 95, 85, 105, 115, 1.0, 0.04, 0.25)


class TestButterfly:
    def test_long_call_butterfly_is_debit(self):
        q = butterfly(100, 90, 100, 110, 1.0, 0.04, 0.25, option="call")
        assert q.net_premium > 0       # debit
        assert q.max_loss is not None and q.max_loss < 0
        assert q.max_profit is not None and q.max_profit > 0

    def test_put_butterfly_symmetric_payoff(self):
        q = butterfly(100, 90, 100, 110, 1.0, 0.04, 0.25, option="put")
        spots, pnl = q.payoff
        # profit peaks near center strike
        peak_idx = int(np.argmax(pnl))
        assert 90 < spots[peak_idx] < 110


class TestCalendarSpread:
    def test_calendar_requires_T_near_less_than_T_far(self):
        with pytest.raises(ValueError):
            calendar_spread(100, 100, 1.0, 0.5, 0.04, 0.2)

    def test_calendar_has_positive_vega(self):
        q = calendar_spread(100, 100, 0.25, 1.0, 0.04, 0.2, option="call")
        # Long far-dated + short near-dated → net long vega
        assert q.greeks["vega"] > 0


class TestCollar:
    def test_collar_caps_upside_and_downside(self):
        q = collar(100, 95, 105, 0.5, 0.04, 0.2)
        assert q.max_profit is not None
        assert q.max_loss is not None
        # Max profit = (call_strike - S) + net_option_credit. With put cheap
        # and call richer, net credit pushes max_profit above width.
        width = 105 - 100
        assert 0 < q.max_profit < width + 10
        assert -10 < q.max_loss < 0


class TestStrangle:
    def test_strangle_break_evens_outside_wings(self):
        q = strangle(100, 95, 105, 1.0, 0.04, 0.25, direction="long")
        # Break-evens should sit below 95 and above 105
        lows = [b for b in q.break_evens if b < 100]
        highs = [b for b in q.break_evens if b > 100]
        assert lows and highs
        assert min(lows) < 95
        assert max(highs) > 105

    def test_strangle_strike_order(self):
        with pytest.raises(ValueError):
            strangle(100, 105, 95, 1.0, 0.04, 0.25, direction="long")
