"""Tests for execution algos + market impact."""
from __future__ import annotations

import numpy as np
import pytest

from quantforge.data.synthetic import generate_ohlcv
from quantforge.execution.algos import (
    implementation_shortfall,
    pov,
    twap,
    vwap,
)
from quantforge.execution.impact import (
    almgren_chriss_schedule,
    linear_impact,
    square_root_impact,
)


@pytest.fixture
def bars():
    return generate_ohlcv(n=50, s0=100, mu=0.0, sigma=0.2, seed=42)


class TestTWAP:
    def test_equal_allocation(self, bars):
        rep = twap(bars, 10_000, "buy")
        assert abs(rep.schedule["qty"].sum() - 10_000) < 1e-6
        # all bars should have equal size
        per_bar = rep.schedule["qty"].values
        assert np.allclose(per_bar, per_bar.mean())

    def test_sell_side_negative(self, bars):
        rep = twap(bars, 10_000, "sell")
        assert rep.total_qty == -10_000
        assert rep.schedule["qty"].sum() == -10_000


class TestVWAP:
    def test_proportional_to_volume(self, bars):
        rep = vwap(bars, 10_000, "buy")
        # higher-volume bars should get more qty
        qty = rep.schedule["qty"].values
        vol = bars["volume"].values
        # correlation between qty and volume ~= 1 (perfect, since we allocate proportionally)
        corr = np.corrcoef(qty, vol)[0, 1]
        assert corr > 0.999

    def test_matches_market_vwap_in_limit(self, bars):
        # executing VWAP at proportional volume → avg price close to market VWAP
        rep = vwap(bars, 10_000, "buy")
        assert abs(rep.avg_price - rep.market_vwap) / rep.market_vwap < 0.005

    def test_zero_volume_falls_back_to_twap(self):
        bars = generate_ohlcv(n=10, seed=1)
        bars.loc[:, "volume"] = 0
        rep = vwap(bars, 1_000, "buy")
        # falls back → equal qty per bar
        per_bar = rep.schedule["qty"].values
        assert np.allclose(per_bar, per_bar.mean())


class TestPOV:
    def test_participation_rate(self, bars):
        rate = 0.05
        rep = pov(bars, participation_rate=rate, side="buy")
        # qty = rate * volume
        expected = bars["volume"].values * rate
        assert np.allclose(rep.schedule["qty"].values, expected)

    def test_max_quantity_caps_execution(self, bars):
        rep = pov(bars, participation_rate=0.5, side="buy", max_quantity=100_000)
        assert abs(rep.schedule["qty"].sum() - 100_000) < 1e-6

    def test_invalid_rate(self, bars):
        with pytest.raises(ValueError):
            pov(bars, participation_rate=0.0, side="buy")
        with pytest.raises(ValueError):
            pov(bars, participation_rate=1.5, side="buy")


class TestImplementationShortfall:
    def test_executes_full_quantity(self, bars):
        rep = implementation_shortfall(bars, 10_000, "buy", risk_aversion=1e-5)
        assert abs(rep.schedule["qty"].sum() - 10_000) < 1e-6

    def test_monotone_decreasing_remaining(self, bars):
        # High risk aversion → front-loaded
        rep = implementation_shortfall(bars, 10_000, "buy", risk_aversion=1e-3)
        qty = rep.schedule["qty"].values
        # At least the first bar has positive execution
        assert qty[0] > 0
        # Remaining shares should be monotone non-increasing
        cum = np.cumsum(qty)
        remaining = 10_000 - cum
        assert all(remaining[i+1] <= remaining[i] + 1e-6 for i in range(len(remaining)-1))

    def test_high_risk_aversion_front_loads(self, bars):
        low = implementation_shortfall(bars, 10_000, "buy", risk_aversion=1e-10)
        high = implementation_shortfall(bars, 10_000, "buy", risk_aversion=1e-2)
        # With high risk aversion, 80% of the order should be done in the first
        # quarter of the horizon
        n = len(bars)
        high_frac_done_early = high.schedule["qty"].iloc[:n // 4].sum() / 10_000
        low_frac_done_early = low.schedule["qty"].iloc[:n // 4].sum() / 10_000
        assert high_frac_done_early > low_frac_done_early


class TestImpactModels:
    def test_sqrt_impact_scales_with_sqrt(self):
        i1 = square_root_impact(q=10_000, adv=1_000_000, eta=0.1)
        i4 = square_root_impact(q=40_000, adv=1_000_000, eta=0.1)
        # 4x quantity → ~2x impact (sqrt relationship)
        assert abs(i4 / i1 - 2.0) < 0.01

    def test_linear_impact_scales(self):
        assert linear_impact(1_000, 1e-7) == pytest.approx(1e-4)
        assert linear_impact(10_000, 1e-7) == pytest.approx(1e-3)

    def test_ac_schedule_sums_to_target(self):
        trades, remaining = almgren_chriss_schedule(
            X=10_000, T=20, sigma=0.01, eta=1e-5, risk_aversion=1e-6,
        )
        assert len(trades) == 20
        assert len(remaining) == 21
        assert abs(trades.sum() - 10_000) < 1e-6
        assert remaining[0] == 10_000
        assert abs(remaining[-1]) < 1e-6

    def test_ac_zero_impact_dumps_everything_at_start(self):
        trades, _ = almgren_chriss_schedule(X=1_000, T=10, sigma=0.01, eta=0.0)
        assert trades[0] == 1_000
        assert trades[1:].sum() == 0
