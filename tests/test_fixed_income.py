"""Tests for fixed-income analytics."""
from __future__ import annotations

import numpy as np
import pytest

from quantforge.fixed_income.bond import (
    bond_price, bond_ytm, macaulay_duration, modified_duration, convexity, dv01, bond_cashflows,
)
from quantforge.fixed_income.yield_curve import (
    NelsonSiegel, NelsonSiegelSvensson, discount_factor, forward_rate, bootstrap_zero_curve,
)


class TestBondPricing:
    def test_par_bond_prices_to_face(self):
        p = bond_price(100, 0.05, 0.05, 10, 2)
        assert abs(p - 100.0) < 1e-6

    def test_zero_coupon_equals_discount(self):
        p = bond_price(100, 0.0, 0.04, 5, 2)
        expected = 100 / (1.02) ** 10
        assert abs(p - expected) < 1e-6

    def test_ytm_round_trip(self):
        p = bond_price(100, 0.05, 0.037, 10, 2)
        ytm = bond_ytm(p, 100, 0.05, 10, 2)
        assert abs(ytm - 0.037) < 1e-6

    def test_price_below_par_when_ytm_above_coupon(self):
        p = bond_price(100, 0.04, 0.06, 5, 2)
        assert p < 100

    def test_price_above_par_when_ytm_below_coupon(self):
        p = bond_price(100, 0.06, 0.04, 5, 2)
        assert p > 100

    def test_cashflows_structure(self):
        times, amounts = bond_cashflows(100, 0.04, 2.0, 2)
        assert len(times) == 4
        assert np.allclose(times, [0.5, 1.0, 1.5, 2.0])
        assert amounts[-1] == pytest.approx(102)  # last has face
        assert amounts[0] == pytest.approx(2.0)


class TestDuration:
    def test_duration_positive(self):
        d = macaulay_duration(100, 0.05, 0.04, 10, 2)
        assert d > 0 and d < 10

    def test_modified_less_than_macaulay(self):
        mac = macaulay_duration(100, 0.05, 0.04, 10, 2)
        mod = modified_duration(100, 0.05, 0.04, 10, 2)
        assert mod < mac
        assert abs(mod - mac / 1.02) < 1e-10

    def test_duration_hedge_approximation(self):
        p0 = bond_price(100, 0.05, 0.04, 10, 2)
        md = modified_duration(100, 0.05, 0.04, 10, 2)
        shock = 0.0001
        p1 = bond_price(100, 0.05, 0.04 + shock, 10, 2)
        linear_pred = p0 * (1 - md * shock)
        assert abs(p1 - linear_pred) < 1e-4

    def test_zero_coupon_duration_equals_maturity(self):
        d = macaulay_duration(100, 0.0, 0.04, 5, 2)
        assert abs(d - 5.0) < 1e-6

    def test_convexity_positive(self):
        c = convexity(100, 0.05, 0.04, 10, 2)
        assert c > 0

    def test_dv01_positive(self):
        dv = dv01(100, 0.05, 0.04, 10, 2)
        assert dv > 0


class TestYieldCurve:
    def test_nelson_siegel_fit(self):
        mats = np.array([1, 2, 5, 10, 20, 30])
        yields = np.array([0.03, 0.035, 0.04, 0.045, 0.046, 0.045])
        ns = NelsonSiegel().fit(mats, yields)
        preds = ns(mats)
        rmse = float(np.sqrt(np.mean((preds - yields) ** 2)))
        assert rmse < 0.01

    def test_nelson_siegel_svensson_fit(self):
        mats = np.array([1, 2, 5, 10, 20, 30])
        yields = np.array([0.03, 0.035, 0.04, 0.045, 0.046, 0.045])
        nss = NelsonSiegelSvensson().fit(mats, yields)
        preds = np.array([float(nss(m)) for m in mats])
        rmse = float(np.sqrt(np.mean((preds - yields) ** 2)))
        assert rmse < 0.01

    def test_discount_factor_decreases(self):
        d1 = discount_factor(0.04, 1.0)
        d2 = discount_factor(0.04, 5.0)
        assert d2 < d1 < 1.0

    def test_forward_rate_interior(self):
        f = forward_rate(0.03, 1.0, 0.04, 2.0)
        assert abs(f - 0.05) < 1e-9

    def test_bootstrap_zero_curve(self):
        # pricing bonds at 4% rate
        r = 0.04
        p1 = 100 * 0.02 / 2 / (1 + r / 2) + (100 * 0.02 / 2 + 100) / (1 + r / 2) ** 2
        bonds = [(1.0, 0.02, p1)]
        zeros = bootstrap_zero_curve(bonds)
        assert len(zeros) == 1
        assert zeros[0][0] == 1.0
