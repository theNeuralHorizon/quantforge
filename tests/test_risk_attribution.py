"""Tests for portfolio risk attribution."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantforge.data.synthetic import generate_correlated_returns
from quantforge.risk.attribution import (
    cvar_attribution, risk_budget_deviation, var_attribution,
    volatility_attribution,
)


class TestVolAttribution:
    def test_components_sum_to_total(self):
        w = np.array([0.5, 0.3, 0.2])
        cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]])
        r = volatility_attribution(w, cov)
        # Euler: sum(component_i) = portfolio vol
        assert abs(r.component_risk.sum() - r.total_risk) < 1e-10

    def test_pct_contribution_sums_to_one(self):
        w = np.array([0.5, 0.3, 0.2])
        cov = np.diag([0.04, 0.09, 0.16])
        r = volatility_attribution(w, cov)
        assert abs(r.percent_contribution.sum() - 1.0) < 1e-10

    def test_uses_series_index_for_names(self):
        w = pd.Series([0.5, 0.5], index=["A", "B"])
        cov = pd.DataFrame([[0.04, 0.02], [0.02, 0.09]], index=["A", "B"], columns=["A", "B"])
        r = volatility_attribution(w, cov)
        assert list(r.component_risk.index) == ["A", "B"]

    def test_single_asset_contributes_everything(self):
        w = np.array([1.0, 0.0, 0.0])
        cov = np.diag([0.04, 0.09, 0.16])
        r = volatility_attribution(w, cov)
        assert r.percent_contribution.iloc[0] == pytest.approx(1.0, abs=1e-12)
        assert r.percent_contribution.iloc[1] == pytest.approx(0.0, abs=1e-12)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            volatility_attribution(np.array([1, 0]), np.eye(3))


class TestVaRAttribution:
    def test_components_sum_to_var(self):
        rets = generate_correlated_returns(n=1000, symbols=("A", "B", "C"), seed=42)
        w = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
        r = var_attribution(w, rets, confidence=0.95)
        # After rescaling, components should sum exactly to VaR
        assert abs(r.component_risk.sum() - r.total_risk) < 1e-9

    def test_var_positive_for_typical_inputs(self):
        rets = generate_correlated_returns(n=500, seed=7)
        w = np.array([1/3, 1/3, 1/3])
        r = var_attribution(w, rets, confidence=0.95)
        assert r.total_risk > 0

    def test_uses_dataframe_columns_for_names(self):
        rets = generate_correlated_returns(n=300, symbols=("X", "Y"), seed=1)
        w = np.array([0.5, 0.5])
        r = var_attribution(w, rets)
        assert list(r.marginal_risk.index) == ["X", "Y"]


class TestCVaRAttribution:
    def test_cvar_geq_var(self):
        rets = generate_correlated_returns(n=1000, seed=1)
        w = np.array([0.5, 0.25, 0.25])
        v = var_attribution(w, rets, confidence=0.95)
        c = cvar_attribution(w, rets, confidence=0.95)
        # CVaR >= VaR by definition (conditional on being in the tail)
        assert c.total_risk >= v.total_risk - 1e-9

    def test_components_sum_to_cvar(self):
        rets = generate_correlated_returns(n=800, symbols=("A","B","C"), seed=3)
        w = pd.Series([0.4, 0.4, 0.2], index=["A","B","C"])
        r = cvar_attribution(w, rets)
        assert abs(r.component_risk.sum() - r.total_risk) < 1e-9


class TestRiskBudget:
    def test_equal_weight_diversified_budget(self):
        # With uncorrelated equal-vol assets and equal weights, risk budgets
        # should be ~equal → deviation ≈ 0 from uniform target.
        w = np.array([1/3, 1/3, 1/3])
        cov = np.eye(3) * 0.04
        df = risk_budget_deviation(w, cov, target_budget=np.array([1/3, 1/3, 1/3]))
        assert df["deviation"].abs().max() < 1e-10

    def test_concentrated_portfolio_has_big_deviation(self):
        w = np.array([0.95, 0.05])
        cov = np.diag([0.04, 0.04])
        df = risk_budget_deviation(w, cov, target_budget={"asset_0": 0.5, "asset_1": 0.5})
        assert df["deviation"].abs().max() > 0.3
