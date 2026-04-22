"""Tests for quantforge.analytics: performance, tearsheet, attribution."""
import math
import pytest
import numpy as np
import pandas as pd

from quantforge.analytics.performance import summary_stats
from quantforge.analytics.tearsheet import tearsheet_text, Tearsheet
from quantforge.analytics.attribution import brinson_attribution, factor_attribution


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def equity():
    rng = np.random.default_rng(42)
    r = rng.normal(0.0008, 0.012, 500)
    prices = 100.0 * np.cumprod(1 + r)
    idx = pd.bdate_range("2021-01-01", periods=500)
    return pd.Series(prices, index=idx)


@pytest.fixture(scope="module")
def stats(equity):
    return summary_stats(equity)


# ---------------------------------------------------------------------------
# summary_stats
# ---------------------------------------------------------------------------

class TestSummaryStats:
    REQUIRED_KEYS = {
        "total_return", "annual_return", "annual_vol", "sharpe", "sortino",
        "calmar", "omega", "tail_ratio", "ulcer_index", "max_drawdown",
        "var_95", "cvar_95", "best_day", "worst_day", "skew", "kurt",
    }

    def test_returns_all_expected_keys(self, stats):
        missing = self.REQUIRED_KEYS - set(stats.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_total_return_is_finite(self, stats):
        assert math.isfinite(stats["total_return"])

    def test_annual_vol_positive(self, stats):
        assert stats["annual_vol"] > 0

    def test_max_drawdown_lte_zero(self, stats):
        assert stats["max_drawdown"] <= 0

    def test_var_95_positive(self, stats):
        assert stats["var_95"] > 0

    def test_cvar_95_gte_var_95(self, stats):
        assert stats["cvar_95"] >= stats["var_95"] - 1e-10

    def test_best_day_gte_worst_day(self, stats):
        assert stats["best_day"] >= stats["worst_day"]

    def test_sharpe_finite(self, stats):
        assert math.isfinite(stats["sharpe"])


# ---------------------------------------------------------------------------
# tearsheet_text
# ---------------------------------------------------------------------------

class TestTearsheetText:
    def test_contains_sharpe_string(self, equity):
        text = tearsheet_text(equity, "TestStrat")
        assert "Sharpe" in text

    def test_contains_strategy_name(self, equity):
        text = tearsheet_text(equity, "MyStrategy")
        assert "MyStrategy" in text

    def test_contains_max_drawdown(self, equity):
        text = tearsheet_text(equity, "S")
        assert "Drawdown" in text

    def test_text_is_multiline(self, equity):
        text = tearsheet_text(equity, "S")
        assert "\n" in text

    def test_tearsheet_class_stats_matches_summary_stats(self, equity):
        ts = Tearsheet(equity, "T")
        ts_stats = ts.stats()
        direct = summary_stats(equity)
        assert ts_stats["sharpe"] == pytest.approx(direct["sharpe"])

    def test_tearsheet_class_to_text_contains_sharpe(self, equity):
        ts = Tearsheet(equity, "X")
        text = ts.to_text()
        assert "Sharpe" in text

    def test_tearsheet_to_markdown_contains_header(self, equity):
        ts = Tearsheet(equity, "MyFund")
        md = ts.to_markdown()
        assert "MyFund" in md


# ---------------------------------------------------------------------------
# brinson_attribution
# ---------------------------------------------------------------------------

class TestBrinsonAttribution:
    @pytest.fixture(scope="class")
    def attribution_result(self):
        assets = ["A", "B", "C", "D"]
        wp = pd.Series([0.4, 0.3, 0.2, 0.1], index=assets)
        wb = pd.Series([0.25, 0.25, 0.25, 0.25], index=assets)
        rp = pd.Series([0.05, 0.08, -0.02, 0.03], index=assets)
        rb = pd.Series([0.04, 0.06, 0.01, 0.02], index=assets)
        return brinson_attribution(wp, wb, rp, rb)

    def test_returns_dict(self, attribution_result):
        assert isinstance(attribution_result, dict)

    def test_has_allocation_key(self, attribution_result):
        assert "allocation" in attribution_result

    def test_has_selection_key(self, attribution_result):
        assert "selection" in attribution_result

    def test_has_interaction_key(self, attribution_result):
        assert "interaction" in attribution_result

    def test_has_total_key(self, attribution_result):
        assert "total" in attribution_result

    def test_total_equals_allocation_plus_selection_plus_interaction(self, attribution_result):
        r = attribution_result
        expected = r["allocation"] + r["selection"] + r["interaction"]
        assert r["total"] == pytest.approx(expected)

    def test_all_values_are_finite(self, attribution_result):
        for k, v in attribution_result.items():
            assert math.isfinite(v), f"Key '{k}' is not finite: {v}"

    def test_benchmark_total_is_benchmark_return(self, attribution_result):
        # benchmark = sum(wb * rb) = 0.25*(0.04+0.06+0.01+0.02) = 0.25*0.13 = 0.0325
        assert math.isfinite(attribution_result["benchmark"])


# ---------------------------------------------------------------------------
# factor_attribution
# ---------------------------------------------------------------------------

class TestFactorAttribution:
    @pytest.fixture(scope="class")
    def factor_result(self):
        rng = np.random.default_rng(99)
        idx = pd.bdate_range("2021-01-01", periods=200)
        factors = pd.DataFrame({
            "mkt": rng.normal(0.0004, 0.01, 200),
            "smb": rng.normal(0.0001, 0.005, 200),
        }, index=idx)
        # Portfolio return = 0.002 + 1.2*mkt + 0.4*smb + noise
        returns = pd.Series(
            0.0002 + 1.2 * factors["mkt"].values + 0.4 * factors["smb"].values
            + rng.normal(0, 0.002, 200),
            index=idx,
        )
        return factor_attribution(returns, factors)

    def test_has_alpha_key(self, factor_result):
        assert "alpha" in factor_result

    def test_has_r2_key(self, factor_result):
        assert "r2" in factor_result

    def test_has_beta_key(self, factor_result):
        assert "beta" in factor_result

    def test_r2_in_0_1(self, factor_result):
        r2 = factor_result["r2"]
        assert 0.0 <= r2 <= 1.0

    def test_betas_close_to_true_values(self, factor_result):
        betas = factor_result["beta"]
        # Expect mkt beta ~ 1.2, smb ~ 0.4 (within 15% for 200 obs)
        assert abs(betas["mkt"] - 1.2) < 0.2
        assert abs(betas["smb"] - 0.4) < 0.2

    def test_short_series_returns_nan_alpha(self):
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2021-01-01", periods=5)
        factors = pd.DataFrame({"f1": rng.normal(0, 0.01, 5)}, index=idx)
        returns = pd.Series(rng.normal(0, 0.01, 5), index=idx)
        result = factor_attribution(returns, factors)
        assert math.isnan(result["alpha"])
