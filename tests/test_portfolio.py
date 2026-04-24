"""Tests for quantforge.portfolio: Markowitz, HRP, ERC."""
import numpy as np
import pandas as pd
import pytest

from quantforge.portfolio.hrp import hierarchical_risk_parity
from quantforge.portfolio.markowitz import (
    efficient_frontier,
    max_sharpe,
    mean_variance,
    min_variance,
)
from quantforge.portfolio.risk_parity import equal_risk_contribution

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_cov():
    """4-asset covariance matrix (positive definite, annualized)."""
    rng = np.random.default_rng(0)
    # Build via random factor model for positive definiteness
    F = rng.standard_normal((4, 4))
    C = F @ F.T / 4 + np.eye(4) * 0.01
    return C


@pytest.fixture(scope="module")
def sample_mu():
    return np.array([0.08, 0.12, 0.10, 0.06])


@pytest.fixture(scope="module")
def cov_df(sample_cov):
    cols = ["A", "B", "C", "D"]
    return pd.DataFrame(sample_cov, index=cols, columns=cols)


@pytest.fixture(scope="module")
def mu_series(sample_mu):
    return pd.Series(sample_mu, index=["A", "B", "C", "D"])


# ---------------------------------------------------------------------------
# Min Variance
# ---------------------------------------------------------------------------

class TestMinVariance:
    def test_weights_sum_to_one(self, sample_cov):
        w = min_variance(sample_cov)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_weights_in_unit_interval(self, sample_cov):
        w = min_variance(sample_cov)
        assert (w >= -1e-8).all()
        assert (w <= 1.0 + 1e-8).all()

    def test_min_variance_beats_equal_weight(self, sample_cov):
        w_mv = min_variance(sample_cov)
        n = sample_cov.shape[0]
        w_ew = np.full(n, 1.0 / n)
        var_mv = float(w_mv @ sample_cov @ w_mv)
        var_ew = float(w_ew @ sample_cov @ w_ew)
        assert var_mv <= var_ew + 1e-10

    def test_min_variance_with_dataframe_input(self, cov_df):
        w = min_variance(cov_df)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_min_variance_returns_correct_length(self, sample_cov):
        w = min_variance(sample_cov)
        assert len(w) == sample_cov.shape[0]


# ---------------------------------------------------------------------------
# Max Sharpe
# ---------------------------------------------------------------------------

class TestMaxSharpe:
    def test_weights_sum_to_one(self, sample_mu, sample_cov):
        w = max_sharpe(sample_mu, sample_cov)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_weights_in_unit_interval(self, sample_mu, sample_cov):
        w = max_sharpe(sample_mu, sample_cov)
        assert (w >= -1e-8).all()
        assert (w <= 1.0 + 1e-8).all()

    def test_max_sharpe_with_series_and_df(self, mu_series, cov_df):
        w = max_sharpe(mu_series, cov_df)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_max_sharpe_achieves_higher_sharpe_than_min_variance(self, sample_mu, sample_cov):
        def sharpe(w, mu, cov):
            ret = float(w @ mu)
            vol = float(np.sqrt(max(1e-14, w @ cov @ w)))
            return ret / vol

        w_ms = max_sharpe(sample_mu, sample_cov)
        w_mv = min_variance(sample_cov)
        sr_ms = sharpe(w_ms, sample_mu, sample_cov)
        sr_mv = sharpe(w_mv, sample_mu, sample_cov)
        # Max Sharpe should achieve at least as high a Sharpe as min variance
        assert sr_ms >= sr_mv - 1e-4

    def test_max_sharpe_returns_correct_length(self, sample_mu, sample_cov):
        w = max_sharpe(sample_mu, sample_cov)
        assert len(w) == len(sample_mu)


# ---------------------------------------------------------------------------
# Mean Variance
# ---------------------------------------------------------------------------

class TestMeanVariance:
    def test_weights_sum_to_one(self, sample_mu, sample_cov):
        w = mean_variance(sample_mu, sample_cov, gamma=1.0)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_high_gamma_is_more_conservative(self, sample_mu, sample_cov):
        # Higher risk aversion -> lower variance portfolio
        w_lo = mean_variance(sample_mu, sample_cov, gamma=0.1)
        w_hi = mean_variance(sample_mu, sample_cov, gamma=10.0)
        var_lo = float(w_lo @ sample_cov @ w_lo)
        var_hi = float(w_hi @ sample_cov @ w_hi)
        assert var_hi <= var_lo + 1e-8


# ---------------------------------------------------------------------------
# HRP
# ---------------------------------------------------------------------------

class TestHRP:
    def test_weights_sum_to_one(self, sample_cov):
        w = hierarchical_risk_parity(sample_cov)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_weights_all_positive(self, sample_cov):
        w = hierarchical_risk_parity(sample_cov)
        assert (w >= 0).all()

    def test_hrp_with_dataframe_input(self, cov_df):
        w = hierarchical_risk_parity(cov_df)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert len(w) == len(cov_df)

    def test_hrp_returns_correct_length(self, sample_cov):
        w = hierarchical_risk_parity(sample_cov)
        assert len(w) == sample_cov.shape[0]


# ---------------------------------------------------------------------------
# Equal Risk Contribution
# ---------------------------------------------------------------------------

class TestERC:
    def test_weights_sum_to_one(self, sample_cov):
        w = equal_risk_contribution(sample_cov)
        assert w.sum() == pytest.approx(1.0, abs=1e-5)

    def test_weights_positive(self, sample_cov):
        w = equal_risk_contribution(sample_cov)
        assert (w > 0).all()

    def test_risk_contributions_approximately_equal(self, sample_cov):
        w = equal_risk_contribution(sample_cov)
        vol = float(np.sqrt(max(1e-14, w @ sample_cov @ w)))
        mrc = sample_cov @ w / vol
        rc = w * mrc
        rc_pct = rc / rc.sum()
        expected = 1.0 / len(w)
        # Each risk contribution should be close to 1/n
        for ri in rc_pct:
            assert abs(ri - expected) < 0.05

    def test_erc_nonzero_contributions(self, sample_cov):
        w = equal_risk_contribution(sample_cov)
        vol = float(np.sqrt(max(1e-14, w @ sample_cov @ w)))
        mrc = sample_cov @ w / vol
        rc = w * mrc
        assert (rc > 0).all()


# ---------------------------------------------------------------------------
# Efficient Frontier
# ---------------------------------------------------------------------------

class TestEfficientFrontier:
    def test_efficient_frontier_returns_dataframe(self, sample_mu, sample_cov):
        ef = efficient_frontier(sample_mu, sample_cov, n_points=10)
        assert isinstance(ef, pd.DataFrame)

    def test_efficient_frontier_has_expected_columns(self, sample_mu, sample_cov):
        ef = efficient_frontier(sample_mu, sample_cov, n_points=10)
        assert "risk" in ef.columns
        assert "return" in ef.columns
