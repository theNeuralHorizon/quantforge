"""Tests for GARCH, cointegration, shrinkage, regime detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantforge.data.synthetic import generate_correlated_returns, generate_gbm
from quantforge.stats.cointegration import (
    engle_granger, half_life_of_mean_reversion, johansen_trace,
    rolling_cointegration,
)
from quantforge.stats.garch import garch11_fit, garch11_forecast
from quantforge.stats.regime import detect_structural_breaks, markov_switching_returns
from quantforge.stats.shrinkage import (
    constant_correlation_target, ledoit_wolf_shrinkage, oracle_shrinkage,
    shrunk_covariance,
)


class TestGARCH:
    def test_fit_recovers_params_approximately(self):
        # Simulate GARCH(1,1) with known parameters
        rng = np.random.default_rng(42)
        n = 3000
        omega_true, alpha_true, beta_true = 1e-5, 0.08, 0.90
        sigma2 = np.empty(n)
        r = np.empty(n)
        sigma2[0] = omega_true / (1 - alpha_true - beta_true)
        for t in range(n):
            if t > 0:
                sigma2[t] = omega_true + alpha_true * r[t-1]**2 + beta_true * sigma2[t-1]
            r[t] = np.sqrt(sigma2[t]) * rng.standard_normal()

        params = garch11_fit(r)
        # Parameter recovery with noise — wide tolerances
        assert params.alpha > 0.02 and params.alpha < 0.25
        assert params.beta > 0.70
        assert params.persistence() < 0.9999
        assert np.isfinite(params.log_likelihood)

    def test_forecast_converges_to_unconditional(self):
        r = generate_gbm(n=500, sigma=0.25, seed=7).pct_change().dropna().values
        params = garch11_fit(r)
        fc = garch11_forecast(r, params, horizon=300)
        # Long-horizon forecast should converge toward unconditional variance
        uv = params.unconditional_variance
        assert abs(fc[-1] / uv - 1) < 0.02


class TestCointegration:
    def test_engle_granger_cointegrated(self):
        # build A = 0.8*B + stationary noise
        rng = np.random.default_rng(1)
        b = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 500)))
        noise = np.zeros(500)
        phi = 0.8
        for t in range(1, 500):
            noise[t] = phi * noise[t-1] + rng.normal(0, 0.4)
        a = 0.8 * b + noise + 20
        idx = pd.bdate_range(end="2024-01-01", periods=500)
        A = pd.Series(a, index=idx)
        B = pd.Series(b, index=idx)
        r = engle_granger(A, B)
        assert r.is_cointegrated_95
        assert r.beta is not None

    def test_engle_granger_non_cointegrated(self):
        rng = np.random.default_rng(2)
        a = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
        b = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
        idx = pd.bdate_range(end="2024-01-01", periods=300)
        r = engle_granger(pd.Series(a, index=idx), pd.Series(b, index=idx))
        # Two unrelated random walks: should not be cointegrated at 5%
        # (this can fail rarely by chance — seed chosen to be stable)
        assert not r.is_cointegrated_95

    def test_half_life_of_stationary_series(self):
        rng = np.random.default_rng(3)
        n = 500
        x = np.zeros(n)
        phi = 0.9  # true half-life = -ln(2)/ln(0.9) ≈ 6.58
        for t in range(1, n):
            x[t] = phi * x[t-1] + rng.normal(0, 1)
        hl = half_life_of_mean_reversion(pd.Series(x))
        assert 4 < hl < 12

    def test_johansen_returns_rows(self):
        data = generate_correlated_returns(n=300, symbols=("A", "B", "C"), seed=5).cumsum()
        rows = johansen_trace(data)
        assert len(rows) == 3
        assert all("trace_stat" in r for r in rows)

    def test_rolling_cointegration_shape(self):
        rng = np.random.default_rng(10)
        n = 800
        b = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
        a = 0.8 * b + np.cumsum(rng.normal(0, 0.3, n)) * 0.1
        idx = pd.bdate_range(end="2024-01-01", periods=n)
        df = rolling_cointegration(pd.Series(a, index=idx), pd.Series(b, index=idx),
                                    window=252, step=21)
        assert len(df) > 5
        assert {"statistic", "pvalue", "beta"}.issubset(df.columns)


class TestShrinkage:
    def test_ledoit_wolf_produces_valid_cov(self):
        rets = generate_correlated_returns(n=200, symbols=("A", "B", "C", "D"), seed=7)
        cov, shrinkage = ledoit_wolf_shrinkage(rets)
        assert cov.shape == (4, 4)
        assert 0.0 <= shrinkage <= 1.0
        # positive semi-definite
        eigvals = np.linalg.eigvalsh(cov)
        assert (eigvals >= -1e-9).all()

    def test_oracle_shrinkage(self):
        rets = generate_correlated_returns(n=150, symbols=("A", "B", "C"), seed=11)
        cov, rho = oracle_shrinkage(rets)
        assert cov.shape == (3, 3)
        assert 0.0 <= rho <= 1.0

    def test_shrunk_covariance_close_to_sample_with_large_n(self):
        rets = generate_correlated_returns(n=5000, symbols=("A", "B", "C"), seed=19)
        sample = rets.cov().values
        shrunk = shrunk_covariance(rets, "ledoit_wolf")
        # With large n, shrunk should be close to sample (low optimal shrinkage)
        diff = np.abs(shrunk - sample).max()
        assert diff < 0.001

    def test_constant_correlation_target_preserves_diagonal(self):
        rets = generate_correlated_returns(n=200, symbols=("A", "B", "C"), seed=1)
        sample = rets.cov().values
        target = constant_correlation_target(sample)
        assert np.allclose(np.diag(target), np.diag(sample))


class TestRegime:
    def test_markov_switching_produces_states(self):
        # Two-regime simulation: low-vol / high-vol
        rng = np.random.default_rng(42)
        n = 500
        rets = np.concatenate([
            rng.normal(0.0005, 0.008, n // 2),
            rng.normal(-0.0002, 0.025, n // 2),
        ])
        df = markov_switching_returns(pd.Series(rets), n_states=2)
        assert len(df) == n
        assert set(df["state"].unique()).issubset({0, 1})
        # posterior probabilities sum to 1
        assert np.allclose(df[["p_state_0", "p_state_1"]].sum(axis=1), 1.0)

    def test_structural_breaks_empty_for_random_walk(self):
        rng = np.random.default_rng(1)
        s = pd.Series(rng.normal(0, 1, 500))
        breaks = detect_structural_breaks(s, window=63, threshold=5.0)
        # Very high threshold → should find at most a handful
        assert len(breaks) < 10
