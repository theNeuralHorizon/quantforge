"""Tests for Monte Carlo portfolio simulation + Kelly criterion + factor model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantforge.ml.factor_model import factor_regression, simulate_fama_french_factors
from quantforge.portfolio.cvar_opt import minimize_cvar, historical_cvar_from_scenarios
from quantforge.risk.kelly import kelly_fraction, kelly_continuous, kelly_vector, fractional_kelly
from quantforge.risk.simulation import simulate_portfolio, simulate_portfolio_returns


class TestKelly:
    def test_binary_kelly_favorable(self):
        # 60% chance to win 1:1 -> Kelly = 0.2
        assert abs(kelly_fraction(0.6, 1.0) - 0.2) < 1e-10

    def test_binary_kelly_unfavorable(self):
        # 40% chance to win 1:1 -> negative -> clamped to 0
        assert kelly_fraction(0.4, 1.0) == 0.0

    def test_binary_kelly_zero_b(self):
        assert kelly_fraction(0.6, 0.0) == 0.0

    def test_continuous_kelly(self):
        # mu=0.10, var=0.04 -> f = 2.5
        assert abs(kelly_continuous(0.10, 0.04) - 2.5) < 1e-10

    def test_continuous_kelly_zero_var(self):
        assert kelly_continuous(0.10, 0.0) == 0.0

    def test_vector_kelly_shape(self):
        mu = np.array([0.10, 0.08])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        w = kelly_vector(mu, cov)
        assert w.shape == (2,)

    def test_fractional_kelly(self):
        assert fractional_kelly(1.0, 0.5) == 0.5
        assert np.allclose(fractional_kelly(np.array([1.0, 2.0]), 0.5), [0.5, 1.0])


class TestSimulation:
    def test_simulate_portfolio_shape(self):
        sim = simulate_portfolio(100_000, 0.08, 0.2, n_steps=100, n_paths=500, seed=1)
        assert sim.paths.shape == (101, 500)
        assert sim.terminal_wealth.shape == (500,)
        assert sim.max_drawdowns.shape == (500,)

    def test_simulate_portfolio_initial_value(self):
        sim = simulate_portfolio(100_000, 0.08, 0.2, n_steps=100, n_paths=500, seed=1)
        assert np.allclose(sim.paths[0], 100_000)

    def test_drawdowns_non_positive(self):
        sim = simulate_portfolio(100_000, 0.08, 0.3, n_steps=252, n_paths=1000, seed=1)
        assert (sim.max_drawdowns <= 1e-12).all()

    def test_ruin_probability_range(self):
        sim = simulate_portfolio(100_000, 0.08, 0.3, n_steps=252, n_paths=1000, seed=1)
        p = sim.ruin_probability(50_000)
        assert 0 <= p <= 1

    def test_simulate_from_returns(self):
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(rng.normal(0.001, 0.01, (252, 3)), columns=["A", "B", "C"])
        w = np.array([0.4, 0.3, 0.3])
        sim = simulate_portfolio_returns(returns, w, initial=100, n_steps=50, n_paths=200, seed=1)
        assert sim.paths.shape == (51, 200)
        assert np.allclose(sim.paths[0], 100)


class TestCVaROpt:
    def test_cvar_weights_valid(self):
        rng = np.random.default_rng(42)
        sims = rng.multivariate_normal([0.001, 0.001, 0.001], 0.01 * np.eye(3), size=500)
        w = minimize_cvar(sims, alpha=0.95)
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w >= -1e-8).all()
        assert (w <= 1 + 1e-8).all()

    def test_cvar_beats_random_portfolio(self):
        rng = np.random.default_rng(42)
        sims = rng.multivariate_normal([0.001, 0.001, 0.001], 0.01 * np.eye(3), size=500)
        w_opt = minimize_cvar(sims, alpha=0.95)
        w_equal = np.array([1 / 3] * 3)
        c_opt = historical_cvar_from_scenarios(w_opt, sims, 0.95)
        c_eq = historical_cvar_from_scenarios(w_equal, sims, 0.95)
        # Optimizer should be at least as good as equal weight
        assert c_opt <= c_eq + 1e-6


class TestFactorModel:
    def test_simulate_ff_shape(self):
        df = simulate_fama_french_factors(252)
        assert df.shape == (252, 3)
        assert list(df.columns) == ["MKT", "SMB", "HML"]

    def test_factor_regression_recovers_betas(self):
        rng = np.random.default_rng(42)
        factors = simulate_fama_french_factors(500, seed=1)
        true_betas = {"MKT": 1.2, "SMB": 0.3, "HML": -0.4}
        noise = rng.normal(0, 0.005, len(factors))
        r = (factors * pd.Series(true_betas)).sum(axis=1) + 0.0003 + noise
        reg = factor_regression(r, factors)
        for k, v in true_betas.items():
            assert abs(reg.betas[k] - v) < 0.1

    def test_factor_regression_r2_high_with_perfect_fit(self):
        factors = simulate_fama_french_factors(300, seed=3)
        r = factors["MKT"] * 1.0  # perfectly explained by MKT
        reg = factor_regression(r, factors)
        assert reg.r_squared > 0.95
