"""Monte Carlo portfolio simulation: terminal wealth, drawdown distribution, ruin probability."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    paths: np.ndarray  # (n_paths, n_steps+1)
    terminal_wealth: np.ndarray
    max_drawdowns: np.ndarray

    def percentiles(self, pcts=(5, 25, 50, 75, 95)) -> pd.DataFrame:
        out = {
            "terminal_wealth": [np.percentile(self.terminal_wealth, p) for p in pcts],
            "max_drawdown": [np.percentile(self.max_drawdowns, p) for p in pcts],
        }
        return pd.DataFrame(out, index=[f"p{p}" for p in pcts])

    def ruin_probability(self, threshold: float = 0.0) -> float:
        """Fraction of paths that ever hit or go below threshold."""
        below = (self.paths <= threshold).any(axis=1)
        return float(below.mean())


def simulate_portfolio(
    initial: float,
    mu: float,
    sigma: float,
    n_steps: int = 252,
    dt: float = 1 / 252,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> SimulationResult:
    """Simulate GBM wealth paths. Returns SimulationResult with paths, terminal wealth, and drawdowns."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_steps, n_paths))
    drift = (mu - 0.5 * sigma**2) * dt
    diff = sigma * np.sqrt(dt) * z
    log_growth = np.cumsum(drift + diff, axis=0)
    paths = initial * np.exp(log_growth)
    paths = np.vstack([np.full((1, n_paths), initial), paths])

    terminal = paths[-1]
    running_max = np.maximum.accumulate(paths, axis=0)
    drawdowns = (paths / running_max - 1).min(axis=0)
    return SimulationResult(paths=paths, terminal_wealth=terminal, max_drawdowns=drawdowns)


def simulate_portfolio_returns(
    returns: pd.DataFrame | np.ndarray,
    weights: np.ndarray,
    initial: float = 100_000.0,
    n_steps: int = 252,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> SimulationResult:
    """Simulate a portfolio using bootstrapped historical returns.

    `returns` has shape (T, N) with per-period returns for N assets.
    """
    r = returns.values if hasattr(returns, "values") else np.asarray(returns)
    rng = np.random.default_rng(seed)
    T = r.shape[0]
    idx = rng.integers(0, T, size=(n_steps, n_paths))
    port_r = r[idx] @ weights  # (n_steps, n_paths)
    log_growth = np.cumsum(np.log1p(port_r), axis=0)
    paths = initial * np.exp(log_growth)
    paths = np.vstack([np.full((1, n_paths), initial), paths])

    terminal = paths[-1]
    running_max = np.maximum.accumulate(paths, axis=0)
    drawdowns = (paths / running_max - 1).min(axis=0)
    return SimulationResult(paths=paths, terminal_wealth=terminal, max_drawdowns=drawdowns)
