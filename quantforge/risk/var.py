"""Value-at-Risk and CVaR (Expected Shortfall)."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis


def _arr(r) -> np.ndarray:
    if isinstance(r, pd.Series):
        return r.dropna().values
    return np.asarray(r)


def historical_var(returns, confidence: float = 0.95, horizon: int = 1) -> float:
    """Returns a POSITIVE number representing the loss threshold."""
    r = _arr(returns)
    if len(r) == 0:
        return np.nan
    q = np.quantile(r, 1 - confidence)
    return float(-q * np.sqrt(horizon))


def historical_cvar(returns, confidence: float = 0.95, horizon: int = 1) -> float:
    r = _arr(returns)
    if len(r) == 0:
        return np.nan
    q = np.quantile(r, 1 - confidence)
    tail = r[r <= q]
    if len(tail) == 0:
        return float(-q * np.sqrt(horizon))
    return float(-tail.mean() * np.sqrt(horizon))


def parametric_var(returns, confidence: float = 0.95, horizon: int = 1) -> float:
    """Gaussian / variance-covariance VaR."""
    r = _arr(returns)
    if len(r) < 2:
        return np.nan
    mu, sigma = float(np.mean(r)), float(np.std(r, ddof=1))
    z = norm.ppf(1 - confidence)
    return float(-(mu + z * sigma) * np.sqrt(horizon))


def parametric_cvar(returns, confidence: float = 0.95, horizon: int = 1) -> float:
    r = _arr(returns)
    if len(r) < 2:
        return np.nan
    mu, sigma = float(np.mean(r)), float(np.std(r, ddof=1))
    z = norm.ppf(1 - confidence)
    es = mu - sigma * norm.pdf(z) / (1 - confidence)
    return float(-es * np.sqrt(horizon))


def cornish_fisher_var(returns, confidence: float = 0.95) -> float:
    """VaR with skew/kurtosis Cornish-Fisher expansion."""
    r = _arr(returns)
    if len(r) < 4:
        return np.nan
    mu, sigma = float(np.mean(r)), float(np.std(r, ddof=1))
    s = float(skew(r))
    k = float(kurtosis(r, fisher=True))
    z = norm.ppf(1 - confidence)
    z_cf = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3 * z) * k / 24
        - (2 * z**3 - 5 * z) * s**2 / 36
    )
    return float(-(mu + z_cf * sigma))


def monte_carlo_var(
    mu: float, sigma: float,
    confidence: float = 0.95, horizon: int = 1,
    n_sims: int = 100_000, seed: int | None = None,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    sims = rng.normal(mu * horizon, sigma * np.sqrt(horizon), size=n_sims)
    var = -np.quantile(sims, 1 - confidence)
    tail = sims[sims <= -var]
    cvar = -tail.mean() if len(tail) > 0 else var
    return float(var), float(cvar)
