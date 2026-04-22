"""Kelly criterion sizing."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def kelly_fraction(p: float, b: float, q: float | None = None) -> float:
    """Binary Kelly: f* = (bp - q) / b where p = prob win, b = win/loss ratio, q = 1-p."""
    if q is None:
        q = 1 - p
    if b <= 0:
        return 0.0
    return max(0.0, (b * p - q) / b)


def kelly_continuous(mean_return: float, variance: float) -> float:
    """Continuous Kelly fraction = mu / sigma^2. Often "half Kelly" is used in practice."""
    if variance <= 0:
        return 0.0
    return mean_return / variance


def kelly_vector(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Multi-asset Kelly: w* = Sigma^-1 * mu (unconstrained)."""
    try:
        return np.linalg.solve(cov, mu)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(cov, mu, rcond=None)[0]


def fractional_kelly(f_star: float | np.ndarray, fraction: float = 0.5) -> float | np.ndarray:
    """Practical half-Kelly (or other fraction)."""
    return f_star * fraction


def kelly_vector_capped(
    mu: np.ndarray,
    cov: np.ndarray,
    max_leverage: float = 1.0,
    long_only: bool = False,
) -> np.ndarray:
    """Kelly-optimal weights, clipped to `max_leverage` gross exposure and optionally long-only.

    Returns weights that sum to at most max_leverage in absolute-value terms.
    """
    w = kelly_vector(mu, cov)
    if long_only:
        w = np.maximum(w, 0.0)
    gross = float(np.abs(w).sum())
    if gross > max_leverage and gross > 0:
        w = w * (max_leverage / gross)
    return w
