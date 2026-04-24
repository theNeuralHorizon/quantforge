"""Market impact models: square-root, linear, and the full Almgren-Chriss schedule."""
from __future__ import annotations

import numpy as np


def square_root_impact(q: float, adv: float, eta: float = 0.1) -> float:
    """Bouchaud-style square-root impact: impact ≈ eta * sign(q) * sqrt(|q|/ADV)."""
    if adv <= 0:
        return 0.0
    return float(eta * np.sign(q) * np.sqrt(abs(q) / adv))


def linear_impact(q: float, gamma: float = 1e-7) -> float:
    """Linear impact: impact = gamma * q (shares). Useful for Almgren-Chriss."""
    return float(gamma * q)


def almgren_chriss_schedule(
    X: float, T: int,
    sigma: float, eta: float, gamma: float = 0.0,
    risk_aversion: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (trades_per_step, shares_remaining) for the Almgren-Chriss problem.

    Parameters
    ----------
    X : total shares to execute
    T : number of time steps in the horizon
    sigma : per-step volatility of the asset
    eta : temporary impact coefficient
    gamma : permanent impact coefficient (default 0)
    risk_aversion : lambda in Almgren-Chriss
    """
    tau = 1.0
    if eta <= 0:
        # zero impact — execute all at T=0
        trades = np.zeros(T)
        trades[0] = X
        return trades, np.concatenate([[X], np.zeros(T)])

    kappa_sq = (risk_aversion * sigma * sigma) / (eta * (1 - 0.5 * gamma * tau / eta))
    kappa = float(np.sqrt(max(kappa_sq, 0.0)))
    j = np.arange(T + 1)
    denom = np.sinh(kappa * T * tau) if kappa * T * tau > 1e-12 else 1.0
    x_j = X * np.sinh(kappa * (T - j) * tau) / denom
    x_j = np.clip(x_j, 0.0, X)
    trades = -np.diff(x_j)
    # make sure trades sum to X (correct any rounding)
    diff = X - trades.sum()
    trades[-1] += diff
    return trades, x_j
