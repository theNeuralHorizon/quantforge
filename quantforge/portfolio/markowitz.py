"""Mean-Variance / Markowitz portfolio optimization.

Implemented with scipy (no CVXPY dependency needed) for robustness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _cov_numpy(cov: pd.DataFrame | np.ndarray) -> np.ndarray:
    return cov.values if isinstance(cov, pd.DataFrame) else np.asarray(cov)


def _mu_numpy(mu: pd.Series | np.ndarray) -> np.ndarray:
    return mu.values if isinstance(mu, pd.Series) else np.asarray(mu)


def min_variance(
    cov: pd.DataFrame | np.ndarray,
    bounds: tuple[float, float] | None = (0.0, 1.0),
    total: float = 1.0,
) -> np.ndarray:
    """Minimum-variance long-only fully-invested portfolio."""
    S = _cov_numpy(cov)
    n = S.shape[0]
    x0 = np.full(n, total / n)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - total},)
    bnds = [bounds] * n
    res = minimize(
        lambda w: w @ S @ w,
        x0, method="SLSQP", bounds=bnds, constraints=cons,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    return res.x


def max_sharpe(
    mu: pd.Series | np.ndarray,
    cov: pd.DataFrame | np.ndarray,
    risk_free: float = 0.0,
    bounds: tuple[float, float] | None = (0.0, 1.0),
    total: float = 1.0,
) -> np.ndarray:
    """Tangency portfolio (max Sharpe)."""
    S = _cov_numpy(cov)
    m = _mu_numpy(mu) - risk_free
    n = S.shape[0]
    x0 = np.full(n, total / n)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - total},)
    bnds = [bounds] * n

    def neg_sharpe(w):
        ret = w @ m
        vol = np.sqrt(max(1e-12, w @ S @ w))
        return -ret / vol

    res = minimize(
        neg_sharpe, x0, method="SLSQP", bounds=bnds, constraints=cons,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    return res.x


def mean_variance(
    mu: pd.Series | np.ndarray,
    cov: pd.DataFrame | np.ndarray,
    gamma: float = 1.0,
    bounds: tuple[float, float] | None = (0.0, 1.0),
    total: float = 1.0,
) -> np.ndarray:
    """Max `w'mu - 0.5*gamma*w'Sw` subject to sum(w)=total and bounds."""
    S = _cov_numpy(cov)
    m = _mu_numpy(mu)
    n = S.shape[0]
    x0 = np.full(n, total / n)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - total},)
    bnds = [bounds] * n
    res = minimize(
        lambda w: -(w @ m) + 0.5 * gamma * (w @ S @ w),
        x0, method="SLSQP", bounds=bnds, constraints=cons,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    return res.x


def efficient_frontier(
    mu: pd.Series | np.ndarray,
    cov: pd.DataFrame | np.ndarray,
    n_points: int = 30,
    bounds: tuple[float, float] | None = (0.0, 1.0),
) -> pd.DataFrame:
    """Sweep target returns; return DataFrame with risk/return/weights."""
    S = _cov_numpy(cov)
    m = _mu_numpy(mu)
    n = S.shape[0]
    target_rets = np.linspace(m.min(), m.max(), n_points)
    out = []
    for tr in target_rets:
        cons = (
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "eq", "fun": lambda w, tr=tr: w @ m - tr},
        )
        res = minimize(
            lambda w: w @ S @ w,
            np.full(n, 1 / n), method="SLSQP", bounds=[bounds] * n, constraints=cons,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if not res.success:
            continue
        w = res.x
        out.append({"target_return": tr, "risk": float(np.sqrt(w @ S @ w)), "return": float(w @ m), "weights": w})
    return pd.DataFrame(out)
