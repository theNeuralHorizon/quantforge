"""Risk Parity / Equal Risk Contribution portfolio."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _cov(cov) -> np.ndarray:
    return cov.values if isinstance(cov, pd.DataFrame) else np.asarray(cov)


def risk_parity(
    cov: pd.DataFrame | np.ndarray,
    target: Optional[np.ndarray] = None,
    total: float = 1.0,
    max_iter: int = 500,
) -> np.ndarray:
    """Risk-budgeting optimizer: minimize sum of (RC_i - t_i)^2."""
    S = _cov(cov)
    n = S.shape[0]
    t = np.full(n, 1 / n) if target is None else np.asarray(target)

    def objective(w):
        vol = np.sqrt(max(1e-14, w @ S @ w))
        mrc = S @ w / vol
        rc = w * mrc
        rc_pct = rc / rc.sum()
        return float(((rc_pct - t) ** 2).sum())

    x0 = np.full(n, total / n)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - total},)
    bnds = [(1e-6, 1.0)] * n
    res = minimize(
        objective, x0, method="SLSQP", bounds=bnds, constraints=cons,
        options={"maxiter": max_iter, "ftol": 1e-12},
    )
    return res.x


def equal_risk_contribution(cov: pd.DataFrame | np.ndarray, total: float = 1.0) -> np.ndarray:
    """ERC = risk parity with uniform target."""
    return risk_parity(cov, target=None, total=total)
