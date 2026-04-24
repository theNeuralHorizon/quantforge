"""Pure-numpy GARCH(1,1) fit + forecast. No statsmodels / arch dependency."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class GARCHParams:
    omega: float
    alpha: float
    beta: float
    log_likelihood: float
    n_obs: int

    @property
    def unconditional_variance(self) -> float:
        denom = 1.0 - self.alpha - self.beta
        if denom <= 0:
            return float("nan")
        return self.omega / denom

    def persistence(self) -> float:
        return self.alpha + self.beta


def _garch11_variance(returns: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """Iterate conditional variance sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}."""
    n = len(returns)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(returns) if n > 1 else omega / max(1e-9, 1 - alpha - beta)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
    return np.maximum(sigma2, 1e-12)


def _neg_log_likelihood(params: np.ndarray, returns: np.ndarray) -> float:
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
        return 1e20
    sigma2 = _garch11_variance(returns, omega, alpha, beta)
    ll = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2).sum()
    return -ll


def garch11_fit(returns: pd.Series | np.ndarray) -> GARCHParams:
    """Fit a GARCH(1,1) model by MLE. Returns fitted (omega, alpha, beta) + log-lik."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 50:
        raise ValueError("need at least 50 observations")
    r = r - r.mean()  # work with demeaned returns

    var_r = r.var()
    # reasonable starting values
    x0 = [var_r * 0.05, 0.1, 0.85]
    bounds = [(1e-9, None), (0.0, 0.9999), (0.0, 0.9999)]
    cons = {"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]}
    res = minimize(
        _neg_log_likelihood, x0, args=(r,),
        method="SLSQP", bounds=bounds, constraints=[cons],
        options={"maxiter": 500, "ftol": 1e-9},
    )
    omega, alpha, beta = res.x
    return GARCHParams(
        omega=float(omega), alpha=float(alpha), beta=float(beta),
        log_likelihood=float(-res.fun), n_obs=len(r),
    )


def garch11_forecast(
    returns: pd.Series | np.ndarray,
    params: GARCHParams,
    horizon: int = 21,
) -> np.ndarray:
    """Forecast conditional variance h steps ahead."""
    r = np.asarray(returns, dtype=float)
    sigma2 = _garch11_variance(r, params.omega, params.alpha, params.beta)
    last_sigma2 = sigma2[-1]
    last_r = r[-1]
    params.alpha + params.beta

    forecasts = np.empty(horizon)
    forecasts[0] = params.omega + params.alpha * last_r**2 + params.beta * last_sigma2
    for h in range(1, horizon):
        forecasts[h] = params.omega + (params.alpha + params.beta) * forecasts[h - 1]
    return forecasts
