"""Factor models (Fama-French 3-factor skeleton + generic OLS factor regression)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FactorRegression:
    alpha: float
    betas: dict[str, float]
    r_squared: float
    residuals: pd.Series
    t_stats: dict[str, float]

    def summary(self) -> str:
        rows = [f"alpha     : {self.alpha:+.4f}  (t = {self.t_stats.get('alpha', float('nan')):+.2f})"]
        for k, v in self.betas.items():
            rows.append(f"{k:>10}: {v:+.4f}  (t = {self.t_stats.get(k, float('nan')):+.2f})")
        rows.append(f"R^2       : {self.r_squared:.4f}")
        return "\n".join(rows)


def factor_regression(returns: pd.Series, factors: pd.DataFrame) -> FactorRegression:
    """OLS regression: r_t = alpha + sum beta_i * f_{i,t} + eps. Returns coefficients + t-stats."""
    data = factors.join(returns.rename("_y")).dropna()
    y = data["_y"].values
    X = data[factors.columns].values
    X1 = np.hstack([np.ones((len(X), 1)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    preds = X1 @ coef
    resid = y - preds
    n, k = X1.shape
    dof = max(1, n - k)
    sigma2 = float((resid @ resid) / dof)
    try:
        cov = sigma2 * np.linalg.pinv(X1.T @ X1)
        se = np.sqrt(np.maximum(np.diag(cov), 1e-24))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    t_stats = {"alpha": float(coef[0] / se[0]) if se[0] > 0 else np.nan}
    for i, name in enumerate(factors.columns, 1):
        t_stats[name] = float(coef[i] / se[i]) if se[i] > 0 else np.nan
    ss_res = float((resid**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return FactorRegression(
        alpha=float(coef[0]),
        betas={name: float(c) for name, c in zip(factors.columns, coef[1:], strict=False)},
        r_squared=float(r2),
        residuals=pd.Series(resid, index=data.index),
        t_stats=t_stats,
    )


def simulate_fama_french_factors(n: int = 252, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic Fama-French-like factors: MKT, SMB, HML."""
    rng = np.random.default_rng(seed)
    cov = np.array([
        [0.0001, 2e-5, 1e-5],
        [2e-5, 5e-5, 1e-5],
        [1e-5, 1e-5, 6e-5],
    ])
    mu = np.array([0.0004, 0.0001, 0.0001])
    returns = rng.multivariate_normal(mu, cov, n)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame(returns, index=idx, columns=["MKT", "SMB", "HML"])
