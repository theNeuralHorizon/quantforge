"""Cointegration tests: Engle-Granger (2-asset), Johansen trace (multi-asset),
plus rolling-window cointegration tracking.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# Approximate Engle-Granger critical values (from MacKinnon 2010, no-trend model)
_EG_CRIT = {
    2: {0.01: -3.90, 0.05: -3.34, 0.10: -3.04},  # 2 variables
    3: {0.01: -4.29, 0.05: -3.74, 0.10: -3.45},
    4: {0.01: -4.65, 0.05: -4.10, 0.10: -3.81},
    5: {0.01: -4.96, 0.05: -4.42, 0.10: -4.13},
}


@dataclass
class CointegrationResult:
    statistic: float
    pvalue: float
    beta: Optional[np.ndarray]   # cointegration vector
    residuals: Optional[pd.Series]
    is_cointegrated_95: bool


def _simple_adf(x: np.ndarray) -> float:
    """Dickey-Fuller t-statistic on a demeaned series (no lag)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 20:
        return float("nan")
    y = np.diff(x)
    lag = x[:-1]
    lag_c = lag - lag.mean()
    X = np.column_stack([lag_c, np.ones_like(lag_c)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = max(1, n - k)
    sigma2 = float((resid @ resid) / dof)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(max(cov[0, 0], 1e-18)))
    return float(beta[0] / se) if se > 0 else float("nan")


def engle_granger(
    y: pd.Series, x: pd.Series, significance: float = 0.05,
) -> CointegrationResult:
    """Engle-Granger 2-step cointegration: regress y on x, ADF-test residuals."""
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 30:
        return CointegrationResult(float("nan"), float("nan"), None, None, False)
    Y = df["y"].values
    X = np.column_stack([df["x"].values, np.ones(len(df))])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    resid_s = pd.Series(resid, index=df.index)

    stat = _simple_adf(resid)
    crit = _EG_CRIT[2]
    if stat < crit[0.01]:
        p = 0.005
    elif stat < crit[0.05]:
        p = 0.025
    elif stat < crit[0.10]:
        p = 0.075
    else:
        p = 0.5
    return CointegrationResult(
        statistic=float(stat), pvalue=float(p),
        beta=np.asarray(beta, dtype=float),
        residuals=resid_s,
        is_cointegrated_95=(p <= 0.05),
    )


def johansen_trace(
    series: pd.DataFrame, max_rank: Optional[int] = None,
) -> List[dict]:
    """Johansen trace test (Osterwald-Lenum critical values at 5%, approx).

    Returns a list, one row per rank r = 0..k-1, with trace_stat and
    critical value at 5%. When trace_stat > crit, reject "rank <= r".
    """
    data = series.dropna().values
    k = data.shape[1]
    n = data.shape[0]
    if n < 30 or k < 2:
        return []

    # VECM: Delta Y_t = alpha * beta' * Y_{t-1} + Pi * Delta Y_{t-1} + eps
    dY = np.diff(data, axis=0)
    Y_lag = data[:-1]
    dY_lag = np.vstack([np.zeros((1, k)), np.diff(data[:-1], axis=0)])

    # Regress dY on dY_lag → residuals R0
    Z = dY_lag
    beta_r0, *_ = np.linalg.lstsq(Z, dY, rcond=None)
    R0 = dY - Z @ beta_r0
    beta_r1, *_ = np.linalg.lstsq(Z, Y_lag, rcond=None)
    R1 = Y_lag - Z @ beta_r1

    S00 = R0.T @ R0 / n
    S11 = R1.T @ R1 / n
    S01 = R0.T @ R1 / n
    S10 = S01.T

    # Solve eigenvalue problem  |lambda S11 - S10 S00^-1 S01| = 0
    try:
        S00_inv = np.linalg.pinv(S00)
        mat = np.linalg.pinv(S11) @ S10 @ S00_inv @ S01
        eigvals = np.sort(np.real(np.linalg.eigvals(mat)))[::-1]
    except np.linalg.LinAlgError:
        return []

    eigvals = np.clip(eigvals, 1e-12, 1 - 1e-12)
    trace_stats = [-n * np.log(1 - eigvals[r:]).sum() for r in range(k)]

    # Approximate 5% critical values from Osterwald-Lenum (no trend)
    ol_crit_5pct = {1: 3.84, 2: 15.49, 3: 29.80, 4: 47.86, 5: 69.82, 6: 95.75}
    rows = []
    for r in range(k):
        # testing H0: rank <= r against H1: rank > r
        k_minus_r = k - r
        crit = ol_crit_5pct.get(k_minus_r, float("inf"))
        rows.append({
            "rank_tested": r,
            "trace_stat": float(trace_stats[r]),
            "crit_value_5pct": float(crit),
            "reject_null": trace_stats[r] > crit,
            "eigenvalue": float(eigvals[r]),
        })
    return rows


def rolling_cointegration(
    y: pd.Series, x: pd.Series, window: int = 252, step: int = 21,
) -> pd.DataFrame:
    """Compute Engle-Granger stat over a rolling window; useful to see whether
    a pair's cointegration is stable.
    """
    out = []
    idx = y.index.intersection(x.index)
    y, x = y.loc[idx], x.loc[idx]
    for end in range(window, len(y) + 1, step):
        w = slice(end - window, end)
        r = engle_granger(y.iloc[w], x.iloc[w])
        out.append({
            "end_date": y.index[end - 1],
            "statistic": r.statistic, "pvalue": r.pvalue,
            "beta": float(r.beta[0]) if r.beta is not None else float("nan"),
            "is_cointegrated_95": r.is_cointegrated_95,
        })
    return pd.DataFrame(out).set_index("end_date")


def half_life_of_mean_reversion(spread: pd.Series) -> float:
    """AR(1) half-life: t_{1/2} = -ln(2) / ln(phi)."""
    s = spread.dropna().values
    if len(s) < 5:
        return float("nan")
    lag = s[:-1]
    y = s[1:]
    X = np.column_stack([lag, np.ones_like(lag)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    phi = float(beta[0])
    if phi >= 1 or phi <= 0:
        return float("inf")
    return float(-np.log(2) / np.log(phi))
