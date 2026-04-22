"""Risk-adjusted performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.core.constants import TRADING_DAYS_YEAR


def _arr(r):
    if isinstance(r, pd.Series):
        return r.dropna().values
    return np.asarray(r)


def sharpe_ratio(returns, risk_free: float = 0.0, periods: int = TRADING_DAYS_YEAR) -> float:
    r = _arr(returns)
    if len(r) < 2:
        return np.nan
    excess = r - risk_free / periods
    std = excess.std(ddof=1)
    if std == 0:
        return np.nan
    return float(np.sqrt(periods) * excess.mean() / std)


def sortino_ratio(returns, target: float = 0.0, periods: int = TRADING_DAYS_YEAR) -> float:
    r = _arr(returns)
    if len(r) < 2:
        return np.nan
    excess = r - target / periods
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.inf
    dd = np.sqrt((downside**2).mean())
    if dd == 0:
        return np.inf
    return float(np.sqrt(periods) * excess.mean() / dd)


def calmar_ratio(equity: pd.Series, periods: int = TRADING_DAYS_YEAR) -> float:
    if len(equity) < 2:
        return np.nan
    returns = equity.pct_change().dropna()
    ann_ret = (1 + returns.mean()) ** periods - 1
    dd = (equity / equity.cummax() - 1).min()
    if dd == 0:
        return np.inf
    return float(ann_ret / abs(dd))


def omega_ratio(returns, threshold: float = 0.0) -> float:
    r = _arr(returns) - threshold
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return np.inf
    return float(gains / losses)


def tail_ratio(returns, q: float = 0.05) -> float:
    r = _arr(returns)
    if len(r) < 10:
        return np.nan
    right = np.quantile(r, 1 - q)
    left = np.quantile(r, q)
    if left == 0:
        return np.inf
    return float(abs(right / left))


def ulcer_index(equity: pd.Series) -> float:
    dd = (equity / equity.cummax() - 1) * 100
    return float(np.sqrt((dd**2).mean()))


def gain_to_pain(returns) -> float:
    r = _arr(returns)
    losses = -r[r < 0].sum()
    if losses == 0:
        return np.inf
    return float(r.sum() / losses)


def information_ratio(returns, benchmark, periods: int = TRADING_DAYS_YEAR) -> float:
    r = _arr(returns)
    b = _arr(benchmark)
    n = min(len(r), len(b))
    if n < 2:
        return np.nan
    active = r[-n:] - b[-n:]
    std = active.std(ddof=1)
    if std == 0:
        return np.nan
    return float(np.sqrt(periods) * active.mean() / std)
