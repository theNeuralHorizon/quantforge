"""Benchmark-relative performance metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantforge.core.constants import TRADING_DAYS_YEAR


def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return df["a"], df["b"]


def alpha_beta(strategy_returns: pd.Series, benchmark_returns: pd.Series, risk_free: float = 0.0, periods: int = TRADING_DAYS_YEAR) -> dict[str, float]:
    """CAPM regression: r_s = alpha + beta * r_b + eps (daily, then alpha annualized)."""
    s, b = _align(strategy_returns, benchmark_returns)
    if len(s) < 10:
        return {"alpha": np.nan, "beta": np.nan, "r_squared": np.nan}
    rf = risk_free / periods
    y = (s - rf).values
    x = (b - rf).values
    n = len(y)
    var_x = x.var(ddof=1)
    if var_x == 0:
        return {"alpha": np.nan, "beta": np.nan, "r_squared": np.nan}
    cov = ((x - x.mean()) * (y - y.mean())).sum() / (n - 1)
    beta = cov / var_x
    alpha_daily = y.mean() - beta * x.mean()
    alpha_ann = (1 + alpha_daily) ** periods - 1
    preds = alpha_daily + beta * x
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"alpha": float(alpha_ann), "beta": float(beta), "r_squared": float(r2)}


def tracking_error(strategy_returns: pd.Series, benchmark_returns: pd.Series, periods: int = TRADING_DAYS_YEAR) -> float:
    s, b = _align(strategy_returns, benchmark_returns)
    if len(s) < 2:
        return np.nan
    diff = s - b
    return float(diff.std(ddof=1) * np.sqrt(periods))


def information_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series, periods: int = TRADING_DAYS_YEAR) -> float:
    s, b = _align(strategy_returns, benchmark_returns)
    if len(s) < 2:
        return np.nan
    diff = s - b
    sd = diff.std(ddof=1)
    if sd == 0:
        return np.nan
    return float(np.sqrt(periods) * diff.mean() / sd)


def up_down_capture(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> dict[str, float]:
    s, b = _align(strategy_returns, benchmark_returns)
    up = b > 0
    down = b < 0
    up_ratio = float(s[up].mean() / b[up].mean()) if up.any() and b[up].mean() != 0 else np.nan
    dn_ratio = float(s[down].mean() / b[down].mean()) if down.any() and b[down].mean() != 0 else np.nan
    return {"up_capture": up_ratio, "down_capture": dn_ratio}


@dataclass
class BenchmarkReport:
    alpha: float
    beta: float
    r_squared: float
    info_ratio: float
    tracking_error: float
    up_capture: float
    down_capture: float

    def __str__(self) -> str:
        def p(x): return f"{x*100:+.2f}%" if not np.isnan(x) else "nan"
        def f(x): return f"{x:+.4f}" if not np.isnan(x) else "nan"
        return (
            f"Alpha (annual): {p(self.alpha)}\n"
            f"Beta          : {f(self.beta)}\n"
            f"R-squared     : {f(self.r_squared)}\n"
            f"Info Ratio    : {f(self.info_ratio)}\n"
            f"Tracking Error: {p(self.tracking_error)}\n"
            f"Up Capture    : {f(self.up_capture)}\n"
            f"Down Capture  : {f(self.down_capture)}"
        )


def benchmark_report(strategy_returns: pd.Series, benchmark_returns: pd.Series, risk_free: float = 0.0) -> BenchmarkReport:
    ab = alpha_beta(strategy_returns, benchmark_returns, risk_free)
    ud = up_down_capture(strategy_returns, benchmark_returns)
    return BenchmarkReport(
        alpha=ab["alpha"], beta=ab["beta"], r_squared=ab["r_squared"],
        info_ratio=information_ratio(strategy_returns, benchmark_returns),
        tracking_error=tracking_error(strategy_returns, benchmark_returns),
        up_capture=ud["up_capture"], down_capture=ud["down_capture"],
    )
