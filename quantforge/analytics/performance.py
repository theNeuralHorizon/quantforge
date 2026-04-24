"""Return-series performance statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.core.constants import TRADING_DAYS_YEAR
from quantforge.risk.drawdown import max_drawdown
from quantforge.risk.metrics import (
    calmar_ratio,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
    ulcer_index,
)
from quantforge.risk.var import historical_cvar, historical_var


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS_YEAR) -> float:
    if len(returns) < 2:
        return 0.0
    r = returns.dropna()
    total = (1 + r).prod()
    years = len(r) / periods
    if years <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def annualized_vol(returns: pd.Series, periods: int = TRADING_DAYS_YEAR) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods))


def cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod() - 1


def rolling_sharpe(returns: pd.Series, window: int = 63, periods: int = TRADING_DAYS_YEAR) -> pd.Series:
    m = returns.rolling(window).mean()
    s = returns.rolling(window).std()
    return (m / s.replace(0, np.nan)) * np.sqrt(periods)


def win_rate(trades: pd.Series) -> float:
    t = trades.dropna()
    if t.empty:
        return 0.0
    return float((t > 0).sum() / len(t))


def profit_factor(trades: pd.Series) -> float:
    t = trades.dropna()
    gains = t[t > 0].sum()
    losses = -t[t < 0].sum()
    if losses == 0:
        return np.inf
    return float(gains / losses)


def avg_win_loss(trades: pd.Series) -> dict[str, float]:
    t = trades.dropna()
    wins = t[t > 0]
    losses = t[t < 0]
    return {
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "win_loss_ratio": float(abs(wins.mean() / losses.mean())) if len(losses) and losses.mean() != 0 else np.inf,
    }


def summary_stats(equity: pd.Series, risk_free: float = 0.0, trades: pd.DataFrame | None = None) -> dict[str, float]:
    r = equity.pct_change().dropna()
    mdd, peak, trough = max_drawdown(equity)
    out = {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0,
        "annual_return": annualized_return(r),
        "annual_vol": annualized_vol(r),
        "sharpe": sharpe_ratio(r, risk_free),
        "sortino": sortino_ratio(r),
        "calmar": calmar_ratio(equity),
        "omega": omega_ratio(r),
        "tail_ratio": tail_ratio(r),
        "ulcer_index": ulcer_index(equity),
        "max_drawdown": mdd,
        "var_95": historical_var(r, 0.95),
        "cvar_95": historical_cvar(r, 0.95),
        "best_day": float(r.max()) if len(r) else 0.0,
        "worst_day": float(r.min()) if len(r) else 0.0,
        "skew": float(r.skew()) if len(r) > 2 else 0.0,
        "kurt": float(r.kurt()) if len(r) > 3 else 0.0,
    }
    if trades is not None and not trades.empty and len(equity) > 0:
        notional = (trades["qty"].abs() * trades["price"]).sum()
        cost = trades["commission"].sum() + (trades["qty"].abs() * trades["slippage"]).sum()
        out["turnover"] = float(notional / equity.mean())
        out["total_cost"] = float(cost)
        out["cost_bps"] = float(1e4 * cost / notional) if notional > 0 else 0.0
        out["n_trades"] = int(len(trades))
    return out
