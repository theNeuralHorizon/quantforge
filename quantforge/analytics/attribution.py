"""Performance attribution models."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def brinson_attribution(
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Dict[str, float]:
    """Brinson-Hood-Beebower attribution decomposition.

    Returns allocation, selection, interaction, and total.
    """
    idx = portfolio_weights.index.intersection(benchmark_weights.index)
    wp, wb = portfolio_weights.reindex(idx).fillna(0), benchmark_weights.reindex(idx).fillna(0)
    rp, rb = portfolio_returns.reindex(idx).fillna(0), benchmark_returns.reindex(idx).fillna(0)

    benchmark_total = float((wb * rb).sum())
    allocation = float(((wp - wb) * (rb - benchmark_total)).sum())
    selection = float((wb * (rp - rb)).sum())
    interaction = float(((wp - wb) * (rp - rb)).sum())
    total = allocation + selection + interaction
    return {
        "allocation": allocation,
        "selection": selection,
        "interaction": interaction,
        "total": total,
        "benchmark": benchmark_total,
    }


def factor_attribution(returns: pd.Series, factor_returns: pd.DataFrame) -> Dict[str, float]:
    """Time-series regression: r_t = alpha + sum(beta_i * f_i) + eps."""
    aligned = returns.to_frame("r").join(factor_returns, how="inner").dropna()
    if len(aligned) < 10:
        return {"alpha": np.nan, "r2": np.nan, "beta": {}}
    y = aligned["r"].values
    X = aligned[factor_returns.columns].values
    X1 = np.hstack([np.ones((len(X), 1)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    alpha = float(coef[0])
    betas = {c: float(b) for c, b in zip(factor_returns.columns, coef[1:])}
    preds = X1 @ coef
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"alpha": alpha, "r2": r2, "beta": betas}
