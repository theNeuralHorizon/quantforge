"""Historical and factor-based stress scenarios."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


HISTORICAL_SHOCKS = {
    "2008_financial_crisis": {"equity": -0.37, "bond": 0.05, "credit": -0.20, "vol": 0.80},
    "2020_covid_crash": {"equity": -0.34, "bond": 0.08, "credit": -0.12, "vol": 1.50},
    "2022_rate_hike": {"equity": -0.18, "bond": -0.15, "credit": -0.10, "vol": 0.40},
    "dotcom_2000": {"equity": -0.49, "bond": 0.08, "credit": -0.05, "vol": 0.60},
    "black_monday_1987": {"equity": -0.22, "bond": 0.02, "credit": -0.15, "vol": 3.00},
    "flash_crash_2010": {"equity": -0.09, "bond": 0.01, "credit": -0.02, "vol": 0.45},
}


def shock_portfolio(weights: pd.Series, shocks: Dict[str, float]) -> float:
    """Apply per-asset shock to weights; returns portfolio PnL %."""
    missing = [k for k in weights.index if k not in shocks]
    if missing:
        raise ValueError(f"Missing shock for assets: {missing}")
    return float(sum(weights[a] * shocks[a] for a in weights.index))


def factor_shock(
    weights: pd.Series,
    factor_exposures: pd.DataFrame,
    factor_shocks: Dict[str, float],
) -> float:
    """PnL = w' * B * shock_f  where B has shape (assets, factors)."""
    shock = np.array([factor_shocks.get(f, 0.0) for f in factor_exposures.columns])
    asset_impact = factor_exposures.values @ shock
    return float(np.dot(weights.values, asset_impact))


def stress_scenarios(weights: pd.Series, asset_class_map: Dict[str, str]) -> pd.DataFrame:
    """Apply all HISTORICAL_SHOCKS to a portfolio given asset-class mapping."""
    rows = []
    for name, shock in HISTORICAL_SHOCKS.items():
        pnl = 0.0
        for asset, w in weights.items():
            ac = asset_class_map.get(asset, "equity")
            pnl += w * shock.get(ac, 0.0)
        rows.append({"scenario": name, "pnl_pct": pnl})
    return pd.DataFrame(rows).sort_values("pnl_pct")
