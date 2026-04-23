"""Portfolio risk attribution — decompose total risk into per-asset contributions.

Standard Euler decomposition: for a homogeneous-of-degree-1 risk measure R,
the marginal contribution of asset i is dR/dw_i, and the component
contribution (Euler) is w_i * dR/dw_i. Summing components recovers R(w).

Supports both volatility (analytical) and historical VaR/CVaR (numerical).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class RiskAttributionReport:
    total_risk: float
    marginal_risk: pd.Series           # dR/dw_i
    component_risk: pd.Series          # w_i * dR/dw_i — sums to total_risk
    percent_contribution: pd.Series    # component_risk / total_risk

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "weight": self.component_risk / self.marginal_risk,
            "marginal": self.marginal_risk,
            "component": self.component_risk,
            "pct_contribution": self.percent_contribution,
        })


def _as_array(x, name: str = "x") -> np.ndarray:
    a = x.values if hasattr(x, "values") else np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return a.astype(float)


def volatility_attribution(
    weights: pd.Series | np.ndarray,
    cov: pd.DataFrame | np.ndarray,
    names: list | None = None,
) -> RiskAttributionReport:
    """Analytical Euler decomposition of portfolio volatility.

    Portfolio variance = w' S w. Portfolio vol sigma_p = sqrt(w' S w).
    Marginal contribution = (S w) / sigma_p.
    Component contribution = w_i * marginal_i; sum = sigma_p exactly.
    """
    w = _as_array(weights, "weights")
    S = cov.values if isinstance(cov, pd.DataFrame) else np.asarray(cov, dtype=float)
    if S.shape != (len(w), len(w)):
        raise ValueError("cov must be (N, N) matching weights")
    if names is None:
        if isinstance(weights, pd.Series):
            names = list(weights.index)
        elif isinstance(cov, pd.DataFrame):
            names = list(cov.columns)
        else:
            names = [f"asset_{i}" for i in range(len(w))]

    var = float(w @ S @ w)
    vol = float(np.sqrt(max(var, 1e-24)))
    marginal = (S @ w) / vol
    component = w * marginal
    pct = component / vol if vol > 0 else np.zeros_like(component)
    return RiskAttributionReport(
        total_risk=vol,
        marginal_risk=pd.Series(marginal, index=names),
        component_risk=pd.Series(component, index=names),
        percent_contribution=pd.Series(pct, index=names),
    )


def var_attribution(
    weights: pd.Series | np.ndarray,
    scenarios: pd.DataFrame | np.ndarray,
    confidence: float = 0.95,
    names: list | None = None,
) -> RiskAttributionReport:
    """Component VaR via historical simulation + conditional expectation.

    The marginal contribution to VaR is the conditional expectation of asset i's
    return given the portfolio return equals the VaR quantile. Component
    contribution is w_i * marginal_i, and they sum to the portfolio VaR.

    We use a neighborhood around the VaR-quantile scenario (default 5% of
    scenarios) to stabilize the conditional expectation.
    """
    w = _as_array(weights, "weights")
    R = scenarios.values if isinstance(scenarios, pd.DataFrame) else np.asarray(scenarios, dtype=float)
    if R.ndim != 2 or R.shape[1] != len(w):
        raise ValueError("scenarios must be (T, N) matching weights length")
    if names is None:
        if isinstance(weights, pd.Series):
            names = list(weights.index)
        elif isinstance(scenarios, pd.DataFrame):
            names = list(scenarios.columns)
        else:
            names = [f"asset_{i}" for i in range(len(w))]

    port = R @ w
    q = np.quantile(port, 1 - confidence)
    var = float(-q)

    # Neighborhood: scenarios within the 5% closest to the q-quantile
    tol = max(port.std() * 0.05, 1e-6)
    mask = np.abs(port - q) <= tol
    if mask.sum() < 5:
        # fall back: bottom 5% of scenarios
        k = max(5, int(len(port) * 0.05))
        order = np.argsort(port)
        mask = np.zeros(len(port), dtype=bool)
        mask[order[:k]] = True

    marginal = -R[mask].mean(axis=0)
    component = w * marginal
    total = float(component.sum())
    # Rescale so the components sum exactly to var (small numerical drift)
    if abs(total) > 1e-12:
        scale = var / total
        component = component * scale
        marginal = marginal * scale
    pct = component / var if var > 0 else np.zeros_like(component)

    return RiskAttributionReport(
        total_risk=var,
        marginal_risk=pd.Series(marginal, index=names),
        component_risk=pd.Series(component, index=names),
        percent_contribution=pd.Series(pct, index=names),
    )


def cvar_attribution(
    weights: pd.Series | np.ndarray,
    scenarios: pd.DataFrame | np.ndarray,
    confidence: float = 0.95,
    names: list | None = None,
) -> RiskAttributionReport:
    """Component CVaR: expected loss contribution per asset conditional on
    being in the tail.
    """
    w = _as_array(weights, "weights")
    R = scenarios.values if isinstance(scenarios, pd.DataFrame) else np.asarray(scenarios, dtype=float)
    if R.ndim != 2 or R.shape[1] != len(w):
        raise ValueError("scenarios must be (T, N) matching weights length")
    if names is None:
        if isinstance(weights, pd.Series):
            names = list(weights.index)
        elif isinstance(scenarios, pd.DataFrame):
            names = list(scenarios.columns)
        else:
            names = [f"asset_{i}" for i in range(len(w))]

    port = R @ w
    q = np.quantile(port, 1 - confidence)
    tail = port <= q
    if not tail.any():
        tail = np.zeros_like(port, dtype=bool)
        tail[int(np.argmin(port))] = True

    cvar = float(-port[tail].mean())
    marginal = -R[tail].mean(axis=0)
    component = w * marginal
    total = float(component.sum())
    if abs(total) > 1e-12:
        scale = cvar / total
        component = component * scale
        marginal = marginal * scale
    pct = component / cvar if cvar > 0 else np.zeros_like(component)

    return RiskAttributionReport(
        total_risk=cvar,
        marginal_risk=pd.Series(marginal, index=names),
        component_risk=pd.Series(component, index=names),
        percent_contribution=pd.Series(pct, index=names),
    )


def risk_budget_deviation(
    weights: pd.Series | np.ndarray,
    cov: pd.DataFrame | np.ndarray,
    target_budget: Dict[str, float] | np.ndarray,
) -> pd.DataFrame:
    """How far is each asset's risk contribution from its target?

    A portfolio is "risk parity" if target_budget is uniform and deviations are 0.
    """
    report = volatility_attribution(weights, cov)
    names = list(report.component_risk.index)
    if isinstance(target_budget, dict):
        target = np.array([target_budget.get(n, 0.0) for n in names])
    else:
        target = np.asarray(target_budget, dtype=float)
    if abs(target.sum() - 1.0) > 1e-6:
        target = target / max(target.sum(), 1e-12)
    actual_pct = report.percent_contribution.values
    return pd.DataFrame({
        "target_pct": target,
        "actual_pct": actual_pct,
        "deviation": actual_pct - target,
    }, index=names)
