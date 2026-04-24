"""CVaR minimization (Rockafellar-Uryasev) via LP surrogate.

Given scenarios r_s (T x N), we minimize CVaR_alpha(w'r) s.t. sum(w)=1, w>=0.
We use the classic scenario-based LP via SciPy's linprog.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def minimize_cvar(
    scenarios: np.ndarray,
    alpha: float = 0.95,
    target_return: float | None = None,
    expected_returns: np.ndarray | None = None,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Return weights that minimize CVaR_alpha.

    Decision vector: [w (N), zeta (1), u_s (T)].
    Minimize zeta + (1/(T*(1-alpha))) * sum(u_s)
    s.t.  -r_s' w - zeta <= u_s,   u_s >= 0
          sum(w) = 1
          w in [lo, hi]
          w' mu >= target_return  (optional)
    """
    r = np.asarray(scenarios)
    T, N = r.shape
    c = np.zeros(N + 1 + T)
    c[N] = 1.0
    c[N + 1:] = 1.0 / (T * (1 - alpha))

    # Inequality: -r w - zeta - u <= 0  => [- r, -1, -I] * [w; zeta; u] <= 0
    A_ub1 = np.hstack([-r, -np.ones((T, 1)), -np.eye(T)])
    b_ub1 = np.zeros(T)

    A_ub, b_ub = A_ub1, b_ub1

    if target_return is not None and expected_returns is not None:
        mu = np.asarray(expected_returns)
        row = np.zeros(N + 1 + T)
        row[:N] = -mu
        A_ub = np.vstack([A_ub, row])
        b_ub = np.concatenate([b_ub, [-target_return]])

    A_eq = np.zeros((1, N + 1 + T))
    A_eq[0, :N] = 1
    b_eq = np.array([1.0])

    bnds = [bounds] * N + [(None, None)] + [(0, None)] * T

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bnds,
                  method="highs", options={"disp": False})
    if not res.success:
        raise RuntimeError(f"CVaR LP failed: {res.message}")
    return res.x[:N]


def historical_cvar_from_scenarios(weights: np.ndarray, scenarios: np.ndarray, alpha: float = 0.95) -> float:
    """CVaR of the portfolio portfolio from scenario returns."""
    port = scenarios @ weights
    q = np.quantile(port, 1 - alpha)
    tail = port[port <= q]
    return float(-tail.mean()) if len(tail) else float(-q)
