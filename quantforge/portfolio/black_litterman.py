"""Black-Litterman posterior mean & covariance."""
from __future__ import annotations

import numpy as np
import pandas as pd


def black_litterman(
    cov: pd.DataFrame | np.ndarray,
    market_caps: np.ndarray,
    risk_aversion: float = 2.5,
    P: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    tau: float = 0.05,
    risk_free: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (posterior_mu, posterior_cov) under the Black-Litterman model.

    Implied equilibrium returns: pi = lambda * Sigma * w_mkt
    Posterior mu: [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1 * pi + P'*Omega^-1*Q]
    """
    S = cov.values if isinstance(cov, pd.DataFrame) else np.asarray(cov)
    w_mkt = np.asarray(market_caps, dtype=float)
    w_mkt = w_mkt / w_mkt.sum()

    pi = risk_aversion * S @ w_mkt + risk_free

    if P is None or Q is None:
        return pi, S

    P = np.atleast_2d(P)
    Q = np.atleast_1d(Q)
    if omega is None:
        omega = np.diag(np.diag(P @ (tau * S) @ P.T))

    tauS_inv = np.linalg.pinv(tau * S)
    omega_inv = np.linalg.pinv(omega)
    A = tauS_inv + P.T @ omega_inv @ P
    b = tauS_inv @ pi + P.T @ omega_inv @ Q
    posterior_mu = np.linalg.solve(A, b)
    posterior_cov = S + np.linalg.pinv(A)
    return posterior_mu, posterior_cov
