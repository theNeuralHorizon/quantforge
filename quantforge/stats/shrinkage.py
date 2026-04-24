"""Covariance matrix shrinkage (Ledoit-Wolf, constant-correlation, OAS)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def constant_correlation_target(sample_cov: np.ndarray) -> np.ndarray:
    """Shrinkage target with constant pairwise correlation."""
    std = np.sqrt(np.diag(sample_cov))
    n = sample_cov.shape[0]
    corr = sample_cov / np.outer(std, std)
    # mean off-diagonal correlation
    avg_corr = (corr.sum() - np.trace(corr)) / (n * (n - 1))
    target = avg_corr * np.outer(std, std)
    np.fill_diagonal(target, np.diag(sample_cov))
    return target


def ledoit_wolf_shrinkage(
    returns: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf optimal linear shrinkage towards constant-correlation target.

    Returns (shrunk_cov, shrinkage_intensity in [0,1]).
    """
    X = returns.values if isinstance(returns, pd.DataFrame) else np.asarray(returns)
    X = X[~np.isnan(X).any(axis=1)]
    n, p = X.shape
    if n < 10 or p < 2:
        raise ValueError("need at least 10 rows and 2 columns")

    Xc = X - X.mean(axis=0)
    sample_cov = (Xc.T @ Xc) / n
    target = constant_correlation_target(sample_cov)

    # pi = sum of asymptotic variance of (x_i - mu_i)(x_j - mu_j) - sample_cov_{ij}
    y2 = Xc**2
    pi_mat = (y2.T @ y2) / n - sample_cov**2
    pi_hat = pi_mat.sum()

    # rho (asymptotic covariance between target and sample cov)
    std = np.sqrt(np.diag(sample_cov))
    corr = sample_cov / np.outer(std, std)
    avg_corr = (corr.sum() - np.trace(corr)) / (p * (p - 1))
    (y2.T @ (Xc ** 2 * 0)) / n  # placeholder; we use simpler form
    # simpler rho: use diagonal contribution
    rho_diag = pi_mat.diagonal().sum()
    term_off = 0.0
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            term_off += avg_corr * 0.5 * (
                np.sqrt(sample_cov[j, j] / sample_cov[i, i]) * ((y2[:, i] * Xc[:, i] * Xc[:, j]).mean() - sample_cov[i, i] * sample_cov[i, j]) +
                np.sqrt(sample_cov[i, i] / sample_cov[j, j]) * ((y2[:, j] * Xc[:, i] * Xc[:, j]).mean() - sample_cov[j, j] * sample_cov[i, j])
            )
    rho_hat = rho_diag + term_off

    gamma = float(((target - sample_cov) ** 2).sum())

    if gamma <= 0:
        shrinkage = 0.0
    else:
        kappa = (pi_hat - rho_hat) / gamma
        shrinkage = max(0.0, min(1.0, kappa / n))

    shrunk = shrinkage * target + (1 - shrinkage) * sample_cov
    return shrunk, float(shrinkage)


def oracle_shrinkage(
    returns: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, float]:
    """Oracle Approximating Shrinkage (Chen, Wiesel, Eldar, Hero 2010)."""
    X = returns.values if isinstance(returns, pd.DataFrame) else np.asarray(returns)
    X = X[~np.isnan(X).any(axis=1)]
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / n
    trace_S = np.trace(S)
    trace_S2 = np.trace(S @ S)

    num = (1 - 2 / p) * trace_S2 + (trace_S ** 2)
    den = (n + 1 - 2 / p) * (trace_S2 - trace_S ** 2 / p)
    if den <= 0:
        rho = 0.0
    else:
        rho = float(min(1.0, num / den))

    mu = trace_S / p
    target = mu * np.eye(p)
    shrunk = rho * target + (1 - rho) * S
    return shrunk, rho


def shrunk_covariance(
    returns: pd.DataFrame | np.ndarray,
    method: str = "ledoit_wolf",
) -> np.ndarray:
    """Convenience wrapper. method ∈ {'ledoit_wolf', 'oracle'}."""
    if method == "ledoit_wolf":
        cov, _ = ledoit_wolf_shrinkage(returns)
    elif method == "oracle":
        cov, _ = oracle_shrinkage(returns)
    else:
        raise ValueError(f"unknown method {method}")
    return cov
