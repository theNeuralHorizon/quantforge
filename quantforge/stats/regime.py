"""Regime detection: Markov switching on returns, structural break detection."""
from __future__ import annotations

import numpy as np
import pandas as pd


def markov_switching_returns(
    returns: pd.Series, n_states: int = 2, max_iter: int = 200, tol: float = 1e-6,
    seed: int = 42,
) -> pd.DataFrame:
    """Hamilton-style Markov switching on i.i.d. Gaussian returns.

    EM-fits `n_states` regimes (different mean + variance) + a transition matrix.
    Returns a DataFrame with the inferred state at each timestamp + the
    smoothed posterior probabilities.
    """
    r = returns.dropna().values
    n = len(r)
    if n < 50:
        raise ValueError("need at least 50 obs")
    np.random.default_rng(seed)

    # initialize
    mu = np.linspace(r.mean() - r.std(), r.mean() + r.std(), n_states)
    sigma = np.full(n_states, r.std())
    trans = np.full((n_states, n_states), 1.0 / n_states)
    trans = 0.9 * np.eye(n_states) + 0.1 / n_states * np.ones((n_states, n_states))
    pi0 = np.full(n_states, 1.0 / n_states)

    def _emit(x):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    prev_ll = -np.inf
    for _ in range(max_iter):
        # Forward
        alpha = np.zeros((n, n_states))
        alpha[0] = pi0 * _emit(r[0])
        c = np.zeros(n)
        c[0] = alpha[0].sum()
        if c[0] <= 0:
            break
        alpha[0] /= c[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ trans) * _emit(r[t])
            c[t] = alpha[t].sum()
            if c[t] <= 0:
                break
            alpha[t] /= c[t]
        # Backward
        beta_ = np.zeros((n, n_states))
        beta_[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta_[t] = (trans @ (_emit(r[t + 1]) * beta_[t + 1])) / max(c[t + 1], 1e-300)
        gamma = alpha * beta_
        gamma = gamma / gamma.sum(axis=1, keepdims=True)

        # xi (joint posterior of consecutive states)
        xi_sum = np.zeros((n_states, n_states))
        for t in range(n - 1):
            denom = 0.0
            temp = np.zeros((n_states, n_states))
            for i in range(n_states):
                for j in range(n_states):
                    temp[i, j] = alpha[t, i] * trans[i, j] * _emit(r[t + 1])[j] * beta_[t + 1, j]
                    denom += temp[i, j]
            if denom > 0:
                xi_sum += temp / denom

        # M-step
        pi0 = gamma[0]
        row_sum = gamma[:-1].sum(axis=0)
        for i in range(n_states):
            if row_sum[i] > 0:
                trans[i] = xi_sum[i] / row_sum[i]
        gsum = gamma.sum(axis=0)
        mu = (gamma * r[:, None]).sum(axis=0) / np.maximum(gsum, 1e-12)
        resid = (r[:, None] - mu) ** 2
        sigma = np.sqrt((gamma * resid).sum(axis=0) / np.maximum(gsum, 1e-12))
        sigma = np.maximum(sigma, 1e-6)

        ll = np.log(np.maximum(c, 1e-300)).sum()
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    # Reorder states by mean return for stable labels
    order = np.argsort(mu)
    gamma = gamma[:, order]
    mu = mu[order]
    sigma = sigma[order]
    state = gamma.argmax(axis=1)

    out = pd.DataFrame(
        {
            "state": state,
            **{f"p_state_{i}": gamma[:, i] for i in range(n_states)},
        },
        index=returns.dropna().index,
    )
    out.attrs["mu"] = mu.tolist()
    out.attrs["sigma"] = sigma.tolist()
    return out


def detect_structural_breaks(
    series: pd.Series, window: int = 63, threshold: float = 3.0,
) -> list[pd.Timestamp]:
    """Simple structural-break detector via rolling z-score of rolling mean.

    Returns a list of timestamps where the rolling mean makes a z-move larger
    than `threshold` vs. its prior-window distribution. Not CUSUM-rigorous
    but useful as a first pass.
    """
    x = series.dropna()
    if len(x) < window * 2:
        return []
    rmean = x.rolling(window).mean()
    rmean_diff = rmean.diff(window)
    rstd = x.rolling(window).std()
    z = rmean_diff / rstd.replace(0, np.nan)
    return list(z.index[np.abs(z.fillna(0)) > threshold])
