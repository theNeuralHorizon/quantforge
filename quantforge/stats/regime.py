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

    Implementation notes:
      - The xi-sum step (E-step's joint posterior of consecutive states)
        is fully vectorized as a 3-D outer product instead of a per-t
        Python triple loop. With n=400 obs and n_states=2 that's
        638k Python iterations replaced by ~5 numpy ops per EM step.
      - The Gaussian emission matrix `E` of shape (n, n_states) is
        recomputed once per EM iteration (after the M-step updates mu /
        sigma) instead of n+1 times inside the forward + backward + xi
        loops.
    """
    r = returns.dropna().values
    n = len(r)
    if n < 50:
        raise ValueError("need at least 50 obs")
    # rng kept (currently unused) so callers can rely on reproducible
    # initialisation when we add randomised restarts later.
    np.random.default_rng(seed)

    # Initialise: spread means across (mean ± std), equal sigmas, and a
    # transition matrix biased toward staying in the current state (this
    # is the empirical sweet spot for daily-return regime detection).
    mu = np.linspace(r.mean() - r.std(), r.mean() + r.std(), n_states)
    sigma = np.full(n_states, r.std())
    trans = 0.9 * np.eye(n_states) + 0.1 / n_states * np.ones((n_states, n_states))
    pi0 = np.full(n_states, 1.0 / n_states)
    sqrt_2pi = np.sqrt(2 * np.pi)

    def _emit_matrix() -> np.ndarray:
        """Gaussian emission probabilities of shape (n, n_states)."""
        z = (r[:, None] - mu) / sigma  # (n, K)
        return np.exp(-0.5 * z * z) / (sigma * sqrt_2pi)

    prev_ll = -np.inf
    gamma = np.zeros((n, n_states))
    for _ in range(max_iter):
        E = _emit_matrix()  # (n, K), reused by forward + backward + xi

        # Forward: alpha[t] = (alpha[t-1] @ trans) * E[t], normalised.
        alpha = np.zeros((n, n_states))
        c = np.zeros(n)
        alpha[0] = pi0 * E[0]
        c[0] = alpha[0].sum()
        if c[0] <= 0:
            break
        alpha[0] /= c[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ trans) * E[t]
            c[t] = alpha[t].sum()
            if c[t] <= 0:
                break
            alpha[t] /= c[t]

        # Backward: beta_[t] = trans @ (E[t+1] * beta_[t+1]) / c[t+1]
        beta_ = np.zeros((n, n_states))
        beta_[-1] = 1.0
        c_safe = np.maximum(c, 1e-300)
        for t in range(n - 2, -1, -1):
            beta_[t] = (trans @ (E[t + 1] * beta_[t + 1])) / c_safe[t + 1]

        gamma = alpha * beta_
        gamma /= gamma.sum(axis=1, keepdims=True)

        # xi-sum (vectorized): xi[t,i,j] = alpha[t,i] * trans[i,j] *
        # E[t+1,j] * beta_[t+1,j]. Shape (n-1, K, K). Each slice is
        # normalised by its own scalar sum (= P(o_{1:T}) up to scaling),
        # then summed over t.
        eb = E[1:] * beta_[1:]                               # (n-1, K)
        xi = alpha[:-1, :, None] * trans[None, :, :] * eb[:, None, :]  # (n-1, K, K)
        denom = xi.sum(axis=(1, 2))                           # (n-1,)
        valid = denom > 0
        if valid.any():
            xi[valid] /= denom[valid, None, None]
            xi[~valid] = 0
            xi_sum = xi.sum(axis=0)
        else:
            xi_sum = np.zeros((n_states, n_states))

        # M-step
        pi0 = gamma[0]
        row_sum = gamma[:-1].sum(axis=0)                     # (K,)
        # Vectorised transition matrix update with safe division.
        trans = np.where(
            row_sum[:, None] > 0,
            xi_sum / np.maximum(row_sum[:, None], 1e-300),
            trans,
        )
        gsum = np.maximum(gamma.sum(axis=0), 1e-12)
        mu = (gamma * r[:, None]).sum(axis=0) / gsum
        resid = (r[:, None] - mu) ** 2
        sigma = np.maximum(np.sqrt((gamma * resid).sum(axis=0) / gsum), 1e-6)

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
