"""Regime detection: simple rule-based + Gaussian-mixture HMM-lite."""
from __future__ import annotations

import numpy as np
import pandas as pd


def bull_bear_regime(close: pd.Series, lookback: int = 200) -> pd.Series:
    """Price above / below long SMA."""
    sma = close.rolling(lookback).mean()
    return pd.Series(np.where(close > sma, 1, -1), index=close.index).fillna(0)


def vol_regime(returns: pd.Series, window: int = 63, hi_quantile: float = 0.8, lo_quantile: float = 0.2) -> pd.Series:
    """Classify as high / mid / low vol regime by rolling percentile."""
    rv = returns.rolling(window).std()
    rolling_rank = rv.rolling(252, min_periods=window).rank(pct=True)
    out = pd.Series(index=returns.index, dtype=object)
    out[rolling_rank > hi_quantile] = "high"
    out[rolling_rank < lo_quantile] = "low"
    out[out.isna()] = "mid"
    return out


def trend_regime(close: pd.Series, short: int = 50, long: int = 200) -> pd.Series:
    """1 = uptrend (short>long), -1 = downtrend."""
    s = close.rolling(short).mean()
    l = close.rolling(long).mean()
    return pd.Series(np.where(s > l, 1, -1), index=close.index).fillna(0)


def hmm_regimes(returns: pd.Series, n_states: int = 2, seed: int | None = 42) -> pd.Series:
    """Fit a Gaussian Mixture on returns as a lightweight regime detector.

    This is not a full HMM (no transition dynamics), but captures the regime
    clustering that HMMs often achieve in practice. Requires sklearn.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return pd.Series(0, index=returns.index)

    r = returns.dropna().values.reshape(-1, 1)
    if len(r) < max(50, n_states * 10):
        return pd.Series(0, index=returns.index)

    gmm = GaussianMixture(n_components=n_states, random_state=seed, covariance_type="full")
    gmm.fit(r)
    labels = gmm.predict(r)

    # Sort regime labels by mean return so labels are stable (0 = worst, n-1 = best)
    order = np.argsort([m[0] for m in gmm.means_])
    relabel = {old: new for new, old in enumerate(order)}
    mapped = np.array([relabel[l] for l in labels])

    out = pd.Series(index=returns.index, dtype="float64")
    out.loc[returns.dropna().index] = mapped
    return out.ffill().fillna(0).astype(int)
