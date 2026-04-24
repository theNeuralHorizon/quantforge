"""Synthetic market data generators. Useful for deterministic tests + demos without network."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def generate_gbm(
    n: int = 252,
    s0: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.2,
    dt: float = 1 / 252,
    seed: int | None = None,
) -> pd.Series:
    """Generate a Geometric Brownian Motion price series."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    drift = (mu - 0.5 * sigma**2) * dt
    shock = sigma * np.sqrt(dt) * z
    log_returns = drift + shock
    prices = s0 * np.exp(np.cumsum(log_returns))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.Series(prices, index=idx, name="close")


def generate_ohlcv(
    n: int = 252,
    s0: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.2,
    intraday_vol: float = 0.5,
    base_volume: float = 1_000_000.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a plausible OHLCV frame from a GBM close series."""
    rng = np.random.default_rng(seed)
    close = generate_gbm(n, s0, mu, sigma, seed=seed)
    prev_close = close.shift(1).fillna(s0).values
    c = close.values

    open_ = prev_close * (1 + rng.normal(0, sigma * intraday_vol * np.sqrt(1 / 252), size=n))
    bar_range = np.abs(rng.normal(0, sigma * intraday_vol * np.sqrt(1 / 252), size=n)) * c
    high = np.maximum(np.maximum(open_, c), np.maximum(open_, c) + bar_range * 0.5)
    low = np.minimum(np.minimum(open_, c), np.minimum(open_, c) - bar_range * 0.5)
    volume = base_volume * np.exp(rng.normal(0, 0.3, size=n))

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": c, "volume": volume},
        index=close.index,
    )


def generate_correlated_returns(
    n: int = 252,
    symbols: Sequence[str] = ("A", "B", "C"),
    mu: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate correlated log-returns for multiple symbols."""
    rng = np.random.default_rng(seed)
    k = len(symbols)
    if mu is None:
        mu = np.full(k, 0.08 / 252)
    if cov is None:
        base = np.full((k, k), 0.3 * (0.2**2) / 252)
        np.fill_diagonal(base, (0.2**2) / 252)
        cov = base
    returns = rng.multivariate_normal(mean=mu, cov=cov, size=n)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame(returns, index=idx, columns=list(symbols))


def generate_panel(
    symbols: Sequence[str] = ("AAA", "BBB", "CCC"),
    n: int = 252,
    seed: int | None = None,
) -> dict:
    """Generate a dict of OHLCV frames keyed by symbol — a small panel."""
    out = {}
    for i, s in enumerate(symbols):
        out[s] = generate_ohlcv(
            n=n,
            s0=100.0 + 20 * i,
            mu=0.05 + 0.03 * i,
            sigma=0.15 + 0.05 * i,
            seed=None if seed is None else seed + i,
        )
    return out
