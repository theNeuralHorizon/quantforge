"""Crypto data helpers — thin adapter on top of yfinance for BTC/ETH/etc.

Yahoo exposes crypto via tickers like BTC-USD, ETH-USD, SOL-USD etc.
This module normalises the ticker (auto-appends -USD if missing) and
adds a 24/7 calendar-aware resampler for strategies that expect business days.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from quantforge.data.loader import DataLoader


_CRYPTO_RX = re.compile(r"^[A-Z0-9]{2,10}(-[A-Z]{3})?$")


# Common shortcuts → fully-qualified yfinance tickers
_ALIASES = {
    "BTC": "BTC-USD", "BITCOIN": "BTC-USD",
    "ETH": "ETH-USD", "ETHEREUM": "ETH-USD",
    "SOL": "SOL-USD", "SOLANA": "SOL-USD",
    "XRP": "XRP-USD", "ADA": "ADA-USD",
    "DOGE": "DOGE-USD", "DOT": "DOT-USD",
    "AVAX": "AVAX-USD", "MATIC": "MATIC-USD",
    "LINK": "LINK-USD", "LTC": "LTC-USD",
    "BCH": "BCH-USD", "BNB": "BNB-USD",
}


def normalize_crypto_ticker(ticker: str) -> str:
    """Turn 'BTC' / 'bitcoin' / 'BTC-USD' into a yfinance-compatible ticker."""
    t = ticker.strip().upper()
    if t in _ALIASES:
        return _ALIASES[t]
    if not _CRYPTO_RX.match(t):
        raise ValueError(f"invalid crypto ticker: {ticker!r}")
    # If the user gave a bare symbol with no quote currency, append -USD
    if "-" not in t:
        t = f"{t}-USD"
    return t


def load_crypto(
    ticker: str,
    start: str = "2020-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    cache_dir: Optional[str] = "data/cache",
) -> pd.DataFrame:
    """Fetch one crypto OHLCV series via yfinance."""
    t = normalize_crypto_ticker(ticker)
    dl = DataLoader(cache_dir=cache_dir)
    return dl.yfinance(
        t, start=start,
        end=end or pd.Timestamp.today().strftime("%Y-%m-%d"),
        interval=interval,
    )


def load_crypto_panel(
    tickers: Iterable[str],
    start: str = "2020-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    cache_dir: Optional[str] = "data/cache",
) -> dict[str, pd.DataFrame]:
    dl = DataLoader(cache_dir=cache_dir)
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        norm = normalize_crypto_ticker(t)
        df = dl.yfinance(norm, start=start,
                          end=end or pd.Timestamp.today().strftime("%Y-%m-%d"),
                          interval=interval)
        if not df.empty:
            out[norm] = df
    return out


def crypto_volatility_24_7(returns: pd.Series, periods: int = 365) -> float:
    """Annualized vol using 365 periods (crypto trades 24/7)."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods))
