"""Data loaders. yfinance is optional; falls back gracefully without network."""
from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def load_csv(path: str | os.PathLike, parse_dates: str = "date") -> pd.DataFrame:
    df = pd.read_csv(path)
    if parse_dates in df.columns:
        df[parse_dates] = pd.to_datetime(df[parse_dates])
        df = df.set_index(parse_dates)
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


@dataclass
class DataLoader:
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str, start: str, end: str, interval: str) -> Path | None:
        if not self.cache_dir:
            return None
        fname = f"{symbol}_{start}_{end}_{interval}.parquet"
        return Path(self.cache_dir) / fname

    def yfinance(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        """Fetch via yfinance. Raises ImportError if yfinance unavailable."""
        cp = self._cache_path(symbol, start, end, interval)
        if cp and cp.exists():
            return pd.read_parquet(cp)
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError("yfinance not installed; use synthetic data or CSV") from e
        df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        if cp is not None and not df.empty:
            df.to_parquet(cp)
        return df

    def yfinance_many(self, symbols: Iterable[str], start: str, end: str, interval: str = "1d") -> dict:
        return {s: self.yfinance(s, start, end, interval) for s in symbols}
