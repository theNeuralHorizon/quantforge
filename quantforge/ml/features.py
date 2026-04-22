"""Feature engineering for quant ML models."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from quantforge.indicators.technical import rsi, macd, atr, bollinger_bands
from quantforge.indicators.statistical import rolling_zscore, realized_vol


def price_features(close: pd.Series, windows: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    for w in windows:
        out[f"ret_{w}"] = close.pct_change(w)
        out[f"logret_{w}"] = np.log(close).diff(w)
        out[f"sma_ratio_{w}"] = close / close.rolling(w).mean() - 1
        out[f"zscore_{w}"] = rolling_zscore(close, w)
    return out


def volume_features(volume: pd.Series, windows: List[int] = [5, 20, 60]) -> pd.DataFrame:
    out = pd.DataFrame(index=volume.index)
    for w in windows:
        out[f"vol_ratio_{w}"] = volume / volume.rolling(w).mean()
        out[f"vol_z_{w}"] = rolling_zscore(volume, w)
    return out


def volatility_features(close: pd.Series, windows: List[int] = [10, 21, 60]) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    r = close.pct_change()
    for w in windows:
        out[f"rv_{w}"] = realized_vol(r, w)
        out[f"skew_{w}"] = r.rolling(w).skew()
        out[f"kurt_{w}"] = r.rolling(w).kurt()
    return out


def cross_sectional_rank(panel: Dict[str, pd.Series]) -> pd.DataFrame:
    """Convert dict of series to a cross-sectional rank (per-timestamp percentile)."""
    wide = pd.DataFrame(panel)
    return wide.rank(axis=1, pct=True)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature matrix from OHLCV frame."""
    feats = [
        price_features(df["close"]),
        volatility_features(df["close"]),
    ]
    if "volume" in df.columns:
        feats.append(volume_features(df["volume"]))
    out = pd.concat(feats, axis=1)
    out["rsi_14"] = rsi(df["close"], 14)
    mac = macd(df["close"])
    out["macd_hist"] = mac["hist"]
    bb = bollinger_bands(df["close"])
    out["bb_pos"] = (df["close"] - bb["lower"]) / (bb["upper"] - bb["lower"])
    if {"high", "low", "close"}.issubset(df.columns):
        out["atr_14"] = atr(df, 14)
    return out.dropna()


def target_labels(
    close: pd.Series,
    horizon: int = 1,
    kind: str = "direction",
    threshold: float = 0.0,
) -> pd.Series:
    """
    `kind`:
      - 'direction': +1 / 0 / -1 based on threshold
      - 'binary': 1 if up else 0
      - 'regression': raw forward return
    """
    fwd = close.pct_change(horizon).shift(-horizon)
    if kind == "binary":
        return (fwd > 0).astype(int)
    if kind == "regression":
        return fwd
    labels = np.where(fwd > threshold, 1, np.where(fwd < -threshold, -1, 0))
    return pd.Series(labels, index=close.index)
