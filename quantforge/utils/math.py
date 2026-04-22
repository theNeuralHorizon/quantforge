"""Small math helpers."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def safe_divide(a: float | np.ndarray, b: float | np.ndarray, default: float = 0.0) -> float | np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(a, b)
        return np.where(np.isfinite(out), out, default)


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def rolling_apply(series: pd.Series, window: int, func: Callable[[np.ndarray], float]) -> pd.Series:
    return series.rolling(window).apply(func, raw=True)
