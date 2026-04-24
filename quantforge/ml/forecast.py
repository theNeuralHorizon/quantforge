"""Simple forecasters: AR(1), EWMA, linear regression on features, sequence helper."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ar_forecast(series: pd.Series, p: int = 1, horizon: int = 1) -> float:
    """Fit AR(p) via least squares and forecast horizon steps."""
    s = series.dropna().values.astype(float)
    if len(s) < p + 10:
        return float(s[-1]) if len(s) else np.nan
    X = np.column_stack([s[p - i - 1 : len(s) - i - 1] for i in range(p)])
    y = s[p:]
    coef, *_ = np.linalg.lstsq(np.hstack([X, np.ones((len(X), 1))]), y, rcond=None)
    a = coef[:-1]
    c = coef[-1]
    hist = list(s[-p:])
    for _ in range(horizon):
        next_v = c + float(np.dot(a, hist[::-1][:p]))
        hist.append(next_v)
    return float(hist[-1])


def ewma_forecast(series: pd.Series, span: int = 20, horizon: int = 1) -> float:
    """Forecast as the latest EWMA — simple but surprisingly robust."""
    s = series.dropna()
    if s.empty:
        return np.nan
    return float(s.ewm(span=span, adjust=False).mean().iloc[-1])


def linear_forecast(features: pd.DataFrame, target: pd.Series, test_size: int = 1) -> tuple[np.ndarray, float]:
    """Fit linear regression, predict the last `test_size` rows. Returns (preds, train_R2)."""
    data = features.join(target.rename("_y")).dropna()
    if len(data) < test_size + 10:
        return np.full(test_size, np.nan), np.nan
    X = data.drop(columns="_y").values
    y = data["_y"].values
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train = y[:-test_size]
    X_train_b = np.hstack([X_train, np.ones((len(X_train), 1))])
    X_test_b = np.hstack([X_test, np.ones((len(X_test), 1))])
    beta, *_ = np.linalg.lstsq(X_train_b, y_train, rcond=None)
    preds = X_test_b @ beta
    train_preds = X_train_b @ beta
    ss_res = float(((y_train - train_preds) ** 2).sum())
    ss_tot = float(((y_train - y_train.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return preds, r2


def make_sequences(series: pd.Series, window: int = 20, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Transform a series into (n_samples, window) X and (n_samples,) y for NN training."""
    s = series.dropna().values.astype(float)
    n = len(s) - window - horizon + 1
    if n <= 0:
        return np.empty((0, window)), np.empty(0)
    X = np.lib.stride_tricks.sliding_window_view(s[: -horizon], window)[:n]
    y = s[window + horizon - 1 : window + horizon - 1 + n]
    return X, y
