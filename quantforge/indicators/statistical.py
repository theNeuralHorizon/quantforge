"""Statistical indicators: rolling stats, mean-reversion tests, volatility estimators."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / s.replace(0, np.nan)


def rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    return a.rolling(window).corr(b)


def rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / var.replace(0, np.nan)


def rolling_skew(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).skew()


def rolling_kurt(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).kurt()


def hurst_exponent(series: pd.Series, max_lag: int = 50) -> float:
    """Estimate Hurst exponent via the log-log slope of variance-of-increments."""
    s = series.dropna().values
    if len(s) < max_lag * 2:
        return np.nan
    lags = np.arange(2, max_lag)
    tau = [np.std(np.subtract(s[lag:], s[:-lag])) for lag in lags]
    tau = np.asarray(tau)
    mask = tau > 0
    if mask.sum() < 5:
        return np.nan
    slope = np.polyfit(np.log(lags[mask]), np.log(tau[mask]), 1)[0]
    return float(slope)


_ADF_CRIT_TABLE = {  # approximate MacKinnon critical values (no-trend model)
    0.01: -3.43, 0.05: -2.86, 0.10: -2.57,
}


def _native_adf(series: pd.Series, max_lag: int = 1) -> tuple[float, float]:
    """Native Dickey-Fuller (with a lag of residuals to be "augmented")."""
    s = series.dropna().values.astype(float)
    if len(s) < 20:
        return (np.nan, np.nan)
    dy = np.diff(s)
    y_lag = s[:-1]
    X = [y_lag]
    for k in range(1, max_lag + 1):
        if k >= len(dy):
            break
        X.append(np.concatenate([np.zeros(k), dy[:-k]]))
    X.append(np.ones_like(y_lag))
    Xm = np.column_stack(X)
    try:
        beta, *_ = np.linalg.lstsq(Xm, dy, rcond=None)
    except np.linalg.LinAlgError:
        return (np.nan, np.nan)
    resid = dy - Xm @ beta
    n, k = Xm.shape
    dof = max(1, n - k)
    sigma2 = float((resid @ resid) / dof)
    try:
        cov = sigma2 * np.linalg.pinv(Xm.T @ Xm)
        se = float(np.sqrt(max(1e-18, cov[0, 0])))
    except np.linalg.LinAlgError:
        return (np.nan, np.nan)
    stat = float(beta[0] / se)
    # very rough p-value interpolation across critical values
    if stat < _ADF_CRIT_TABLE[0.01]:
        p = 0.005
    elif stat < _ADF_CRIT_TABLE[0.05]:
        p = 0.025
    elif stat < _ADF_CRIT_TABLE[0.10]:
        p = 0.075
    else:
        p = 0.5
    return stat, p


def adf_test(series: pd.Series) -> tuple[float, float]:
    """Augmented Dickey-Fuller (stat, pvalue). Uses statsmodels if available; else native."""
    try:
        from statsmodels.tsa.stattools import adfuller
        s = series.dropna()
        if len(s) < 20:
            return (np.nan, np.nan)
        res = adfuller(s, autolag="AIC")
        return (float(res[0]), float(res[1]))
    except ImportError:
        return _native_adf(series, max_lag=1)


def half_life(series: pd.Series) -> float:
    """Half-life of mean reversion from AR(1) fit on differences."""
    s = series.dropna().values
    if len(s) < 5:
        return np.nan
    lag = s[:-1]
    diff = np.diff(s)
    beta = np.polyfit(lag - lag.mean(), diff, 1)[0]
    if beta >= 0 or np.isnan(beta):
        return np.inf
    return float(-np.log(2) / beta)


def realized_vol(returns: pd.Series, window: int = 21, annualize: float = 252) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(annualize)


def ewma_vol(returns: pd.Series, lam: float = 0.94, annualize: float = 252) -> pd.Series:
    """RiskMetrics EWMA variance estimator."""
    var = returns.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var * annualize)


def garman_klass_vol(df: pd.DataFrame, window: int = 21, annualize: float = 252) -> pd.Series:
    """Garman-Klass range volatility estimator."""
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    rs = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    return np.sqrt(rs.rolling(window).mean() * annualize)
