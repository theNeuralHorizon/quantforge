"""Example 01: Generate synthetic data and compute a stack of indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.data.synthetic import generate_ohlcv
from quantforge.indicators.technical import (
    sma, ema, rsi, macd, bollinger_bands, atr, adx, stochastic, vwap, obv,
)
from quantforge.indicators.statistical import (
    rolling_zscore, realized_vol, ewma_vol, garman_klass_vol,
    hurst_exponent, adf_test, half_life,
)


def main() -> None:
    df = generate_ohlcv(500, s0=100, mu=0.1, sigma=0.22, seed=42)
    close = df["close"]

    print("=" * 70)
    print("Last 3 bars of OHLCV")
    print("=" * 70)
    print(df.tail(3).round(2))

    print("\n" + "=" * 70)
    print("Technical indicators (latest values)")
    print("=" * 70)
    print(f"SMA(20)      : {sma(close, 20).iloc[-1]:.2f}")
    print(f"EMA(20)      : {ema(close, 20).iloc[-1]:.2f}")
    print(f"RSI(14)      : {rsi(close).iloc[-1]:.2f}")
    mac = macd(close).iloc[-1]
    print(f"MACD         : {mac['macd']:.3f}  signal={mac['signal']:.3f}  hist={mac['hist']:.3f}")
    bb = bollinger_bands(close).iloc[-1]
    print(f"Bollinger    : mid={bb['mid']:.2f}  upper={bb['upper']:.2f}  lower={bb['lower']:.2f}")
    print(f"ATR(14)      : {atr(df).iloc[-1]:.3f}")
    adx_ = adx(df).iloc[-1]
    print(f"ADX(14)      : +DI={adx_['plus_di']:.2f}  -DI={adx_['minus_di']:.2f}  ADX={adx_['adx']:.2f}")
    sto = stochastic(df).iloc[-1]
    print(f"Stochastic   : %K={sto['k']:.2f}  %D={sto['d']:.2f}")
    print(f"VWAP         : {vwap(df).iloc[-1]:.2f}")
    print(f"OBV          : {obv(df).iloc[-1]:,.0f}")

    print("\n" + "=" * 70)
    print("Statistical indicators")
    print("=" * 70)
    r = close.pct_change()
    print(f"Rolling z-score(20) : {rolling_zscore(close, 20).iloc[-1]:.3f}")
    print(f"Realized vol (21d)  : {realized_vol(r, 21).iloc[-1] * 100:.2f}%")
    print(f"EWMA vol            : {ewma_vol(r).iloc[-1] * 100:.2f}%")
    print(f"Garman-Klass vol    : {garman_klass_vol(df).iloc[-1] * 100:.2f}%")
    print(f"Hurst exponent      : {hurst_exponent(close):.3f}  (0.5 = random walk)")
    stat, pv = adf_test(close)
    print(f"ADF                 : stat={stat:.3f}  p={pv:.3f}")
    hl = half_life(close)
    print(f"Half-life           : {hl:.1f} bars")


if __name__ == "__main__":
    main()
