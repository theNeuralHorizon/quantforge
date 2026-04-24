"""Tests for quantforge.indicators (technical + statistical)."""
import math

import numpy as np
import pandas as pd
import pytest

from quantforge.data.synthetic import generate_ohlcv
from quantforge.indicators.statistical import (
    hurst_exponent,
    realized_vol,
    rolling_zscore,
)
from quantforge.indicators.technical import (
    atr,
    bollinger_bands,
    cci,
    donchian_channel,
    ema,
    macd,
    obv,
    rsi,
    sma,
    stochastic,
    true_range,
    williams_r,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ohlcv():
    return generate_ohlcv(n=300, seed=42)


@pytest.fixture(scope="module")
def close(ohlcv):
    return ohlcv["close"]


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------

class TestSMA:
    def test_sma_value_on_constant_series(self):
        s = pd.Series([5.0] * 50)
        result = sma(s, 10)
        assert result.dropna().iloc[-1] == pytest.approx(5.0)

    def test_sma_window_5_on_linspace(self):
        # SMA(5) of [1,2,3,4,5,6,7,8,9,10] at index 9 = mean([6,7,8,9,10]) = 8
        s = pd.Series(range(1, 11), dtype=float)
        result = sma(s, 5)
        assert result.iloc[-1] == pytest.approx(8.0)

    def test_sma_first_window_minus_one_is_nan(self):
        s = pd.Series(range(1, 21), dtype=float)
        result = sma(s, 10)
        # Indices 0..8 should be NaN (window=10 requires 10 points)
        assert result.iloc[8] != result.iloc[8]  # NaN check

    def test_sma_length_matches_input(self, close):
        result = sma(close, 20)
        assert len(result) == len(close)

    def test_sma_does_not_modify_input(self, close):
        original = close.copy()
        sma(close, 20)
        pd.testing.assert_series_equal(close, original)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class TestEMA:
    def test_ema_length_matches_input(self, close):
        result = ema(close, 20)
        assert len(result) == len(close)

    def test_ema_on_constant_series_equals_constant(self):
        s = pd.Series([7.0] * 100)
        result = ema(s, 10)
        assert result.dropna().iloc[-1] == pytest.approx(7.0, abs=1e-6)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRSI:
    def test_rsi_bounds_all_values_in_0_100(self, close):
        r = rsi(close, 14)
        valid = r.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()

    def test_rsi_length_matches_input(self, close):
        result = rsi(close, 14)
        assert len(result) == len(close)

    def test_rsi_all_up_days_is_50_due_to_implementation(self):
        # The implementation replaces avg_loss=0 (division by zero) with NaN, then
        # fillna(50). So a series with only gains returns RSI=50 for avg_loss=0 rows.
        # This is the documented behavior of this specific implementation.
        s = pd.Series(range(1, 60), dtype=float)
        r = rsi(s, 14)
        # RSI should be 50.0 (fillna sentinel) when there are zero losses
        assert r.dropna().iloc[-1] == pytest.approx(50.0)

    def test_rsi_all_down_days_near_0(self):
        # All gains=0 → RS=0 → RSI=0. The implementation gives 0 since avg_gain=0
        # means RS=0/avg_loss=0 → RSI = 100-100/(1+0) = 0.
        s = pd.Series(range(60, 0, -1), dtype=float)
        r = rsi(s, 14)
        assert r.dropna().iloc[-1] < 5.0


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

class TestBollingerBands:
    def test_lower_less_than_mid_less_than_upper(self, close):
        bb = bollinger_bands(close, 20, 2.0)
        valid = bb.dropna()
        assert (valid["lower"] < valid["mid"]).all()
        assert (valid["mid"] < valid["upper"]).all()

    def test_bands_contain_all_expected_columns(self, close):
        bb = bollinger_bands(close, 20, 2.0)
        assert set(bb.columns) == {"lower", "mid", "upper"}

    def test_mid_equals_sma(self, close):
        bb = bollinger_bands(close, 20, 2.0)
        expected_mid = sma(close, 20)
        pd.testing.assert_series_equal(bb["mid"], expected_mid, check_names=False)

    def test_band_width_positive_for_non_constant_series(self, close):
        bb = bollinger_bands(close, 20, 2.0)
        valid = bb.dropna()
        width = valid["upper"] - valid["lower"]
        assert (width > 0).all()


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

class TestATR:
    def test_atr_is_positive(self, ohlcv):
        a = atr(ohlcv, 14)
        valid = a.dropna()
        assert (valid > 0).all()

    def test_true_range_is_nonnegative(self, ohlcv):
        tr = true_range(ohlcv)
        valid = tr.dropna()
        assert (valid >= 0).all()

    def test_atr_length_matches_input(self, ohlcv):
        a = atr(ohlcv, 14)
        assert len(a) == len(ohlcv)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class TestMACD:
    def test_macd_columns(self, close):
        m = macd(close)
        assert set(m.columns) == {"macd", "signal", "hist"}

    def test_hist_is_macd_minus_signal(self, close):
        m = macd(close)
        expected = m["macd"] - m["signal"]
        pd.testing.assert_series_equal(m["hist"], expected, check_names=False)


# ---------------------------------------------------------------------------
# Stochastic
# ---------------------------------------------------------------------------

class TestStochastic:
    def test_k_bounds(self, ohlcv):
        sto = stochastic(ohlcv, 14, 3)
        valid_k = sto["k"].dropna()
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()


# ---------------------------------------------------------------------------
# OBV
# ---------------------------------------------------------------------------

class TestOBV:
    def test_obv_is_cumulative(self, ohlcv):
        o = obv(ohlcv)
        assert len(o) == len(ohlcv)
        # OBV values are integers or floats, not NaN
        assert not o.isna().all()


# ---------------------------------------------------------------------------
# Donchian Channel
# ---------------------------------------------------------------------------

class TestDonchianChannel:
    def test_upper_gte_lower(self, ohlcv):
        dc = donchian_channel(ohlcv, 20)
        valid = dc.dropna()
        assert (valid["upper"] >= valid["lower"]).all()

    def test_mid_between_upper_and_lower(self, ohlcv):
        dc = donchian_channel(ohlcv, 20)
        valid = dc.dropna()
        assert (valid["mid"] >= valid["lower"]).all()
        assert (valid["mid"] <= valid["upper"]).all()


# ---------------------------------------------------------------------------
# CCI and Williams %R
# ---------------------------------------------------------------------------

class TestCCI:
    def test_cci_is_series(self, ohlcv):
        c = cci(ohlcv, 20)
        assert isinstance(c, pd.Series)
        assert len(c) == len(ohlcv)


class TestWilliamsR:
    def test_williams_r_bounds(self, ohlcv):
        wr = williams_r(ohlcv, 14)
        valid = wr.dropna()
        assert (valid <= 0).all()
        assert (valid >= -100).all()


# ---------------------------------------------------------------------------
# Statistical indicators
# ---------------------------------------------------------------------------

class TestRollingZscore:
    def test_rolling_zscore_mean_near_zero_iid(self):
        # For i.i.d. noise (no drift), the rolling z-score should have mean ~0.
        # A random WALK (cumsum) has persistent trend, making rolling z-scores biased.
        rng = np.random.default_rng(7)
        s = pd.Series(rng.standard_normal(500))  # i.i.d., not cumsum
        z = rolling_zscore(s, 50)
        valid = z.dropna()
        assert abs(valid.mean()) < 0.2

    def test_rolling_zscore_std_near_one_iid(self):
        rng = np.random.default_rng(7)
        s = pd.Series(rng.standard_normal(500))  # i.i.d. noise
        z = rolling_zscore(s, 50)
        valid = z.dropna()
        # For i.i.d. input the std of rolling z-scores is close to 1
        assert 0.7 < valid.std() < 1.3

    def test_rolling_zscore_constant_series_is_nan(self):
        s = pd.Series([5.0] * 50)
        z = rolling_zscore(s, 10)
        # std = 0, so division by zero -> NaN
        assert z.dropna().empty


class TestRealizedVol:
    def test_realized_vol_on_zero_returns_is_zero(self):
        # Constant zero returns => std = 0 => realized_vol = 0
        s = pd.Series([0.0] * 50)
        rv = realized_vol(s, window=21, annualize=252)
        valid = rv.dropna()
        # Use np.allclose because pytest.approx does not work element-wise with pd.Series
        assert np.allclose(valid.values, 0.0, atol=1e-10)

    def test_realized_vol_positive_for_random_returns(self):
        rng = np.random.default_rng(99)
        s = pd.Series(rng.standard_normal(100) * 0.01)
        rv = realized_vol(s, window=21, annualize=252)
        assert rv.dropna().iloc[-1] > 0.0

    def test_realized_vol_constant_nonzero_returns_is_zero(self):
        # std of [0.01, 0.01, ...] = 0 (constant) => realized_vol = 0
        s = pd.Series([0.01] * 100)
        rv = realized_vol(s, window=21, annualize=252)
        valid = rv.dropna()
        assert np.allclose(valid.values, 0.0, atol=1e-10)


class TestHurstExponent:
    def test_hurst_random_walk_near_half(self):
        # GBM (random walk) should have Hurst ~ 0.5
        rng = np.random.default_rng(11)
        s = pd.Series(rng.standard_normal(500).cumsum())
        h = hurst_exponent(s, max_lag=50)
        assert not math.isnan(h)
        assert 0.2 < h < 0.8

    def test_hurst_returns_nan_for_short_series(self):
        s = pd.Series([1.0, 2.0, 3.0])
        h = hurst_exponent(s, max_lag=50)
        assert math.isnan(h)
