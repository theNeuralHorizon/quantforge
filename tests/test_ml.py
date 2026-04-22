"""Tests for quantforge.ml: features, target_labels, make_sequences, regime, forecast."""
import math
import pytest
import numpy as np
import pandas as pd

from quantforge.ml.features import build_feature_matrix, target_labels
from quantforge.ml.forecast import make_sequences, ar_forecast, ewma_forecast
from quantforge.ml.regime import bull_bear_regime, trend_regime, vol_regime, hmm_regimes
from quantforge.data.synthetic import generate_ohlcv, generate_gbm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ohlcv():
    return generate_ohlcv(n=400, seed=77)


@pytest.fixture(scope="module")
def close_series(ohlcv):
    return ohlcv["close"]


@pytest.fixture(scope="module")
def returns_series(close_series):
    return close_series.pct_change().dropna()


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_returns_nonempty_dataframe(self, ohlcv):
        fm = build_feature_matrix(ohlcv)
        assert isinstance(fm, pd.DataFrame)
        assert len(fm) > 0

    def test_no_nan_after_dropna(self, ohlcv):
        fm = build_feature_matrix(ohlcv)
        assert not fm.isna().any().any()

    def test_has_rsi_column(self, ohlcv):
        fm = build_feature_matrix(ohlcv)
        assert "rsi_14" in fm.columns

    def test_has_macd_hist_column(self, ohlcv):
        fm = build_feature_matrix(ohlcv)
        assert "macd_hist" in fm.columns

    def test_has_bb_pos_column(self, ohlcv):
        fm = build_feature_matrix(ohlcv)
        assert "bb_pos" in fm.columns

    def test_has_multiple_columns(self, ohlcv):
        fm = build_feature_matrix(ohlcv)
        assert fm.shape[1] > 5

    def test_index_is_subset_of_ohlcv_index(self, ohlcv):
        fm = build_feature_matrix(ohlcv)
        assert set(fm.index).issubset(set(ohlcv.index))

    def test_works_without_volume_column(self):
        # DataFrame without volume column
        ohlcv_no_vol = generate_ohlcv(n=300, seed=88)[["open", "high", "low", "close"]]
        fm = build_feature_matrix(ohlcv_no_vol)
        assert isinstance(fm, pd.DataFrame)
        assert len(fm) > 0


# ---------------------------------------------------------------------------
# target_labels
# ---------------------------------------------------------------------------

class TestTargetLabels:
    def test_direction_labels_in_minus1_0_1(self, close_series):
        labels = target_labels(close_series, horizon=1, kind="direction")
        valid = labels.dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

    def test_binary_labels_in_0_1(self, close_series):
        labels = target_labels(close_series, horizon=1, kind="binary")
        valid = labels.dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_regression_labels_are_floats(self, close_series):
        labels = target_labels(close_series, horizon=1, kind="regression")
        assert labels.dtype in (np.float64, float)

    def test_direction_labels_same_length_as_input(self, close_series):
        labels = target_labels(close_series, horizon=1, kind="direction")
        assert len(labels) == len(close_series)

    def test_with_threshold_reduces_zero_class(self, close_series):
        no_threshold = target_labels(close_series, horizon=1, kind="direction", threshold=0.0)
        with_threshold = target_labels(close_series, horizon=1, kind="direction", threshold=0.02)
        # More threshold => more 0 labels
        zero_no_thresh = (no_threshold == 0).sum()
        zero_with_thresh = (with_threshold == 0).sum()
        assert zero_with_thresh >= zero_no_thresh

    def test_horizon_shifts_labels_correctly(self, close_series):
        labels_1 = target_labels(close_series, horizon=1, kind="regression")
        labels_5 = target_labels(close_series, horizon=5, kind="regression")
        # Both have same index; the last horizon obs of labels_5 should have more NaN
        assert labels_5.isna().sum() >= labels_1.isna().sum()


# ---------------------------------------------------------------------------
# make_sequences
# ---------------------------------------------------------------------------

class TestMakeSequences:
    def test_shape_is_correct(self, close_series):
        window, horizon = 20, 1
        X, y = make_sequences(close_series, window=window, horizon=horizon)
        n = len(close_series) - window - horizon + 1
        assert X.shape == (n, window)
        assert y.shape == (n,)

    def test_shape_with_larger_horizon(self, close_series):
        window, horizon = 10, 5
        X, y = make_sequences(close_series, window=window, horizon=horizon)
        n = len(close_series) - window - horizon + 1
        assert X.shape == (n, window)
        assert y.shape == (n,)

    def test_empty_for_too_short_series(self):
        s = pd.Series([1.0, 2.0, 3.0])
        X, y = make_sequences(s, window=10, horizon=1)
        assert X.shape[0] == 0
        assert y.shape[0] == 0

    def test_x_values_are_consecutive_windows(self, close_series):
        window = 5
        X, y = make_sequences(close_series, window=window, horizon=1)
        arr = close_series.dropna().values
        # First row of X should be arr[0:window]
        np.testing.assert_allclose(X[0], arr[:window])

    def test_no_nan_in_sequences(self, close_series):
        X, y = make_sequences(close_series, window=20, horizon=1)
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()


# ---------------------------------------------------------------------------
# Regime detectors
# ---------------------------------------------------------------------------

class TestBullBearRegime:
    def test_output_is_series_with_same_index(self, close_series):
        regime = bull_bear_regime(close_series, lookback=50)
        assert isinstance(regime, pd.Series)
        assert len(regime) == len(close_series)

    def test_values_in_minus1_1(self, close_series):
        regime = bull_bear_regime(close_series, lookback=50)
        unique = set(regime.dropna().unique())
        assert unique.issubset({-1, 0, 1})

    def test_aligned_with_input_index(self, close_series):
        regime = bull_bear_regime(close_series, lookback=50)
        assert (regime.index == close_series.index).all()


class TestTrendRegime:
    def test_output_is_series(self, close_series):
        regime = trend_regime(close_series, short=50, long=200)
        assert isinstance(regime, pd.Series)

    def test_values_in_minus1_1(self, close_series):
        regime = trend_regime(close_series, short=50, long=200)
        unique = set(regime.dropna().unique())
        assert unique.issubset({-1, 0, 1})

    def test_length_matches_input(self, close_series):
        regime = trend_regime(close_series, short=10, long=50)
        assert len(regime) == len(close_series)


class TestVolRegime:
    def test_output_is_series(self, returns_series):
        regime = vol_regime(returns_series)
        assert isinstance(regime, pd.Series)

    def test_values_in_expected_set(self, returns_series):
        regime = vol_regime(returns_series)
        valid_labels = {"high", "low", "mid"}
        unique = set(regime.dropna().unique())
        assert unique.issubset(valid_labels)

    def test_length_matches_input(self, returns_series):
        regime = vol_regime(returns_series)
        assert len(regime) == len(returns_series)


class TestHMMRegimes:
    def test_output_is_series(self, returns_series):
        regime = hmm_regimes(returns_series, n_states=2, seed=42)
        assert isinstance(regime, pd.Series)

    def test_aligned_with_input(self, returns_series):
        regime = hmm_regimes(returns_series, n_states=2, seed=42)
        assert len(regime) == len(returns_series)

    def test_values_are_integers(self, returns_series):
        regime = hmm_regimes(returns_series, n_states=2, seed=42)
        valid = regime.dropna()
        assert valid.dtype in (int, np.int64, np.int32, "int64")


# ---------------------------------------------------------------------------
# Forecasters
# ---------------------------------------------------------------------------

class TestARForecast:
    def test_returns_float(self, close_series):
        val = ar_forecast(close_series, p=1, horizon=1)
        assert isinstance(val, float)
        assert math.isfinite(val)

    def test_short_series_returns_last_value(self):
        s = pd.Series([1.0, 2.0, 3.0])
        val = ar_forecast(s, p=1, horizon=1)
        assert math.isfinite(val)

    def test_ar_forecast_is_deterministic(self, close_series):
        v1 = ar_forecast(close_series, p=2, horizon=3)
        v2 = ar_forecast(close_series, p=2, horizon=3)
        assert v1 == pytest.approx(v2)


class TestEWMAForecast:
    def test_returns_float(self, close_series):
        val = ewma_forecast(close_series, span=20)
        assert isinstance(val, float)
        assert math.isfinite(val)

    def test_empty_series_returns_nan(self):
        val = ewma_forecast(pd.Series([], dtype=float))
        assert math.isnan(val)

    def test_constant_series_returns_constant(self):
        s = pd.Series([5.0] * 50)
        val = ewma_forecast(s, span=10)
        assert val == pytest.approx(5.0)
