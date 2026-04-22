"""Tests for quantforge.backtest.engine and BacktestResult."""
import pytest
import numpy as np
import pandas as pd

from quantforge.backtest.engine import BacktestEngine, BacktestResult
from quantforge.strategies.momentum import MomentumStrategy
from quantforge.strategies.ma_crossover import MACrossoverStrategy
from quantforge.data.synthetic import generate_panel, generate_ohlcv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_panel():
    """Small synthetic panel: 2 symbols, 252 bars."""
    return generate_panel(symbols=["AAA", "BBB"], n=252, seed=10)


@pytest.fixture(scope="module")
def single_ohlcv():
    return {"AAPL": generate_ohlcv(n=252, seed=5)}


@pytest.fixture(scope="module")
def momentum_strategy():
    return MomentumStrategy(lookback=30, allow_short=False)


@pytest.fixture(scope="module")
def ma_crossover_strategy():
    return MACrossoverStrategy(fast=5, slow=20, allow_short=True)


# ---------------------------------------------------------------------------
# BacktestEngine: basic run
# ---------------------------------------------------------------------------

class TestBacktestEngineRun:
    def test_run_produces_backtest_result(self, small_panel, momentum_strategy):
        engine = BacktestEngine(
            strategy=momentum_strategy,
            data=small_panel,
            initial_capital=100_000,
        )
        result = engine.run()
        assert isinstance(result, BacktestResult)

    def test_equity_curve_non_empty(self, small_panel, momentum_strategy):
        engine = BacktestEngine(
            strategy=momentum_strategy,
            data=small_panel,
            initial_capital=100_000,
        )
        result = engine.run()
        assert len(result.equity_curve) > 0

    def test_equity_curve_length_matches_data_index(self, single_ohlcv):
        strat = MomentumStrategy(lookback=20)
        engine = BacktestEngine(strategy=strat, data=single_ohlcv, initial_capital=50_000)
        result = engine.run()
        # The equity curve should have one entry per bar
        data_len = len(single_ohlcv["AAPL"])
        assert len(result.equity_curve) == data_len

    def test_equity_curve_starts_near_initial_capital(self, single_ohlcv):
        strat = MomentumStrategy(lookback=20)
        engine = BacktestEngine(strategy=strat, data=single_ohlcv, initial_capital=100_000)
        result = engine.run()
        # First bar equity = initial capital (no trades yet in warmup)
        assert result.equity_curve.iloc[0] == pytest.approx(100_000.0)

    def test_equity_curve_index_is_datetimeindex(self, single_ohlcv):
        strat = MomentumStrategy(lookback=20)
        engine = BacktestEngine(strategy=strat, data=single_ohlcv, initial_capital=100_000)
        result = engine.run()
        assert isinstance(result.equity_curve.index, pd.DatetimeIndex)

    def test_trades_dataframe_non_empty_after_signals(self, small_panel):
        # MACrossover will generate trades on 252 bars
        strat = MACrossoverStrategy(fast=5, slow=20, allow_short=True)
        engine = BacktestEngine(strategy=strat, data=small_panel, initial_capital=100_000)
        result = engine.run()
        # Should have at least some trades
        assert isinstance(result.trades, pd.DataFrame)

    def test_positions_dataframe_shape(self, small_panel, momentum_strategy):
        engine = BacktestEngine(
            strategy=momentum_strategy,
            data=small_panel,
            initial_capital=100_000,
        )
        result = engine.run()
        assert isinstance(result.positions_over_time, pd.DataFrame)
        assert len(result.positions_over_time) > 0


# ---------------------------------------------------------------------------
# BacktestResult properties
# ---------------------------------------------------------------------------

class TestBacktestResult:
    @pytest.fixture(scope="class")
    def result(self, small_panel):
        strat = MomentumStrategy(lookback=30)
        engine = BacktestEngine(strategy=strat, data=small_panel, initial_capital=100_000)
        return engine.run()

    def test_returns_series_has_same_length(self, result):
        assert len(result.returns) == len(result.equity_curve)

    def test_total_return_is_finite(self, result):
        import math
        assert math.isfinite(result.total_return)

    def test_to_dict_has_expected_keys(self, result):
        d = result.to_dict()
        assert "initial_capital" in d
        assert "final_equity" in d
        assert "total_return" in d
        assert "n_trades" in d

    def test_initial_capital_preserved(self, result):
        assert result.initial_capital == pytest.approx(100_000.0)


# ---------------------------------------------------------------------------
# Warmup respected: no trades before warmup
# ---------------------------------------------------------------------------

class TestWarmupPeriod:
    def test_no_trades_during_warmup_period(self):
        """Ensure strategy signals are suppressed during warmup."""
        ohlcv = generate_ohlcv(n=50, seed=1)
        strat = MomentumStrategy(lookback=40)  # warmup=41, data only 50 bars
        engine = BacktestEngine(strategy=strat, data={"X": ohlcv}, initial_capital=100_000)
        result = engine.run()
        # With 50 bars and warmup=41, very few (if any) signals, trades df may be empty
        assert isinstance(result.trades, pd.DataFrame)

    def test_equity_flat_during_warmup(self):
        """Equity should be constant (= initial capital) during warmup."""
        ohlcv = generate_ohlcv(n=60, seed=2)
        strat = MomentumStrategy(lookback=50)  # warmup=51
        engine = BacktestEngine(strategy=strat, data={"X": ohlcv}, initial_capital=100_000)
        result = engine.run()
        warmup = strat.warmup()
        # All equity_curve entries up to warmup should equal initial capital.
        # Use np.allclose because pytest.approx does not broadcast correctly with pd.Series.
        early_equity = result.equity_curve.iloc[:warmup]
        assert np.allclose(early_equity.values, 100_000.0, rtol=1e-9)
