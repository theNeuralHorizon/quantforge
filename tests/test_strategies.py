"""Tests for quantforge.strategies: signal validity, warmup, direction constraints."""
import pandas as pd

from quantforge.core.event import EventType, SignalEvent
from quantforge.data.synthetic import generate_ohlcv
from quantforge.strategies.ma_crossover import MACrossoverStrategy
from quantforge.strategies.mean_reversion import BollingerMeanReversion
from quantforge.strategies.momentum import MomentumStrategy
from quantforge.strategies.pairs_trading import PairsTradingStrategy
from quantforge.strategies.rsi_reversal import RSIReversalStrategy
from quantforge.strategies.trend_breakout import DonchianBreakout

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ohlcv(n: int, seed: int) -> pd.DataFrame:
    return generate_ohlcv(n=n, seed=seed)


def assert_valid_signals(signals, symbol: str):
    """Check that a list of signals is either empty or contains valid SignalEvents."""
    assert isinstance(signals, list)
    for sig in signals:
        assert isinstance(sig, SignalEvent)
        assert sig.event_type == EventType.SIGNAL
        assert sig.direction in (-1, 0, 1)
        assert 0.0 <= sig.strength <= 1.0
        assert sig.symbol == symbol


# ---------------------------------------------------------------------------
# MomentumStrategy
# ---------------------------------------------------------------------------

class TestMomentumStrategy:
    def test_warmup_returns_positive_int(self):
        strat = MomentumStrategy(lookback=60)
        assert strat.warmup() > 0
        assert isinstance(strat.warmup(), int)

    def test_returns_empty_list_with_insufficient_history(self):
        strat = MomentumStrategy(lookback=60)
        df = make_ohlcv(30, seed=1)
        bar = df.iloc[-1]
        bar.name = df.index[-1]
        signals = strat.on_bar("AAPL", bar, df)
        assert signals == []

    def test_returns_valid_signals_with_enough_history(self):
        strat = MomentumStrategy(lookback=20)
        df = make_ohlcv(100, seed=1)
        bar = df.iloc[-1]
        bar.name = df.index[-1]
        signals = strat.on_bar("AAPL", bar, df)
        assert_valid_signals(signals, "AAPL")

    def test_direction_1_when_price_rose_above_threshold(self):
        strat = MomentumStrategy(lookback=10, threshold=0.0, allow_short=True)
        # Create a series where price strictly rises over lookback
        idx = pd.bdate_range("2020-01-01", periods=50)
        prices = pd.Series(range(100, 150), index=idx, dtype=float)
        df = pd.DataFrame({
            "open": prices, "high": prices + 1, "low": prices - 1,
            "close": prices, "volume": 1e6,
        })
        bar = df.iloc[-1]
        bar.name = df.index[-1]
        signals = strat.on_bar("X", bar, df)
        assert len(signals) == 1
        assert signals[0].direction == 1

    def test_no_short_when_allow_short_false(self):
        strat = MomentumStrategy(lookback=10, threshold=0.0, allow_short=False)
        # Create declining price series
        idx = pd.bdate_range("2020-01-01", periods=50)
        prices = pd.Series(range(150, 100, -1), index=idx, dtype=float)
        df = pd.DataFrame({
            "open": prices, "high": prices + 1, "low": prices - 1,
            "close": prices, "volume": 1e6,
        })
        bar = df.iloc[-1]
        bar.name = df.index[-1]
        signals = strat.on_bar("X", bar, df)
        for sig in signals:
            assert sig.direction in (0, 1)  # never -1


# ---------------------------------------------------------------------------
# MACrossoverStrategy
# ---------------------------------------------------------------------------

class TestMACrossover:
    def test_warmup_exceeds_slow_window(self):
        strat = MACrossoverStrategy(fast=10, slow=30)
        assert strat.warmup() > 30

    def test_returns_empty_with_insufficient_history(self):
        strat = MACrossoverStrategy(fast=5, slow=20)
        df = make_ohlcv(10, seed=2)
        bar = df.iloc[-1]
        bar.name = df.index[-1]
        signals = strat.on_bar("SPY", bar, df)
        assert signals == []

    def test_signals_valid_with_enough_history(self):
        strat = MACrossoverStrategy(fast=5, slow=20)
        df = make_ohlcv(100, seed=3)
        bar = df.iloc[-1]
        bar.name = df.index[-1]
        signals = strat.on_bar("SPY", bar, df)
        assert_valid_signals(signals, "SPY")


# ---------------------------------------------------------------------------
# BollingerMeanReversion
# ---------------------------------------------------------------------------

class TestBollingerMeanReversion:
    def test_warmup_positive(self):
        strat = BollingerMeanReversion(window=20)
        assert strat.warmup() > 0

    def test_signals_valid(self):
        strat = BollingerMeanReversion(window=20)
        df = make_ohlcv(100, seed=4)
        bar = df.iloc[-1]
        bar.name = df.index[-1]
        signals = strat.on_bar("TEST", bar, df)
        assert_valid_signals(signals, "TEST")

    def test_returns_empty_insufficient_history(self):
        strat = BollingerMeanReversion(window=20)
        df = make_ohlcv(10, seed=5)
        bar = df.iloc[-1]
        bar.name = df.index[-1]
        signals = strat.on_bar("TEST", bar, df)
        assert signals == []


# ---------------------------------------------------------------------------
# DonchianBreakout
# ---------------------------------------------------------------------------

class TestDonchianBreakout:
    def test_warmup_positive(self):
        strat = DonchianBreakout(entry_window=20, exit_window=10)
        assert strat.warmup() > 0

    def test_signals_valid_with_enough_history(self):
        strat = DonchianBreakout(entry_window=20, exit_window=10)
        df = make_ohlcv(100, seed=6)
        bar = df.iloc[-1]
        bar.name = df.index[-1]
        signals = strat.on_bar("QQQ", bar, df)
        assert_valid_signals(signals, "QQQ")

    def test_direction_in_valid_set(self):
        strat = DonchianBreakout(entry_window=20, allow_short=True)
        df = make_ohlcv(150, seed=7)
        bar = df.iloc[-1]
        bar.name = df.index[-1]
        signals = strat.on_bar("QQQ", bar, df)
        for sig in signals:
            assert sig.direction in (-1, 0, 1)


# ---------------------------------------------------------------------------
# RSIReversalStrategy
# ---------------------------------------------------------------------------

class TestRSIReversal:
    def test_warmup_positive(self):
        strat = RSIReversalStrategy(window=14)
        assert strat.warmup() > 0

    def test_signals_valid(self):
        strat = RSIReversalStrategy(window=14)
        df = make_ohlcv(100, seed=8)
        bar = df.iloc[-1]
        bar.name = df.index[-1]
        signals = strat.on_bar("GLD", bar, df)
        assert_valid_signals(signals, "GLD")

    def test_returns_empty_insufficient_history(self):
        strat = RSIReversalStrategy(window=14)
        df = make_ohlcv(5, seed=9)
        bar = df.iloc[-1]
        bar.name = df.index[-1]
        signals = strat.on_bar("GLD", bar, df)
        assert signals == []

    def test_long_signal_when_rsi_oversold(self):
        strat = RSIReversalStrategy(window=14, oversold=30.0)
        # Force oversold: series that drops sharply
        n = 60
        idx = pd.bdate_range("2020-01-01", periods=n)
        prices = pd.Series([100.0 - i * 0.8 for i in range(n)], index=idx)
        df = pd.DataFrame({
            "open": prices, "high": prices * 1.005, "low": prices * 0.995,
            "close": prices, "volume": 1e6,
        })
        bar = df.iloc[-1]
        bar.name = df.index[-1]
        signals = strat.on_bar("GLD", bar, df)
        if signals:
            assert signals[0].direction == 1


# ---------------------------------------------------------------------------
# PairsTradingStrategy — basic wiring test
# ---------------------------------------------------------------------------

class TestPairsTrading:
    def test_warmup_positive(self):
        strat = PairsTradingStrategy(asset_a="A", asset_b="B", window=60)
        assert strat.warmup() > 0

    def test_no_signal_with_one_leg(self):
        strat = PairsTradingStrategy(asset_a="A", asset_b="B", window=60)
        df = make_ohlcv(150, seed=10)
        bar = df.iloc[-1]
        bar.name = df.index[-1]
        # Only feeding asset A — should produce no signals (needs B too)
        signals = strat.on_bar("A", bar, df)
        assert signals == []

    def test_signals_after_both_legs_fed(self):
        strat = PairsTradingStrategy(asset_a="A", asset_b="B", window=40)
        df_a = make_ohlcv(150, seed=11)
        df_b = make_ohlcv(150, seed=12)

        for i in range(len(df_a)):
            bar_a = df_a.iloc[i]
            bar_a_named = bar_a.copy()
            bar_a_named.name = df_a.index[i]
            strat.on_bar("A", bar_a_named, df_a.iloc[:i+1])

            bar_b = df_b.iloc[i]
            bar_b_named = bar_b.copy()
            bar_b_named.name = df_b.index[i]
            signals = strat.on_bar("B", bar_b_named, df_b.iloc[:i+1])

        # After full history, signals are list (possibly empty if no crossings occurred)
        assert isinstance(signals, list)
