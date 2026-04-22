"""Tests for TCA."""
from __future__ import annotations

import pandas as pd
import pytest

from quantforge.backtest.tca import analyze_trades


class TestTCA:
    def test_empty_trades(self):
        r = analyze_trades(pd.DataFrame())
        assert r.n_trades == 0
        assert r.total_commission == 0

    def test_basic_tca(self):
        trades = pd.DataFrame({
            "qty": [100, -50],
            "price": [100.0, 101.0],
            "commission": [1.0, 0.5],
            "slippage": [0.05, 0.02],
        })
        r = analyze_trades(trades)
        assert r.n_trades == 2
        assert r.total_notional == pytest.approx(100 * 100.0 + 50 * 101.0)
        assert r.total_commission == pytest.approx(1.5)
        assert r.total_slippage_cost == pytest.approx(100 * 0.05 + 50 * 0.02)

    def test_bps_conversion(self):
        trades = pd.DataFrame({
            "qty": [100],
            "price": [100.0],
            "commission": [0.5],
            "slippage": [0.05],
        })
        r = analyze_trades(trades)
        # notional=10000, total cost = (100*0.05 + 0.5) = 5.5, bps = 5.5/10000 * 10000 = 5.5 bps
        assert r.avg_bps == pytest.approx(5.5)

    def test_summary_string_contains_stats(self):
        trades = pd.DataFrame({"qty": [100], "price": [100.0], "commission": [1.0], "slippage": [0.01]})
        out = analyze_trades(trades).summary()
        assert "Trades" in out and "Commission" in out
