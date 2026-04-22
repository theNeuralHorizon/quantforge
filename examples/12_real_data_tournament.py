"""Example 12: Real-data tournament — SPY/QQQ/IWM/TLT/GLD 2015-2025.

Tests all 8 strategies + Buy-and-Hold on 10 years of actual ETF prices.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from quantforge.analytics.benchmark import benchmark_report
from quantforge.analytics.performance import summary_stats
from quantforge.analytics.tearsheet import tearsheet_text
from quantforge.backtest import BacktestEngine, FixedBpsCommission, FixedBpsSlippage
from quantforge.data.loader import DataLoader
from quantforge.strategies import (
    BollingerMeanReversion, BuyAndHoldStrategy, DonchianBreakout,
    FactorStrategy, MACrossoverStrategy, MomentumStrategy,
    RSIReversalStrategy,
)


def main() -> None:
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
    print(f"Loading real market data ({', '.join(tickers)}) 2015-2025...")
    dl = DataLoader(cache_dir="data/cache")
    panel = dl.yfinance_many(tickers, start="2015-01-01", end="2025-01-01")
    n_bars = min(len(v) for v in panel.values())
    print(f"  Got {n_bars} bars across {len(panel)} tickers")

    strategies = {
        "Buy & Hold SPY": (BuyAndHoldStrategy(), {"SPY": panel["SPY"]}),
        "Buy & Hold 5-asset": (BuyAndHoldStrategy(), panel),
        "Mom(60)": (MomentumStrategy(lookback=60, allow_short=False), panel),
        "Mom(120)": (MomentumStrategy(lookback=120, allow_short=False), panel),
        "MA 20/100": (MACrossoverStrategy(fast=20, slow=100), panel),
        "MA 50/200": (MACrossoverStrategy(fast=50, slow=200), panel),
        "RSI Reversal": (RSIReversalStrategy(oversold=30, overbought=70, allow_short=False), panel),
        "Bollinger MR": (BollingerMeanReversion(window=20, k=2.0), panel),
        "Donchian": (DonchianBreakout(entry_window=20, exit_window=10, allow_short=False), panel),
        "Factor": (FactorStrategy(lookback_mom=120, lookback_vol=60, top_q=0.4), panel),
    }

    print("\nRunning backtests (real data, 10bps slippage, 1bps commission, monthly rebalance)...")
    rows = []
    equity_curves = {}
    benchmark_returns = None

    for name, (strat, data) in strategies.items():
        t0 = time.perf_counter()
        eng = BacktestEngine(
            strategy=strat,
            data=data,
            initial_capital=100_000,
            sizing_fraction=0.6 if len(data) == 1 else 0.2,
            broker=None,  # default: FixedBpsSlippage=5bps, FixedBpsCommission=1bp
            rebalance="monthly" if "Buy" not in name else "bar",
        )
        res = eng.run()
        elapsed = time.perf_counter() - t0
        stats = summary_stats(res.equity_curve, trades=res.trades)
        stats["runtime_s"] = elapsed
        rows.append({"strategy": name, **stats})
        equity_curves[name] = res.equity_curve
        if name == "Buy & Hold SPY":
            benchmark_returns = res.equity_curve.pct_change().dropna()

    df = pd.DataFrame(rows).set_index("strategy")

    print("\n" + "=" * 90)
    print("TOURNAMENT RESULTS (real data, 2015-2025, cost-aware)")
    print("=" * 90)
    keep = ["total_return", "annual_return", "annual_vol", "sharpe",
            "sortino", "max_drawdown", "turnover", "cost_bps", "n_trades", "runtime_s"]
    keep = [k for k in keep if k in df.columns]
    print(df[keep].round(3).to_string())

    # Benchmark comparison (vs buy-and-hold SPY)
    print("\n" + "=" * 90)
    print("ALPHA / BETA / INFO RATIO vs Buy & Hold SPY")
    print("=" * 90)
    benchmark_rows = []
    for name, curve in equity_curves.items():
        if name == "Buy & Hold SPY":
            continue
        r = curve.pct_change().dropna()
        rep = benchmark_report(r, benchmark_returns)
        benchmark_rows.append({
            "strategy": name,
            "alpha": rep.alpha,
            "beta": rep.beta,
            "info_ratio": rep.info_ratio,
            "tracking_error": rep.tracking_error,
            "up_capture": rep.up_capture,
            "down_capture": rep.down_capture,
        })
    bdf = pd.DataFrame(benchmark_rows).set_index("strategy")
    print(bdf.round(3).to_string())

    # Rankings
    print("\n" + "=" * 90)
    print("RANKINGS")
    print("=" * 90)
    print(f"  Best Sharpe:         {df['sharpe'].idxmax():>25}  ({df['sharpe'].max():+.2f})")
    print(f"  Best Total Return:   {df['total_return'].idxmax():>25}  ({df['total_return'].max()*100:+.1f}%)")
    print(f"  Smallest Drawdown:   {df['max_drawdown'].idxmax():>25}  ({df['max_drawdown'].max()*100:+.1f}%)")
    print(f"  Highest Info Ratio:  {bdf['info_ratio'].idxmax():>25}  ({bdf['info_ratio'].max():+.2f})")
    print(f"  Highest Alpha:       {bdf['alpha'].idxmax():>25}  ({bdf['alpha'].max()*100:+.2f}%)")

    # Save
    import os
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/real_tournament_stats.csv")
    bdf.to_csv("results/real_tournament_benchmark.csv")
    eq_out = pd.DataFrame(equity_curves).ffill()
    eq_out.to_csv("results/real_tournament_equity.csv")
    print(f"\nSaved artifacts to ./results/real_tournament_*.csv ({len(eq_out)} bars)")


if __name__ == "__main__":
    main()
