"""Example 15: Improved tournament on real data with the new strategies."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from quantforge.analytics.benchmark import benchmark_report
from quantforge.analytics.performance import summary_stats
from quantforge.backtest import BacktestEngine
from quantforge.data.loader import DataLoader
from quantforge.strategies import (
    BollingerMeanReversion, BuyAndHoldStrategy, CrossSectionalMomentum,
    DonchianBreakout, DualMomentum, FactorStrategy, MACrossoverStrategy,
    MomentumStrategy, RegimeSwitch, RSIReversalStrategy, VolTarget,
)


def main() -> None:
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
    print(f"Loading {tickers} 2015-2025...")
    dl = DataLoader(cache_dir="data/cache")
    panel = dl.yfinance_many(tickers, "2015-01-01", "2025-01-01")
    print(f"  {min(len(v) for v in panel.values())} bars")

    strategies = {
        "Buy & Hold SPY": (BuyAndHoldStrategy(), {"SPY": panel["SPY"]}, 1.0, "bar"),
        "Mom(120)": (MomentumStrategy(120, allow_short=False), panel, 0.2, "monthly"),
        "MA 20/100": (MACrossoverStrategy(20, 100), panel, 0.2, "monthly"),
        "Donchian": (DonchianBreakout(20, 10, allow_short=False), panel, 0.2, "monthly"),
        "Bollinger MR": (BollingerMeanReversion(20, 2.0), panel, 0.2, "monthly"),
        "Factor": (FactorStrategy(120, 60, top_q=0.4), panel, 0.2, "monthly"),
        # NEW strategies
        "Cross-Sec Mom": (CrossSectionalMomentum(lookback=252, skip=21, top_q=0.4), panel, 0.25, "monthly"),
        "Dual Momentum": (DualMomentum(lookback=252, cash_asset="TLT"), panel, 1.0, "monthly"),
        "Regime Switch": (RegimeSwitch(risk_on_asset="SPY", risk_off_asset="TLT", signal_asset="SPY"),
                          panel, 1.0, "monthly"),
        "VolTarget(Mom)": (VolTarget(base=MomentumStrategy(120), target_vol=0.10), panel, 0.2, "monthly"),
        "VolTarget(B&H)": (VolTarget(base=BuyAndHoldStrategy(), target_vol=0.10), {"SPY": panel["SPY"]},
                            1.0, "weekly"),
    }

    print("\nRunning tournament (5bps slippage, 1bp commission)...\n")
    rows = []
    equities = {}
    for name, (strat, data, size, rebalance) in strategies.items():
        t0 = time.perf_counter()
        eng = BacktestEngine(strategy=strat, data=data, initial_capital=100_000,
                              sizing_fraction=size, rebalance=rebalance)
        res = eng.run()
        elapsed = time.perf_counter() - t0
        s = summary_stats(res.equity_curve, trades=res.trades)
        rows.append({"strategy": name, **s, "runtime_s": elapsed})
        equities[name] = res.equity_curve

    df = pd.DataFrame(rows).set_index("strategy")
    bench_r = equities["Buy & Hold SPY"].pct_change().dropna()

    print("=" * 100)
    print("RESULTS")
    print("=" * 100)
    keep = ["total_return", "annual_return", "sharpe", "sortino", "max_drawdown",
            "turnover", "cost_bps", "n_trades", "runtime_s"]
    keep = [k for k in keep if k in df.columns]
    print(df[keep].round(3).to_string())

    print("\n" + "=" * 100)
    print("vs Buy & Hold SPY")
    print("=" * 100)
    bench_rows = []
    for name, curve in equities.items():
        if name == "Buy & Hold SPY":
            continue
        r = curve.pct_change().dropna()
        rep = benchmark_report(r, bench_r)
        bench_rows.append({
            "strategy": name, "alpha": rep.alpha, "beta": rep.beta,
            "info_ratio": rep.info_ratio, "tracking_error": rep.tracking_error,
            "up_capture": rep.up_capture, "down_capture": rep.down_capture,
        })
    bdf = pd.DataFrame(bench_rows).set_index("strategy")
    print(bdf.round(3).to_string())

    print("\n" + "=" * 100)
    print("RANKINGS")
    print("=" * 100)
    print(f"  Best Sharpe:         {df['sharpe'].idxmax():>25}  ({df['sharpe'].max():+.2f})")
    print(f"  Best Return:         {df['total_return'].idxmax():>25}  ({df['total_return'].max()*100:+.1f}%)")
    print(f"  Smallest DD:         {df['max_drawdown'].idxmax():>25}  ({df['max_drawdown'].max()*100:+.1f}%)")
    print(f"  Highest Alpha:       {bdf['alpha'].idxmax():>25}  ({bdf['alpha'].max()*100:+.2f}%)")
    print(f"  Highest Info Ratio:  {bdf['info_ratio'].idxmax():>25}  ({bdf['info_ratio'].max():+.2f})")

    import os
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/improved_tournament_stats.csv")
    bdf.to_csv("results/improved_tournament_benchmark.csv")
    pd.DataFrame(equities).ffill().to_csv("results/improved_tournament_equity.csv")
    print(f"\nSaved to ./results/improved_tournament_*.csv")


if __name__ == "__main__":
    main()
