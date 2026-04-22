"""Example 08: Walk-forward validation of a strategy.

Splits data into N sequential folds, runs backtest on each, and reports
out-of-sample robustness.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from quantforge.analytics.performance import summary_stats
from quantforge.backtest import BacktestEngine
from quantforge.data.synthetic import generate_panel
from quantforge.strategies import MACrossoverStrategy, MomentumStrategy


def walk_forward(strategy_factory, data: dict, n_folds: int = 5) -> pd.DataFrame:
    symbols = list(data.keys())
    shared_index = data[symbols[0]].index
    fold_size = len(shared_index) // n_folds
    rows: List[dict] = []

    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else len(shared_index)
        fold = {s: df.iloc[start:end] for s, df in data.items()}
        engine = BacktestEngine(strategy=strategy_factory(), data=fold, initial_capital=100_000, sizing_fraction=0.25)
        res = engine.run()
        stats = summary_stats(res.equity_curve)
        rows.append({
            "fold": i + 1,
            "start": fold[symbols[0]].index[0].date(),
            "end": fold[symbols[0]].index[-1].date(),
            "return": stats["total_return"],
            "sharpe": stats["sharpe"],
            "max_dd": stats["max_drawdown"],
            "trades": len(res.trades),
        })
    return pd.DataFrame(rows)


def main() -> None:
    panel = generate_panel(["A", "B", "C"], n=1260, seed=99)

    print("=" * 70)
    print("Walk-forward: MA Crossover (20/100)")
    print("=" * 70)
    wf = walk_forward(lambda: MACrossoverStrategy(fast=20, slow=100), panel, n_folds=5)
    print(wf.to_string(index=False))
    print(f"  Mean Sharpe: {wf['sharpe'].mean():+.3f}")
    print(f"  Sharpe std : {wf['sharpe'].std():+.3f}")
    print(f"  Hit rate   : {(wf['return'] > 0).mean() * 100:.1f}%")

    print("\n" + "=" * 70)
    print("Walk-forward: Momentum(120)")
    print("=" * 70)
    wf2 = walk_forward(lambda: MomentumStrategy(lookback=120, allow_short=True), panel, n_folds=5)
    print(wf2.to_string(index=False))
    print(f"  Mean Sharpe: {wf2['sharpe'].mean():+.3f}")
    print(f"  Sharpe std : {wf2['sharpe'].std():+.3f}")
    print(f"  Hit rate   : {(wf2['return'] > 0).mean() * 100:.1f}%")

    print("\n" + "=" * 70)
    print("Parameter sweep: MA Crossover")
    print("=" * 70)
    sweep = []
    for fast in [5, 10, 20]:
        for slow in [20, 50, 100, 200]:
            if fast >= slow:
                continue
            wf = walk_forward(lambda f=fast, s=slow: MACrossoverStrategy(f, s), panel, n_folds=5)
            sweep.append({
                "fast": fast, "slow": slow,
                "mean_sharpe": wf["sharpe"].mean(),
                "sharpe_std": wf["sharpe"].std(),
                "mean_return": wf["return"].mean(),
                "mean_dd": wf["max_dd"].mean(),
            })
    sdf = pd.DataFrame(sweep).sort_values("mean_sharpe", ascending=False)
    print(sdf.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
