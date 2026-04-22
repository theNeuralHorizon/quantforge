"""Example 04: Compare multiple strategies on identical synthetic data."""
from __future__ import annotations

import pandas as pd

from quantforge.backtest import BacktestEngine
from quantforge.data.synthetic import generate_panel
from quantforge.analytics.performance import summary_stats
from quantforge.strategies import (
    MomentumStrategy, MACrossoverStrategy, RSIReversalStrategy,
    BollingerMeanReversion, DonchianBreakout, FactorStrategy,
)


def _run(name: str, strat, panel: dict, capital: float = 100_000) -> dict:
    engine = BacktestEngine(strategy=strat, data=panel, initial_capital=capital, sizing_fraction=0.25)
    res = engine.run()
    s = summary_stats(res.equity_curve)
    return {"strategy": name, **s, "n_trades": len(res.trades), "final_equity": float(res.equity_curve.iloc[-1])}


def main() -> None:
    panel = generate_panel(["AAA", "BBB", "CCC", "DDD"], n=600, seed=11)

    rows = []
    rows.append(_run("Momentum(60)", MomentumStrategy(lookback=60, allow_short=False), panel))
    rows.append(_run("Momentum(120)", MomentumStrategy(lookback=120, allow_short=True), panel))
    rows.append(_run("MA 10/30", MACrossoverStrategy(fast=10, slow=30), panel))
    rows.append(_run("MA 20/100", MACrossoverStrategy(fast=20, slow=100), panel))
    rows.append(_run("RSI Reversal", RSIReversalStrategy(oversold=30, overbought=70), panel))
    rows.append(_run("Bollinger MR", BollingerMeanReversion(window=20, k=2.0), panel))
    rows.append(_run("Donchian", DonchianBreakout(entry_window=20, exit_window=10), panel))
    rows.append(_run("Factor Composite", FactorStrategy(lookback_mom=120, lookback_vol=60), panel))

    df = pd.DataFrame(rows).set_index("strategy")
    keep = ["total_return", "annual_return", "annual_vol", "sharpe", "sortino",
            "calmar", "max_drawdown", "n_trades"]
    print("=" * 80)
    print("Strategy comparison on synthetic 4-asset panel, 600 bars, seed=11")
    print("=" * 80)
    print(df[keep].round(4).to_string())

    print("\nBest Sharpe:", df["sharpe"].idxmax())
    print("Lowest DD  :", df["max_drawdown"].idxmax())


if __name__ == "__main__":
    main()
