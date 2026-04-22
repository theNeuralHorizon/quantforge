"""Example 10: The full research pipeline end-to-end.

1. Generate multi-asset synthetic universe
2. Engineer features + regimes
3. Backtest multiple strategies
4. Build a multi-strategy portfolio via HRP on strategy returns
5. Risk analysis on the combined portfolio
6. Save tearsheet + equity curves to disk
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.analytics.performance import summary_stats
from quantforge.analytics.tearsheet import tearsheet_markdown, tearsheet_text
from quantforge.backtest import BacktestEngine
from quantforge.data.synthetic import generate_panel
from quantforge.ml.regime import bull_bear_regime, vol_regime
from quantforge.portfolio.hrp import hierarchical_risk_parity
from quantforge.risk.stress_test import stress_scenarios
from quantforge.strategies import (
    BollingerMeanReversion, DonchianBreakout, FactorStrategy,
    MACrossoverStrategy, MomentumStrategy, RSIReversalStrategy,
)


def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    print("[1/6] Generating synthetic universe (6 assets, 504 bars)")
    panel = generate_panel(["TECH", "HEALTH", "ENERGY", "BONDS", "GOLD", "CRYPTO"], n=504, seed=2026)

    print("[2/6] Running 6 strategies")
    strategies = {
        "Mom(120)": MomentumStrategy(lookback=120, allow_short=True),
        "MA(20/100)": MACrossoverStrategy(20, 100),
        "BB MR": BollingerMeanReversion(20, 2.0),
        "Donchian": DonchianBreakout(20, 10),
        "RSI": RSIReversalStrategy(),
        "Factor": FactorStrategy(120, 60),
    }

    equity_frames = {}
    per_strategy_stats = []
    for name, strat in strategies.items():
        eng = BacktestEngine(strategy=strat, data=panel, initial_capital=100_000, sizing_fraction=0.2)
        res = eng.run()
        equity_frames[name] = res.equity_curve
        stats = summary_stats(res.equity_curve)
        per_strategy_stats.append({"strategy": name, **stats, "trades": len(res.trades)})

    eq_df = pd.DataFrame(equity_frames).ffill().dropna()
    strat_returns = eq_df.pct_change().dropna()

    print("[3/6] Computing per-strategy summary stats")
    stats_df = pd.DataFrame(per_strategy_stats).set_index("strategy")
    print(stats_df[["total_return", "sharpe", "max_drawdown", "trades"]].round(4))

    print("\n[4/6] Combining via Hierarchical Risk Parity on strategy returns")
    cov = strat_returns.cov().values * 252
    weights = hierarchical_risk_parity(cov)
    w = pd.Series(weights, index=strat_returns.columns)
    print("  HRP weights:")
    print(w.round(3).to_string())

    combined_returns = strat_returns @ w.values
    combined_equity = 100_000 * (1 + combined_returns).cumprod()
    combined_equity = pd.concat([pd.Series([100_000.0], index=[eq_df.index[0]]), combined_equity])

    print("\n[5/6] Risk analysis on combined portfolio")
    print(tearsheet_text(combined_equity, "HRP Combined"))

    print("\n  Stress tests (treating everything as equity):")
    asset_map = {n: "equity" for n in w.index}
    stress_w = w.copy()
    print(stress_scenarios(stress_w, asset_map).to_string(index=False))

    print("\n[6/6] Saving artifacts to ./results/")
    eq_df.to_csv(out_dir / "per_strategy_equity.csv")
    pd.Series(combined_equity, name="combined").to_csv(out_dir / "combined_equity.csv")
    stats_df.to_csv(out_dir / "per_strategy_stats.csv")
    (out_dir / "tearsheet.md").write_text(tearsheet_markdown(combined_equity, "HRP Combined"))
    print(f"  -> {out_dir.resolve()}")
    print("\nDONE")


if __name__ == "__main__":
    main()
