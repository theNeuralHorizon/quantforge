"""Example 13: Full research pipeline on real market data (2015-2025).

1. Load real ETF data (SPY/QQQ/IWM/TLT/GLD)
2. Run 6 strategies on the universe
3. Compute alpha/beta vs SPY
4. Combine strategies via HRP on strategy returns
5. Stress test + full tearsheet
6. Compare to passive 60/40 portfolio
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.analytics.benchmark import benchmark_report
from quantforge.analytics.performance import summary_stats
from quantforge.analytics.tearsheet import tearsheet_markdown, tearsheet_text
from quantforge.backtest import BacktestEngine
from quantforge.data.loader import DataLoader
from quantforge.portfolio.hrp import hierarchical_risk_parity
from quantforge.risk.stress_test import stress_scenarios
from quantforge.strategies import (
    BollingerMeanReversion, BuyAndHoldStrategy, DonchianBreakout,
    FactorStrategy, MACrossoverStrategy, MomentumStrategy,
)


def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    print("[1/7] Loading real market data — SPY/QQQ/IWM/TLT/GLD, 2015-2025")
    dl = DataLoader(cache_dir="data/cache")
    panel = dl.yfinance_many(["SPY", "QQQ", "IWM", "TLT", "GLD"],
                              start="2015-01-01", end="2025-01-01")
    print(f"     loaded {min(len(v) for v in panel.values())} bars x {len(panel)} assets")

    print("\n[2/7] Benchmark: Buy & Hold SPY (single asset)")
    bench_eng = BacktestEngine(strategy=BuyAndHoldStrategy(), data={"SPY": panel["SPY"]},
                                initial_capital=100_000, sizing_fraction=1.0)
    bench_res = bench_eng.run()
    bench_returns = bench_res.equity_curve.pct_change().dropna()
    print(f"     SPY 2015-2024: {summary_stats(bench_res.equity_curve)['total_return']*100:+.1f}% total, "
          f"Sharpe {summary_stats(bench_res.equity_curve)['sharpe']:+.2f}, "
          f"MDD {summary_stats(bench_res.equity_curve)['max_drawdown']*100:+.1f}%")

    print("\n[3/7] Running 6 strategies on 5-asset panel (monthly rebalance)")
    strategies = {
        "Mom(120)": MomentumStrategy(lookback=120, allow_short=False),
        "MA 20/100": MACrossoverStrategy(20, 100),
        "MA 50/200": MACrossoverStrategy(50, 200),
        "Donchian": DonchianBreakout(20, 10, allow_short=False),
        "Bollinger MR": BollingerMeanReversion(20, 2.0),
        "Factor": FactorStrategy(120, 60, top_q=0.4),
    }

    equity_curves = {"SPY B&H": bench_res.equity_curve}
    per_strategy_stats = []
    all_trades = {}
    for name, strat in strategies.items():
        eng = BacktestEngine(strategy=strat, data=panel, initial_capital=100_000,
                              sizing_fraction=0.2, rebalance="monthly")
        res = eng.run()
        equity_curves[name] = res.equity_curve
        all_trades[name] = res.trades
        stats = summary_stats(res.equity_curve, trades=res.trades)
        per_strategy_stats.append({"strategy": name, **stats})

    stats_df = pd.DataFrame(per_strategy_stats).set_index("strategy")
    print("\n     Per-strategy summary:")
    print(stats_df[["total_return", "annual_return", "sharpe", "max_drawdown",
                     "turnover", "cost_bps", "n_trades"]].round(3).to_string())

    print("\n[4/7] Benchmark-relative (alpha / beta / IR vs SPY)")
    bench_rows = []
    for name, curve in equity_curves.items():
        if name == "SPY B&H":
            continue
        r = curve.pct_change().dropna()
        rep = benchmark_report(r, bench_returns)
        bench_rows.append({
            "strategy": name, "alpha": rep.alpha, "beta": rep.beta,
            "info_ratio": rep.info_ratio, "tracking_error": rep.tracking_error,
            "up_capture": rep.up_capture, "down_capture": rep.down_capture,
        })
    bench_df = pd.DataFrame(bench_rows).set_index("strategy")
    print(bench_df.round(3).to_string())

    print("\n[5/7] HRP combining strategies on their returns")
    eq_df = pd.DataFrame({k: v for k, v in equity_curves.items() if k != "SPY B&H"}).ffill().dropna()
    strat_returns = eq_df.pct_change().dropna()
    cov = strat_returns.cov().values * 252
    weights = hierarchical_risk_parity(cov)
    w = pd.Series(weights, index=strat_returns.columns)
    print(w.round(3).to_string())

    combined_ret = strat_returns @ w.values
    combined_eq = 100_000 * (1 + combined_ret).cumprod()
    combined_eq = pd.concat([pd.Series([100_000.0], index=[eq_df.index[0]]), combined_eq])

    print("\n[6/7] Combined HRP portfolio tearsheet:")
    print(tearsheet_text(combined_eq, "HRP Combined"))

    rep = benchmark_report(combined_eq.pct_change().dropna(), bench_returns)
    print(f"\nvs SPY B&H:\n{rep}")

    print("\n     60/40 SPY/TLT passive benchmark:")
    spy_r = panel["SPY"]["close"].pct_change().dropna()
    tlt_r = panel["TLT"]["close"].pct_change().dropna()
    sixty_forty = 0.6 * spy_r + 0.4 * tlt_r
    sixty_forty_eq = 100_000 * (1 + sixty_forty).cumprod()
    sf_stats = summary_stats(sixty_forty_eq)
    print(f"     60/40 return: {sf_stats['total_return']*100:+.1f}%  Sharpe {sf_stats['sharpe']:+.2f}  MDD {sf_stats['max_drawdown']*100:+.1f}%")

    print("\n[7/7] Saving artifacts")
    eq_df["SPY B&H"] = bench_res.equity_curve
    eq_df["60/40"] = sixty_forty_eq
    eq_df["HRP Combined"] = combined_eq
    eq_df.to_csv(out_dir / "real_pipeline_equity.csv")
    stats_df.to_csv(out_dir / "real_pipeline_stats.csv")
    bench_df.to_csv(out_dir / "real_pipeline_benchmark.csv")
    (out_dir / "real_pipeline_tearsheet.md").write_text(tearsheet_markdown(combined_eq, "HRP Combined (real data)"))
    print(f"     -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
