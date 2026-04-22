"""Command-line entry for QuantForge.

Usage:
    python -m quantforge backtest --strategy momentum --lookback 120
    python -m quantforge price --S 100 --K 100 --T 1.0 --sigma 0.2
    python -m quantforge tearsheet --strategy ma --fast 20 --slow 100
    python -m quantforge tournament
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List

from quantforge.analytics.performance import summary_stats
from quantforge.analytics.tearsheet import tearsheet_text
from quantforge.backtest import BacktestEngine
from quantforge.data.synthetic import generate_panel
from quantforge.options.black_scholes import bs_call, bs_put, bs_implied_vol
from quantforge.options.binomial import crr_price, crr_american
from quantforge.options.monte_carlo import mc_european
from quantforge.options.greeks import all_greeks
from quantforge.strategies import (
    BollingerMeanReversion, DonchianBreakout, FactorStrategy,
    MACrossoverStrategy, MomentumStrategy, RSIReversalStrategy,
)


_STRATEGY_MAP = {
    "momentum": MomentumStrategy,
    "ma": MACrossoverStrategy,
    "bollinger": BollingerMeanReversion,
    "donchian": DonchianBreakout,
    "rsi": RSIReversalStrategy,
    "factor": FactorStrategy,
}


def _make_strategy(name: str, kwargs: dict):
    if name not in _STRATEGY_MAP:
        raise SystemExit(f"Unknown strategy: {name}. Choices: {list(_STRATEGY_MAP)}")
    cls = _STRATEGY_MAP[name]
    filtered = {k: v for k, v in kwargs.items() if v is not None}
    return cls(**filtered)


def _kv_args(lst: List[str]) -> dict:
    out = {}
    for item in lst:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def cmd_backtest(args) -> int:
    params = _kv_args(args.params or [])
    strat = _make_strategy(args.strategy, params)
    panel = generate_panel([f"S{i}" for i in range(args.n_assets)], n=args.bars, seed=args.seed)
    engine = BacktestEngine(strategy=strat, data=panel, initial_capital=args.capital, sizing_fraction=args.sizing)
    res = engine.run()
    s = summary_stats(res.equity_curve)
    out = {
        "strategy": args.strategy,
        "params": params,
        "n_assets": args.n_assets,
        "bars": args.bars,
        "trades": len(res.trades),
        "final_equity": float(res.equity_curve.iloc[-1]),
        "stats": s,
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_tearsheet(args) -> int:
    params = _kv_args(args.params or [])
    strat = _make_strategy(args.strategy, params)
    panel = generate_panel([f"S{i}" for i in range(args.n_assets)], n=args.bars, seed=args.seed)
    engine = BacktestEngine(strategy=strat, data=panel, initial_capital=args.capital, sizing_fraction=args.sizing)
    res = engine.run()
    print(tearsheet_text(res.equity_curve, args.strategy))
    return 0


def cmd_price(args) -> int:
    bs = bs_call(args.S, args.K, args.T, args.r, args.sigma) if args.type == "call" else bs_put(args.S, args.K, args.T, args.r, args.sigma)
    crr = crr_price(args.S, args.K, args.T, args.r, args.sigma, args.type, args.steps)
    crr_am = crr_american(args.S, args.K, args.T, args.r, args.sigma, args.type, args.steps)
    mc_p, mc_se = mc_european(args.S, args.K, args.T, args.r, args.sigma, args.type, n_paths=args.n_paths, seed=args.seed)
    greeks = all_greeks(args.S, args.K, args.T, args.r, args.sigma, args.type)
    print(f"Black-Scholes : {bs:.6f}")
    print(f"CRR ({args.steps:,} steps): {crr:.6f}")
    print(f"American      : {crr_am:.6f}  (premium {crr_am - bs:+.6f})")
    print(f"Monte Carlo   : {mc_p:.6f}  (±{mc_se:.6f}, n={args.n_paths:,})")
    print("\nGreeks:")
    for k, v in greeks.items():
        print(f"  {k:>6}: {v:+.6f}")
    return 0


def cmd_iv(args) -> int:
    iv = bs_implied_vol(args.price, args.S, args.K, args.T, args.r, args.type)
    print(f"Implied volatility: {iv:.6f}")
    return 0


def cmd_tournament(args) -> int:
    import pandas as pd
    panel = generate_panel(["AAA", "BBB", "CCC"], n=args.bars, seed=args.seed)
    strategies = {
        "Mom(60)": MomentumStrategy(lookback=60),
        "Mom(120)": MomentumStrategy(lookback=120, allow_short=True),
        "MA 10/30": MACrossoverStrategy(10, 30),
        "MA 20/100": MACrossoverStrategy(20, 100),
        "BB MR": BollingerMeanReversion(20, 2.0),
        "Donchian": DonchianBreakout(20, 10),
        "RSI": RSIReversalStrategy(),
        "Factor": FactorStrategy(120, 60),
    }
    rows = []
    for n, s in strategies.items():
        eng = BacktestEngine(strategy=s, data=panel, initial_capital=100_000, sizing_fraction=0.25)
        r = eng.run()
        stats = summary_stats(r.equity_curve)
        rows.append({"strategy": n, **stats, "trades": len(r.trades)})
    df = pd.DataFrame(rows).set_index("strategy")
    print(df[["total_return", "annual_return", "annual_vol", "sharpe", "sortino",
             "max_drawdown", "trades"]].round(4).to_string())
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantforge", description="QuantForge CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("backtest", help="Run a strategy backtest")
    pb.add_argument("--strategy", required=True, choices=list(_STRATEGY_MAP.keys()))
    pb.add_argument("--params", nargs="*", help="key=value pairs passed to the strategy")
    pb.add_argument("--bars", type=int, default=500)
    pb.add_argument("--seed", type=int, default=42)
    pb.add_argument("--capital", type=float, default=100_000)
    pb.add_argument("--sizing", type=float, default=0.25)
    pb.add_argument("--n-assets", type=int, default=3)
    pb.set_defaults(func=cmd_backtest)

    pt = sub.add_parser("tearsheet", help="Run backtest and print tearsheet")
    pt.add_argument("--strategy", required=True, choices=list(_STRATEGY_MAP.keys()))
    pt.add_argument("--params", nargs="*")
    pt.add_argument("--bars", type=int, default=500)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--capital", type=float, default=100_000)
    pt.add_argument("--sizing", type=float, default=0.25)
    pt.add_argument("--n-assets", type=int, default=3)
    pt.set_defaults(func=cmd_tearsheet)

    pp = sub.add_parser("price", help="Price an option with all models")
    pp.add_argument("--S", type=float, required=True)
    pp.add_argument("--K", type=float, required=True)
    pp.add_argument("--T", type=float, required=True)
    pp.add_argument("--r", type=float, default=0.04)
    pp.add_argument("--sigma", type=float, required=True)
    pp.add_argument("--type", choices=["call", "put"], default="call")
    pp.add_argument("--steps", type=int, default=500)
    pp.add_argument("--n-paths", type=int, default=50_000)
    pp.add_argument("--seed", type=int, default=42)
    pp.set_defaults(func=cmd_price)

    pi = sub.add_parser("iv", help="Solve implied volatility")
    pi.add_argument("--price", type=float, required=True)
    pi.add_argument("--S", type=float, required=True)
    pi.add_argument("--K", type=float, required=True)
    pi.add_argument("--T", type=float, required=True)
    pi.add_argument("--r", type=float, default=0.04)
    pi.add_argument("--type", choices=["call", "put"], default="call")
    pi.set_defaults(func=cmd_iv)

    pto = sub.add_parser("tournament", help="Compare 8 strategies head-to-head")
    pto.add_argument("--bars", type=int, default=600)
    pto.add_argument("--seed", type=int, default=11)
    pto.set_defaults(func=cmd_tournament)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
