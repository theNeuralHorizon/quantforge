"""Example 06: Train a GradientBoosting classifier and backtest it."""
from __future__ import annotations

from quantforge.backtest import BacktestEngine
from quantforge.data.synthetic import generate_ohlcv
from quantforge.strategies.ml_strategy import MLClassifierStrategy
from quantforge.analytics.tearsheet import tearsheet_text


def main() -> None:
    df = generate_ohlcv(800, s0=100, mu=0.08, sigma=0.18, seed=3)

    strat = MLClassifierStrategy(
        train_window=252,
        retrain_every=21,
        prob_threshold=0.55,
        allow_short=True,
    )
    panel = {"ASSET": df}
    engine = BacktestEngine(strategy=strat, data=panel, initial_capital=100_000, sizing_fraction=0.5)
    res = engine.run()

    print(tearsheet_text(res.equity_curve, "ML GradientBoosting"))
    print(f"\nTrades: {len(res.trades)}")


if __name__ == "__main__":
    main()
