"""Example 14: ACTUAL ML training on real SPY data.

- Downloads real SPY 2010-2024 via yfinance
- Builds 20+ features (lagged returns, vol, RSI, MACD, BB-position, volume ratios)
- Time-ordered train/val/test split (no shuffle to avoid lookahead)
- Hyperparameter search over GradientBoosting depth + n_estimators
- Reports train/val/test accuracy + AUC + baseline + feature importance + confusion matrix
- Walk-forward out-of-sample evaluation
- Backtests the trained model on the test period and compares to Buy & Hold
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.analytics.benchmark import benchmark_report
from quantforge.analytics.performance import summary_stats
from quantforge.analytics.tearsheet import tearsheet_text
from quantforge.backtest import BacktestEngine
from quantforge.core.event import EventType, SignalEvent
from quantforge.data.loader import DataLoader
from quantforge.ml.features import build_feature_matrix, target_labels
from quantforge.ml.trainer import train_classifier, walk_forward_train
from quantforge.strategies import BuyAndHoldStrategy
from quantforge.strategies.base import Strategy


class PreTrainedMLStrategy(Strategy):
    """Uses a pre-fitted model + a feature function; does NO training during backtest."""
    name = "pretrained_ml"

    def __init__(self, model, feature_fn, prob_threshold: float = 0.55, allow_short: bool = False):
        self.model = model
        self.feature_fn = feature_fn
        self.prob_threshold = prob_threshold
        self.allow_short = allow_short

    def warmup(self) -> int:
        return 250

    def on_bar(self, symbol, bar, history):
        if len(history) < self.warmup():
            return []
        feats = self.feature_fn(history)
        if feats.empty:
            return []
        x = feats.iloc[[-1]].values
        try:
            proba = self.model.predict_proba(x)[0]
        except Exception:
            return []
        prob_up = proba[1] if len(proba) == 2 else 0.5
        if prob_up > self.prob_threshold:
            return [SignalEvent(EventType.SIGNAL, bar.name, symbol, 1, float(prob_up), self.name)]
        if prob_up < 1 - self.prob_threshold and self.allow_short:
            return [SignalEvent(EventType.SIGNAL, bar.name, symbol, -1, float(1 - prob_up), self.name)]
        if 0.45 < prob_up < 0.55:
            return [SignalEvent(EventType.SIGNAL, bar.name, symbol, 0, 1.0, self.name)]
        return []


def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    print("[1/6] Loading real SPY 2010-2024")
    dl = DataLoader(cache_dir="data/cache")
    df = dl.yfinance("SPY", start="2010-01-01", end="2025-01-01")
    print(f"     {len(df)} bars, {df.index[0].date()} -> {df.index[-1].date()}")

    print("\n[2/6] Building features + binary direction target (next-day up/down)")
    feats = build_feature_matrix(df)
    target = target_labels(df["close"], horizon=1, kind="binary")
    data = feats.join(target.rename("y")).dropna()
    X = data.drop(columns="y")
    y = data["y"]
    print(f"     Features: {len(X.columns)}   Samples: {len(X)}")
    print(f"     Feature names: {list(X.columns)[:10]}...")
    print(f"     Target distribution: up={(y == 1).mean():.2%}  down={(y == 0).mean():.2%}")

    print("\n[3/6] Training with hyperparameter search (val AUC)")
    from sklearn.ensemble import GradientBoostingClassifier
    hp_grid = []
    for n_est in [50, 100, 200]:
        for max_d in [2, 3, 5]:
            for lr in [0.05, 0.1]:
                hp_grid.append({"n_estimators": n_est, "max_depth": max_d, "learning_rate": lr, "random_state": 42})

    model, report = train_classifier(
        X, y,
        model_cls=GradientBoostingClassifier,
        hp_grid=hp_grid,
        train_frac=0.6,
        val_frac=0.2,
        verbose=True,
    )
    print()
    print(report.summary())

    print("\n[4/6] Walk-forward out-of-sample evaluation (expanding window, 1m blocks)")
    best_kwargs = report.hp_search.iloc[0].drop(["val_acc", "val_auc"]).to_dict()
    for k in ("n_estimators", "max_depth", "random_state"):
        if k in best_kwargs:
            best_kwargs[k] = int(best_kwargs[k])
    wf = walk_forward_train(
        X, y,
        model_cls=GradientBoostingClassifier,
        model_kwargs=best_kwargs,
        initial_train=252 * 3,  # 3 years
        step=21,                # 1 month
    )
    print(f"     Walk-forward blocks: {len(wf)}")
    print(f"     Mean OOS accuracy: {wf['acc'].mean():.4f}  (baseline ~{(y == 1).mean():.4f})")
    print(f"     Mean OOS AUC     : {wf['auc'].mean():.4f}")
    print(f"     Accuracy > baseline in {(wf['acc'] > (y == 1).mean()).mean() * 100:.1f}% of blocks")

    print("\n[5/6] Backtest the trained model on the OOS test period")
    test_start_idx = int(len(X) * 0.8)
    test_start_ts = X.index[test_start_idx]
    df_test = df.loc[df.index >= test_start_ts]

    def feature_fn(history):
        return build_feature_matrix(history)

    strat = PreTrainedMLStrategy(model=model, feature_fn=feature_fn,
                                  prob_threshold=0.53, allow_short=False)
    eng = BacktestEngine(strategy=strat, data={"SPY": df_test},
                          initial_capital=100_000, sizing_fraction=1.0,
                          rebalance="bar")
    res = eng.run()

    bh_eng = BacktestEngine(strategy=BuyAndHoldStrategy(), data={"SPY": df_test},
                             initial_capital=100_000, sizing_fraction=1.0)
    bh_res = bh_eng.run()

    print(f"\n     Test period: {df_test.index[0].date()} -> {df_test.index[-1].date()}  ({len(df_test)} bars)")
    print(f"     ML strategy:  {tearsheet_text(res.equity_curve, 'ML OOS')}")
    print(f"\n     Buy & Hold:   {tearsheet_text(bh_res.equity_curve, 'SPY B&H')}")

    rep = benchmark_report(res.equity_curve.pct_change().dropna(), bh_res.equity_curve.pct_change().dropna())
    print(f"\n     ML vs B&H:\n{rep}")

    print("\n[6/6] Saving artifacts")
    fi_df = pd.DataFrame(sorted(report.feature_importances.items(), key=lambda kv: kv[1], reverse=True),
                          columns=["feature", "importance"])
    fi_df.to_csv(out_dir / "ml_feature_importance.csv", index=False)
    wf.to_csv(out_dir / "ml_walk_forward.csv", index=False)
    if report.hp_search is not None:
        report.hp_search.to_csv(out_dir / "ml_hp_search.csv", index=False)
    pd.DataFrame({"ml": res.equity_curve, "buy_hold": bh_res.equity_curve}).ffill().to_csv(
        out_dir / "ml_test_equity.csv")
    print(f"     -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
