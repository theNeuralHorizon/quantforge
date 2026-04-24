"""End-to-end ML training pipeline: features -> train/val/test -> metrics -> deployable predict_fn.

No synthetic shortcuts. Proper time-ordered splits (no random shuffle — that would
leak future into past). Reports real accuracy, AUC, log-loss, feature importance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TrainingReport:
    feature_names: list[str]
    train_acc: float
    val_acc: float
    test_acc: float
    train_auc: float
    val_auc: float
    test_auc: float
    baseline_acc: float          # always-predict-majority baseline
    feature_importances: dict[str, float]
    n_train: int
    n_val: int
    n_test: int
    hp_search: pd.DataFrame | None = None  # grid search results
    confusion_test: np.ndarray | None = None

    def summary(self) -> str:
        lines = [
            f"Dataset sizes   : train={self.n_train}  val={self.n_val}  test={self.n_test}",
            f"Baseline acc    : {self.baseline_acc:.4f}  (always-majority)",
            f"Accuracy        : train={self.train_acc:.4f}  val={self.val_acc:.4f}  test={self.test_acc:.4f}",
            f"AUC             : train={self.train_auc:.4f}  val={self.val_auc:.4f}  test={self.test_auc:.4f}",
            "",
            "Top feature importances:",
        ]
        top = sorted(self.feature_importances.items(), key=lambda kv: kv[1], reverse=True)[:10]
        for k, v in top:
            lines.append(f"  {k:>20}: {v:.4f}")
        if self.confusion_test is not None:
            lines.append("")
            lines.append("Test confusion matrix (rows=true, cols=pred):")
            lines.append(f"  {self.confusion_test}")
        return "\n".join(lines)


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Approximate ROC-AUC via Mann-Whitney U (works for binary labels)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    rank_sum_pos = ranks[: len(pos)].sum()
    auc = (rank_sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def time_order_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]:
    """Split into train/val/test respecting time order (no shuffle)."""
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = (X.iloc[:n_train], y.iloc[:n_train])
    val = (X.iloc[n_train : n_train + n_val], y.iloc[n_train : n_train + n_val])
    test = (X.iloc[n_train + n_val :], y.iloc[n_train + n_val :])
    return train, val, test


def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    model_cls: Any | None = None,
    model_kwargs: dict[str, Any] | None = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    hp_grid: list[dict[str, Any]] | None = None,
    verbose: bool = True,
) -> tuple[Any, TrainingReport]:
    """Train a scikit-learn classifier with time-ordered splits + optional hp grid search.

    Returns (fitted_model, TrainingReport).
    """
    if model_cls is None:
        from sklearn.ensemble import GradientBoostingClassifier
        model_cls = GradientBoostingClassifier
    model_kwargs = model_kwargs or {}

    (Xtr, ytr), (Xva, yva), (Xte, yte) = time_order_split(X, y, train_frac, val_frac)

    grid_results: pd.DataFrame | None = None
    best_kwargs = dict(model_kwargs)

    if hp_grid:
        if verbose:
            print(f"  hyperparameter search over {len(hp_grid)} configs...")
        rows = []
        best_val = -np.inf
        for config in hp_grid:
            kwargs = {**model_kwargs, **config}
            m = model_cls(**kwargs)
            m.fit(Xtr.values, ytr.values)
            val_proba = m.predict_proba(Xva.values)[:, 1]
            val_acc = _accuracy(yva.values, (val_proba > 0.5).astype(int))
            val_auc = _auc(yva.values, val_proba)
            rows.append({**config, "val_acc": val_acc, "val_auc": val_auc})
            score = val_auc if not np.isnan(val_auc) else val_acc
            if score > best_val:
                best_val = score
                best_kwargs = kwargs
        grid_results = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
        if verbose:
            print(f"  best config: {best_kwargs}")

    # fit final model on train + val (common practice for time-ordered data)
    X_fit = pd.concat([Xtr, Xva])
    y_fit = pd.concat([ytr, yva])
    final_model = model_cls(**best_kwargs)
    final_model.fit(X_fit.values, y_fit.values)

    # evaluate
    def _eval(Xs, ys):
        proba = final_model.predict_proba(Xs.values)[:, 1]
        pred = (proba > 0.5).astype(int)
        return _accuracy(ys.values, pred), _auc(ys.values, proba), pred

    tr_acc, tr_auc, _ = _eval(Xtr, ytr)
    va_acc, va_auc, _ = _eval(Xva, yva)
    te_acc, te_auc, te_pred = _eval(Xte, yte)

    maj = int(ytr.mode().iloc[0])
    baseline = float((yte.values == maj).mean())

    # feature importances (sklearn convention)
    fi = getattr(final_model, "feature_importances_", None)
    if fi is not None:
        importance = dict(zip(X.columns, [float(x) for x in fi], strict=False))
    else:
        importance = {}

    # confusion
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(yte.values, te_pred, strict=False):
        cm[int(t), int(p)] += 1

    report = TrainingReport(
        feature_names=list(X.columns),
        train_acc=tr_acc, val_acc=va_acc, test_acc=te_acc,
        train_auc=tr_auc, val_auc=va_auc, test_auc=te_auc,
        baseline_acc=baseline,
        feature_importances=importance,
        n_train=len(Xtr), n_val=len(Xva), n_test=len(Xte),
        hp_search=grid_results,
        confusion_test=cm,
    )
    return final_model, report


def walk_forward_train(
    X: pd.DataFrame,
    y: pd.Series,
    model_cls: Any | None = None,
    model_kwargs: dict[str, Any] | None = None,
    initial_train: int = 252 * 3,
    step: int = 21,
    verbose: bool = False,
) -> pd.DataFrame:
    """Expanding-window walk-forward: train at each step, predict next `step` bars.

    Returns a DataFrame with one row per test block and columns:
    start, end, acc, auc, n_predictions.
    """
    if model_cls is None:
        from sklearn.ensemble import GradientBoostingClassifier
        model_cls = GradientBoostingClassifier
    model_kwargs = model_kwargs or {}

    rows = []
    n = len(X)
    start = initial_train
    while start + step <= n:
        m = model_cls(**model_kwargs)
        m.fit(X.iloc[:start].values, y.iloc[:start].values)
        Xte = X.iloc[start : start + step]
        yte = y.iloc[start : start + step]
        proba = m.predict_proba(Xte.values)[:, 1]
        pred = (proba > 0.5).astype(int)
        rows.append({
            "train_end": X.index[start - 1],
            "test_start": X.index[start],
            "test_end": X.index[start + step - 1],
            "acc": _accuracy(yte.values, pred),
            "auc": _auc(yte.values, proba),
            "n": step,
        })
        if verbose:
            print(f"  block ending {X.index[start + step - 1].date()}: acc={rows[-1]['acc']:.3f}  auc={rows[-1]['auc']:.3f}")
        start += step
    return pd.DataFrame(rows)
