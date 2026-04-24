"""ML-based strategy using scikit-learn: sklearn-style classifier predicts next-bar direction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantforge.core.event import SignalEvent
from quantforge.strategies.base import Strategy


def _default_features(close: pd.Series) -> pd.DataFrame:
    r = close.pct_change()
    return pd.DataFrame({
        "r1": r,
        "r5": close.pct_change(5),
        "r10": close.pct_change(10),
        "vol20": r.rolling(20).std(),
        "sma_ratio": close / close.rolling(20).mean() - 1,
        "rsi_proxy": r.rolling(14).apply(lambda x: (x > 0).sum() / len(x), raw=False),
    }).dropna()


@dataclass
class MLClassifierStrategy(Strategy):
    """Trains a classifier on rolling window; predicts next-bar direction."""
    train_window: int = 252
    retrain_every: int = 21
    model_cls: Any | None = None  # e.g. RandomForestClassifier
    model_kwargs: dict = field(default_factory=dict)
    prob_threshold: float = 0.55
    allow_short: bool = True
    name: str = "ml_classifier"
    _model: Any = None
    _bars_since_retrain: int = 0

    def warmup(self) -> int:
        return self.train_window + 30

    def _get_model(self):
        if self.model_cls is None:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=42, **self.model_kwargs
            )
        return self.model_cls(**self.model_kwargs)

    def _prepare(self, history: pd.DataFrame):
        feats = _default_features(history["close"])
        target = (history["close"].pct_change().shift(-1) > 0).astype(int)
        data = feats.join(target.rename("y")).dropna()
        if len(data) < 30:
            return None
        return data

    def _fit(self, history: pd.DataFrame) -> bool:
        data = self._prepare(history)
        if data is None:
            return False
        train = data.iloc[-self.train_window:]
        X = train.drop(columns=["y"]).values
        y = train["y"].values
        if len(np.unique(y)) < 2:
            return False
        self._model = self._get_model()
        self._model.fit(X, y)
        return True

    def on_bar(self, symbol: str, bar: pd.Series, history: pd.DataFrame) -> list[SignalEvent]:
        if len(history) < self.warmup():
            return []
        if self._model is None or self._bars_since_retrain >= self.retrain_every:
            ok = self._fit(history.iloc[:-1])
            self._bars_since_retrain = 0
            if not ok:
                return []
        self._bars_since_retrain += 1

        feats = _default_features(history["close"])
        if feats.empty:
            return []
        x = feats.iloc[[-1]].values
        try:
            proba = self._model.predict_proba(x)[0]
        except Exception:
            return []
        prob_up = proba[1] if len(proba) == 2 else 0.5
        if prob_up > self.prob_threshold:
            return [self._signal(bar.name, symbol, 1, float(prob_up), self.name)]
        if prob_up < 1 - self.prob_threshold and self.allow_short:
            return [self._signal(bar.name, symbol, -1, float(1 - prob_up), self.name)]
        if 0.45 < prob_up < 0.55:
            return [self._signal(bar.name, symbol, 0, 1.0, self.name)]
        return []
