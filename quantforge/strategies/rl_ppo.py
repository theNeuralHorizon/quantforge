"""Pure-numpy PPO-lite trading policy.

A minimal actor-critic that learns position sizing (long/flat/short) from
a small feature window. No torch/TF — one hidden layer, softmax policy,
linear value head, single-threaded gradient descent. Good enough to
demonstrate RL for quant; not a production model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantforge.core.event import EventType, SignalEvent
from quantforge.strategies.base import Strategy


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class PPONet:
    n_features: int
    n_actions: int = 3                  # short(-1) / flat(0) / long(+1)
    hidden: int = 16
    W1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)
    Wp: np.ndarray = field(init=False)
    bp: np.ndarray = field(init=False)
    Wv: np.ndarray = field(init=False)
    bv: np.ndarray = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(42)
        sd = 1.0 / np.sqrt(self.n_features)
        self.W1 = rng.normal(0, sd, (self.n_features, self.hidden)).astype(np.float64)
        self.b1 = np.zeros(self.hidden)
        self.Wp = rng.normal(0, 1.0 / np.sqrt(self.hidden), (self.hidden, self.n_actions))
        self.bp = np.zeros(self.n_actions)
        self.Wv = rng.normal(0, 1.0 / np.sqrt(self.hidden), (self.hidden, 1))
        self.bv = np.zeros(1)

    def _hidden(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.W1 + self.b1)

    def policy_probs(self, x: np.ndarray) -> np.ndarray:
        h = self._hidden(x)
        return _softmax(h @ self.Wp + self.bp)

    def value(self, x: np.ndarray) -> np.ndarray:
        h = self._hidden(x)
        return (h @ self.Wv + self.bv).squeeze(-1)


def _make_features(close: pd.Series, window: int) -> np.ndarray:
    r = close.pct_change().fillna(0).values
    # simple lag-window: r_{t-0..window-1}, vol, ma-ratio
    feats = []
    for i in range(window - 1, len(r) - 1):
        lags = r[i - window + 1:i + 1]
        vol = lags.std()
        feats.append(np.concatenate([lags, [vol, lags.mean(), lags.sum()]]))
    return np.asarray(feats, dtype=np.float64)


@dataclass
class PPOStrategy(Strategy):
    """Train on history, deploy on the latest bar.

    Retrains every `retrain_every` bars, picking actions via the policy
    when deployed. Actions {0,1,2} map to direction {-1, 0, +1}.
    """
    window: int = 20
    train_window: int = 504
    retrain_every: int = 63
    n_epochs: int = 4
    clip: float = 0.2
    lr: float = 0.005
    allow_short: bool = True
    name: str = "ppo_lite"
    _net: PPONet | None = None
    _bars_since_retrain: int = 0

    def warmup(self) -> int:
        return self.train_window + self.window + 5

    def _build_net(self, n_features: int) -> PPONet:
        return PPONet(n_features=n_features, n_actions=3, hidden=16)

    def _train(self, close: pd.Series) -> None:
        X = _make_features(close, self.window)
        if len(X) < 50:
            return
        r = close.pct_change().fillna(0).values
        # Next-step reward: realized return * direction. Shift so X[i]'s reward
        # is r[i + window].
        start_idx = self.window - 1
        rewards_available = r[start_idx + 1:start_idx + 1 + len(X)]
        if len(rewards_available) < len(X):
            X = X[:len(rewards_available)]

        # On first call or feature-dim change, build a fresh net
        if self._net is None or self._net.n_features != X.shape[1]:
            self._net = self._build_net(X.shape[1])

        # Rollout: pick actions under current policy, compute rewards
        probs = self._net.policy_probs(X)
        rng = np.random.default_rng(1)
        actions = np.array([rng.choice(3, p=p) for p in probs])
        directions = actions - 1  # {-1, 0, +1}
        if not self.allow_short:
            directions = np.maximum(directions, 0)
        rewards = directions * rewards_available

        # Advantage: (r - v(x)) with value fit by simple bootstrap
        values = self._net.value(X)
        advantages = rewards - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Old policy log-probs (fixed during update)
        old_logp = np.log(probs[np.arange(len(actions)), actions] + 1e-8)

        # Gradient-descent PPO update (single batch; enough for a demo)
        for _ in range(self.n_epochs):
            new_probs = self._net.policy_probs(X)
            new_logp = np.log(new_probs[np.arange(len(actions)), actions] + 1e-8)
            ratio = np.exp(new_logp - old_logp)
            clipped = np.clip(ratio, 1 - self.clip, 1 + self.clip)
            # finite-difference gradient on the softmax output
            # We only update policy (no full backprop for brevity); step parameters
            # in the direction of the probability of good actions.
            good = advantages > 0
            adj = np.where(good, 1.0, -1.0) * np.minimum(ratio * advantages, clipped * advantages)
            # Update the bias of the chosen action (simplistic; shows the idea)
            for k in range(3):
                mask = actions == k
                if mask.any():
                    self._net.bp[k] += self.lr * adj[mask].mean()
            # Update value head towards true rewards via least-squares step
            h = np.tanh(X @ self._net.W1 + self._net.b1)
            pred_v = (h @ self._net.Wv + self._net.bv).squeeze(-1)
            err = rewards - pred_v
            self._net.Wv += self.lr * h.T @ err.reshape(-1, 1) / len(X)
            self._net.bv += self.lr * err.mean()

    def on_bar(self, symbol: str, bar: pd.Series, history: pd.DataFrame) -> list[SignalEvent]:
        if len(history) < self.warmup():
            return []
        if self._net is None or self._bars_since_retrain >= self.retrain_every:
            self._train(history["close"].iloc[-self.train_window:])
            self._bars_since_retrain = 0
            if self._net is None:
                return []
        self._bars_since_retrain += 1

        X = _make_features(history["close"].iloc[-self.window - 1:], self.window)
        if len(X) == 0:
            return []
        probs = self._net.policy_probs(X[[-1]])
        action = int(np.argmax(probs[0]))
        direction = action - 1
        if not self.allow_short and direction < 0:
            direction = 0
        return [SignalEvent(
            event_type=EventType.SIGNAL, timestamp=bar.name,
            symbol=symbol, direction=direction,
            strength=float(probs[0, action]), strategy_id=self.name,
        )]
