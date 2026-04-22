# QuantForge — Strategy Reference

All strategies implement the `Strategy` base class (`quantforge/strategies/base.py`):

```python
class Strategy(ABC):
    def warmup(self) -> int: ...
    def on_bar(self, symbol: str, bar: pd.Series, history: pd.DataFrame) -> List[SignalEvent]: ...
```

A `SignalEvent` has `symbol`, `direction ∈ {-1, 0, +1}`, and `strength ∈ [0, 1]`. Direction 0 means "close position" — the engine translates it into the right delta order.

## Built-in strategies

### `MomentumStrategy`
Time-series momentum. Long when lookback return > threshold.

```python
MomentumStrategy(lookback=60, threshold=0.0, allow_short=False)
```

### `MACrossoverStrategy`
Classic EMA cross. Buy on fast-crosses-up-slow, sell/flat on fast-crosses-down-slow.

```python
MACrossoverStrategy(fast=10, slow=30, allow_short=False)
```

### `MeanReversionStrategy`
Z-score mean reversion. Long when z < -entry, short when z > entry, flat when |z| < exit.

```python
MeanReversionStrategy(lookback=20, entry_z=2.0, exit_z=0.5, allow_short=True)
```

### `BollingerMeanReversion`
Same idea but using Bollinger Bands.

```python
BollingerMeanReversion(window=20, k=2.0)
```

### `RSIReversalStrategy`
Long on oversold, short on overbought.

```python
RSIReversalStrategy(window=14, oversold=30, overbought=70, allow_short=True)
```

### `DonchianBreakout`
Turtle-style breakout. Long when price > N-day high; exit when price < M-day low.

```python
DonchianBreakout(entry_window=20, exit_window=10, allow_short=True)
```

### `PairsTradingStrategy`
Trades the spread of two cointegrated assets using a rolling z-score.

```python
PairsTradingStrategy(asset_a="A", asset_b="B", window=60, entry_z=2.0, exit_z=0.5)
```

### `FactorStrategy`
Composite factor score (momentum + low-vol) applied cross-sectionally. Goes long top quintile.

```python
FactorStrategy(lookback_mom=120, lookback_vol=60, top_q=0.2)
```

### `MLClassifierStrategy`
Rolling-retrained sklearn classifier on price/vol features. Defaults to `GradientBoostingClassifier`.

```python
MLClassifierStrategy(train_window=252, retrain_every=21, prob_threshold=0.55)
```

## Writing your own strategy

```python
from quantforge.strategies.base import Strategy
from quantforge.indicators.technical import rsi

class MyStrategy(Strategy):
    name = "my_strategy"
    def __init__(self, window=14):
        self.window = window

    def warmup(self):
        return self.window * 2

    def on_bar(self, symbol, bar, history):
        if len(history) < self.warmup():
            return []
        r = rsi(history["close"], self.window).iloc[-1]
        if r < 25:
            return [self._signal(bar.name, symbol, +1, 1.0, self.name)]
        if r > 75:
            return [self._signal(bar.name, symbol, -1, 1.0, self.name)]
        return []
```

## Strategy tournament results (synthetic seed=11)

```
                total_return  annual_return  annual_vol  sharpe  max_drawdown  trades
Momentum(60)           0.132         0.054       0.082   0.676        -0.092    1541
Momentum(120)          0.250         0.098       0.103   0.963        -0.077    1912
MA 10/30               0.139         0.056       0.093   0.635        -0.097      85
MA 20/100              0.309         0.120       0.103   1.156        -0.076      20
RSI Reversal          -0.021        -0.009       0.049  -0.158        -0.082     193
Bollinger MR           0.025         0.010       0.034   0.322        -0.052     363
Donchian              -0.202        -0.091       0.097  -0.932        -0.275     515
Factor Composite       0.059         0.024       0.055   0.461        -0.070     509
```

*Note:* These are run against synthetic GBM data. Real performance depends entirely on the market regime.
