# QuantForge — Architecture

## High-level view

```
                +-------------------+
                |    dashboard/     |  <-- Streamlit UI (7 pages)
                +---------+---------+
                          |
                          v
+-------------+   +-------+-------+   +---------------+
|  examples/  |-->|   quantforge/ |<--|     cli.py    |
+-------------+   +-------+-------+   +---------------+
                          |
        +-----+-----+-----+-----+-----+-----+-----+
        v     v     v     v     v     v     v     v
      core  data indic opts portf risk  ml  analytics
            |     |     |     |     |    |     |
            |     +---+-+     |     +----+--+  |
            v         v       v            v   v
        backtest  strategies  portfolio-opt  tearsheet
```

## Module responsibilities

### `quantforge.core`
Primitives shared across the platform. No dependencies on the rest of `quantforge`.
- `event.py` — dataclasses: `MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`
- `order.py` — `Order` with fill tracking
- `position.py` — per-symbol quantity, average cost, realized/unrealized P&L
- `portfolio.py` — aggregates positions, tracks cash and equity

### `quantforge.data`
Market-data producers.
- `synthetic.py` — GBM, OHLCV, correlated multi-asset returns, panel of assets
- `loader.py` — CSV and optional yfinance with caching

### `quantforge.indicators`
Pure NumPy/Pandas indicators. Stateless.
- `technical.py` — 15 classic indicators (SMA/EMA/WMA, RSI, MACD, BBands, ATR, ADX, Stoch, OBV, VWAP, Donchian, Keltner, ROC, CCI, Williams %R)
- `statistical.py` — rolling z-score, correlations, betas, realized/EWMA/Garman-Klass vol, Hurst, ADF (native fallback), half-life

### `quantforge.options`
Pricing + Greeks. Pure functions.
- `black_scholes.py` — `bs_call`, `bs_put`, `bs_implied_vol` (robust bisection+Newton)
- `greeks.py` — Δ, Γ, Vega, Θ, ρ
- `binomial.py` — Cox-Ross-Rubinstein European + American
- `monte_carlo.py` — European, Asian (arithmetic/geometric), barrier (4 kinds), lookback (fixed/floating)

### `quantforge.portfolio`
Portfolio optimizers. Scipy-based (no CVXPY).
- `markowitz.py` — `min_variance`, `max_sharpe`, `mean_variance`, `efficient_frontier`
- `risk_parity.py` — `equal_risk_contribution`, `risk_parity` with custom risk budgets
- `black_litterman.py` — Posterior mean + covariance under investor views
- `hrp.py` — Hierarchical Risk Parity (Lopez de Prado)

### `quantforge.risk`
- `var.py` — Historical / Parametric / Cornish-Fisher / Monte-Carlo VaR and CVaR
- `drawdown.py` — drawdown series, max drawdown, drawdown table
- `stress_test.py` — Historical-shock library, per-asset-class and factor shocks
- `metrics.py` — Sharpe, Sortino, Calmar, Omega, Tail Ratio, Ulcer, Gain-to-pain, IR

### `quantforge.ml`
- `features.py` — price/volume/volatility feature blocks, `build_feature_matrix`, `target_labels`
- `regime.py` — bull/bear, vol, trend, GMM-lite HMM regime detectors
- `forecast.py` — AR(p), EWMA, linear-regression forecasts, sequence helper

### `quantforge.analytics`
- `performance.py` — `summary_stats` (all-in-one metrics dict)
- `tearsheet.py` — text + markdown tearsheets
- `attribution.py` — Brinson-Hood-Beebower + time-series factor attribution

### `quantforge.backtest`
Event-driven engine built on top of all the above.
- `slippage.py` — `NoSlippage`, `FixedBpsSlippage`, `VolumeImpactSlippage` (sqrt impact)
- `commission.py` — `NoCommission`, `FixedBpsCommission`, `PerShareCommission`
- `broker.py` — `SimulatedBroker` fills orders against bars
- `engine.py` — `BacktestEngine`: main event loop, sizing, equity tracking

### `quantforge.strategies`
- `momentum`, `mean_reversion`, `ma_crossover`, `rsi_reversal`, `trend_breakout` (Donchian), `pairs_trading`, `factor_strategy`, `ml_strategy` (sklearn)

### `quantforge.dashboard`
Streamlit app with 7 pages:
1. Overview
2. Data & Indicators
3. Strategy Backtest (parameterized)
4. Strategy Tournament (head-to-head of 8 strategies)
5. Options Pricing (BS + CRR + MC + Greeks + IV smile)
6. Portfolio Optimization (efficient frontier + 4 optimizers)
7. Risk Analysis (VaR methods, drawdown, distribution)

## Event flow in the backtest

```
+--------+     MarketEvent      +----------+      SignalEvent     +--------+
|  Data  | -------------------> | Strategy | -------------------> | Engine |
+--------+                      +----------+                      +---+----+
                                                                      |
                                                                      v
                                                                OrderEvent
                                                                      |
                                                                      v
                                                                  +---+---+
                                                                  | Broker|
                                                                  +---+---+
                                                                      |
                                                                      v
                                                                  FillEvent
                                                                      |
                                                                      v
                                                              +-------+-------+
                                                              | Portfolio +    |
                                                              | equity curve   |
                                                              +----------------+
```

## Design decisions

1. **Pure NumPy + scipy** for math. No torch/tensorflow dependency; sklearn is optional.
2. **Dataclasses everywhere** for events / positions / orders. Frozen where reasonable.
3. **Target-position sizing** in the engine: signals specify target direction, the engine computes the delta. Prevents over-accumulation on repeated signals.
4. **Vectorized indicators** — indicators take a full Series and return a Series, never a single value.
5. **Deterministic**: seeded RNG everywhere, no network calls in the core path.
6. **Testable**: core math is pure, state is localized to Portfolio / Broker / BacktestEngine.

## Testing

266 tests across 9 files cover all critical math paths:
- Put-call parity, IV round-trip, CRR-to-BS convergence, MC vs analytical
- Optimizer constraints (weights sum, bounds, variance ordering)
- VaR/CVaR sanity on known distributions
- Position fill math (long/short/flip)
- Portfolio cash accounting
- All indicators' boundary conditions

Run `pytest tests/ -q` from the repo root.
