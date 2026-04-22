# QuantForge

**A Full-Stack Quantitative Trading Research Platform**

Built in a single session. **304 tests, 0 failing. 60+ Python files. 11 modules. 11 end-to-end examples. Streamlit dashboard. CLI.**

QuantForge is an end-to-end quant research library covering:
- Synthetic + historical data ingestion
- 25+ technical & statistical indicators
- Event-driven backtesting with realistic broker simulation
- 8 built-in trading strategies (momentum, mean reversion, pairs, factor, ML, ...)
- Options pricing (Black-Scholes, CRR binomial, Monte Carlo exotics, Greeks)
- 5 portfolio optimizers (Markowitz, ERC, Black-Litterman, HRP, CVaR)
- Fixed-income analytics (duration, convexity, Nelson-Siegel yield curves)
- Risk analytics (VaR, CVaR, stress tests, drawdown, Kelly sizing, Monte Carlo)
- ML features, regime detection, factor models
- Performance analytics, tearsheets, attribution
- Interactive Streamlit dashboard
- CLI for scripting

## Modules

| Module | Purpose |
|--------|---------|
| `quantforge.core` | Portfolio, positions, orders, event primitives |
| `quantforge.data` | Market data (synthetic + yfinance + CSV) |
| `quantforge.indicators` | 15 technical + 10 statistical indicators |
| `quantforge.options` | Black-Scholes, binomial, Monte Carlo, Greeks, IV solver |
| `quantforge.portfolio` | Markowitz, ERC, Black-Litterman, HRP, CVaR |
| `quantforge.fixed_income` | Bonds, duration, convexity, Nelson-Siegel/Svensson |
| `quantforge.risk` | VaR, CVaR, drawdown, metrics, Kelly, MC simulation |
| `quantforge.ml` | Features, regime detection, forecasting, factor models |
| `quantforge.analytics` | Performance, tearsheet, attribution |
| `quantforge.backtest` | Event-driven engine + broker + slippage + TCA |
| `quantforge.strategies` | 8 strategies: momentum, MA cross, MR, Donchian, RSI, pairs, factor, ML |
| `quantforge.dashboard` | 7-page Streamlit app |
| `quantforge.cli` | `python -m quantforge {backtest,price,iv,tournament,tearsheet}` |

## Quick Start

```bash
pip install -r requirements.txt

# Run any example
PYTHONPATH=. python examples/02_options_surface.py
PYTHONPATH=. python examples/04_strategy_comparison.py
PYTHONPATH=. python examples/10_full_research_pipeline.py

# CLI
python -m quantforge price --S 100 --K 100 --T 1 --sigma 0.2
python -m quantforge tournament --bars 600

# Dashboard
streamlit run quantforge/dashboard/app.py

# Tests
pytest tests/ -q
```

## Examples

`examples/` contains 11 end-to-end runnable scripts:

1. `01_data_and_indicators.py` — generate synthetic OHLCV, compute 20+ indicators
2. `02_options_surface.py` — BS, IV smile, CRR convergence, Greeks, MC exotics
3. `03_portfolio_optimization.py` — efficient frontier + 5 optimizers
4. `04_strategy_comparison.py` — head-to-head of 8 strategies
5. `05_risk_analysis.py` — VaR/CVaR methods, drawdown, stress tests
6. `06_ml_strategy.py` — GradientBoosting classifier backtest
7. `07_pairs_trading.py` — stat arb on cointegrated series
8. `08_walk_forward.py` — walk-forward validation + parameter sweep
9. `09_regime_aware_strategy.py` — switch between momentum / mean-reversion by Hurst
10. `10_full_research_pipeline.py` — 6 strategies combined via HRP on strategy returns
11. `11_fixed_income.py` — bond analytics, yield curve fit, duration hedge, CVaR opt

## Design Principles

- **Vectorized first.** Indicators and analytics are pure NumPy/Pandas; the event loop is explicit only where it must be (backtest engine).
- **Pure math, stateful classes.** Pricing functions are pure. Portfolio/Broker own the state.
- **Deterministic.** All RNG is seeded, zero network calls in the test suite.
- **No hidden globals.** Config flows via dataclasses / explicit function arguments.
- **Minimal deps.** numpy, pandas, scipy, scikit-learn. statsmodels is optional (native ADF fallback). No CVXPY, no torch.
- **Target-position sizing.** Signals produce target directions; the engine computes delta orders so repeated signals don't over-accumulate.

## Test Coverage

`pytest tests/ -q` → **304 passed in ~11s**

| File | Tests | Covers |
|---|---:|---|
| `test_core.py` | 26 | Position fills, portfolio cash/equity |
| `test_indicators.py` | 34 | All technical + statistical indicators |
| `test_options.py` | 46 | BS, IV, CRR, MC, Greeks, put-call parity, convergence |
| `test_portfolio.py` | 22 | MinVar, MaxSharpe, HRP, ERC |
| `test_risk.py` | 32 | VaR methods, drawdown, metrics |
| `test_backtest.py` | 13 | Engine end-to-end, broker |
| `test_strategies.py` | 21 | All 8 strategies |
| `test_analytics.py` | 29 | summary_stats, tearsheet, attribution |
| `test_ml.py` | 43 | features, regime, forecast |
| `test_fixed_income.py` | 25 | Bond math, duration hedge, NS/NSS |
| `test_simulation_kelly.py` | 19 | MC sim, Kelly, CVaR opt, factor model |
| `test_tca.py` | 4 | Transaction cost analysis |

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Module map + event flow
- [docs/QUICKSTART.md](docs/QUICKSTART.md) — 60-second tour
- [docs/STRATEGIES.md](docs/STRATEGIES.md) — Strategy reference + how to write your own

## License
MIT
