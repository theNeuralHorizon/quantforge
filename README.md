# QuantForge

**A Full-Stack Quantitative Trading Research Platform**

![tests](https://img.shields.io/badge/tests-322%20passing-brightgreen) ![coverage](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue) ![license](https://img.shields.io/badge/license-MIT-blue) ![docker](https://img.shields.io/badge/docker-ready-2496ed) ![k8s](https://img.shields.io/badge/kubernetes-ready-326ce5)

QuantForge is an end-to-end quant research stack: Python library + hardened REST API + web terminal + dashboard + CLI, all in one repo.

**What's in the box**
- **Library** — 12 strategies, 25+ indicators, options pricing, portfolio optimizers, risk analytics, ML trainer
- **REST API** — FastAPI with API-key auth, rate limiting, Prometheus metrics, full OpenAPI docs
- **Web terminal** — Tailwind + ApexCharts + Alpine.js single-page app served at `/ui/`
- **Streamlit dashboard** — 8 pages for interactive research
- **CLI** — `python -m quantforge {price,iv,backtest,tournament,tearsheet}`
- **Deployment** — multi-stage Dockerfiles, docker-compose, hardened Kubernetes manifests with HPA + NetworkPolicy
- **CI/CD** — GitHub Actions with test matrix, bandit, safety, CodeQL, gitleaks, Trivy container scan
- **n8n integrations** — 4 ready-to-import workflow JSONs (daily VaR alerts, weekly tournament, monthly ML retrain, options webhook)
- **Security** — SECURITY.md with full threat model, CSP, HSTS, non-root containers, secret hashing

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
