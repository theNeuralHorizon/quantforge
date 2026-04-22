# QuantForge — Quickstart

## Install

```bash
pip install -r requirements.txt
# or, for an editable install:
pip install -e .
```

## 60-second tour

### Price an option
```bash
python -m quantforge price --S 100 --K 100 --T 1 --sigma 0.2
```

### Solve implied volatility
```bash
python -m quantforge iv --price 10.45 --S 100 --K 100 --T 1 --r 0.04
```

### Run a strategy backtest
```bash
python -m quantforge backtest --strategy ma --params fast=20 slow=100 --bars 600
```

### Strategy tournament
```bash
python -m quantforge tournament --bars 600 --seed 11
```

### Launch the dashboard
```bash
streamlit run quantforge/dashboard/app.py
```

## Python API

### Backtest in 10 lines
```python
from quantforge.data.synthetic import generate_panel
from quantforge.strategies import MACrossoverStrategy
from quantforge.backtest import BacktestEngine
from quantforge.analytics.tearsheet import tearsheet_text

panel = generate_panel(["AAA", "BBB", "CCC"], n=600, seed=11)
strat = MACrossoverStrategy(fast=20, slow=100)
engine = BacktestEngine(strategy=strat, data=panel, initial_capital=100_000, sizing_fraction=0.25)
res = engine.run()
print(tearsheet_text(res.equity_curve, "MA 20/100"))
```

### Price an option and get Greeks
```python
from quantforge.options.black_scholes import bs_call, bs_implied_vol
from quantforge.options.greeks import all_greeks

price = bs_call(S=100, K=100, T=1.0, r=0.04, sigma=0.25)
greeks = all_greeks(S=100, K=100, T=1.0, r=0.04, sigma=0.25, option="call")
iv = bs_implied_vol(price, S=100, K=100, T=1.0, r=0.04)
```

### Portfolio optimization
```python
import numpy as np
from quantforge.portfolio.markowitz import max_sharpe, efficient_frontier
from quantforge.portfolio.hrp import hierarchical_risk_parity

cov = np.array([[0.04, 0.01, 0.02], [0.01, 0.09, 0.03], [0.02, 0.03, 0.16]])
mu  = np.array([0.08, 0.12, 0.15])
weights = max_sharpe(mu, cov, risk_free=0.02)
hrp_w   = hierarchical_risk_parity(cov)
ef      = efficient_frontier(mu, cov, n_points=30)
```

### Risk analytics on a return stream
```python
from quantforge.data.synthetic import generate_gbm
from quantforge.risk.var import historical_var, cornish_fisher_var, historical_cvar
from quantforge.risk.drawdown import max_drawdown, drawdown_table
from quantforge.analytics.performance import summary_stats

prices = generate_gbm(1000, mu=0.1, sigma=0.25, seed=1)
returns = prices.pct_change().dropna()
print(f"VaR(95) hist: {historical_var(returns) * 100:.2f}%")
print(f"CVaR(95):     {historical_cvar(returns) * 100:.2f}%")
print(summary_stats(prices))
```

## Running the examples

All examples live in `examples/`. Run from the repo root:

```bash
PYTHONPATH=. python examples/01_data_and_indicators.py
PYTHONPATH=. python examples/02_options_surface.py
PYTHONPATH=. python examples/03_portfolio_optimization.py
PYTHONPATH=. python examples/04_strategy_comparison.py
PYTHONPATH=. python examples/05_risk_analysis.py
PYTHONPATH=. python examples/06_ml_strategy.py
PYTHONPATH=. python examples/07_pairs_trading.py
PYTHONPATH=. python examples/08_walk_forward.py
PYTHONPATH=. python examples/09_regime_aware_strategy.py
PYTHONPATH=. python examples/10_full_research_pipeline.py   # ~2-3 min
```

## Running tests

```bash
pytest tests/ -q
# 266 passed in ~10 seconds
```

## Using real market data

```python
from quantforge.data.loader import DataLoader

dl = DataLoader(cache_dir="data/cache")
df = dl.yfinance("SPY", "2020-01-01", "2024-12-31")
```

(Requires `pip install yfinance` and network access. Synthetic data is used everywhere in the test suite.)
