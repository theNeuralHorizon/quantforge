"""Example 05: VaR/CVaR methods, drawdown analytics, stress tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.data.synthetic import generate_gbm
from quantforge.risk.var import (
    historical_var, historical_cvar, parametric_var, parametric_cvar,
    cornish_fisher_var, monte_carlo_var,
)
from quantforge.risk.drawdown import drawdown_series, drawdown_table, max_drawdown
from quantforge.risk.stress_test import stress_scenarios, HISTORICAL_SHOCKS
from quantforge.analytics.tearsheet import tearsheet_text


def main() -> None:
    prices = generate_gbm(n=1000, s0=100, mu=0.1, sigma=0.28, seed=42)
    returns = prices.pct_change().dropna()

    print("=" * 70)
    print("Value-at-Risk methods (95% confidence, daily)")
    print("=" * 70)
    print(f"  Historical   : {historical_var(returns) * 100:+.3f}% loss")
    print(f"  Parametric   : {parametric_var(returns) * 100:+.3f}% loss")
    print(f"  Cornish-Fisher: {cornish_fisher_var(returns) * 100:+.3f}% loss")
    mc_v, mc_c = monte_carlo_var(float(returns.mean()), float(returns.std()), seed=1)
    print(f"  Monte Carlo  : VaR={mc_v * 100:+.3f}%  CVaR={mc_c * 100:+.3f}%")

    print("\n" + "=" * 70)
    print("CVaR (Expected Shortfall)")
    print("=" * 70)
    print(f"  Historical   : {historical_cvar(returns) * 100:+.3f}%")
    print(f"  Parametric   : {parametric_cvar(returns) * 100:+.3f}%")

    print("\n" + "=" * 70)
    print("Drawdown analysis")
    print("=" * 70)
    dd = drawdown_series(prices)
    mdd, peak, trough = max_drawdown(prices)
    print(f"  Max drawdown : {mdd:+.4f}  peak={peak.date()} trough={trough.date()}")
    print("  Top 5 drawdowns:")
    print(drawdown_table(prices, top_n=5).to_string(index=False))

    print("\n" + "=" * 70)
    print("Stress tests on a 60/30/10 equity/bond/gold portfolio")
    print("=" * 70)
    weights = pd.Series({"stocks": 0.6, "treasuries": 0.3, "gold": 0.1})
    mapping = {"stocks": "equity", "treasuries": "bond", "gold": "equity"}  # simplified
    sc = stress_scenarios(weights, mapping)
    print(sc.to_string(index=False))

    print("\n" + "=" * 70)
    print("Full tearsheet")
    print("=" * 70)
    print(tearsheet_text(prices, "GBM Asset"))


if __name__ == "__main__":
    main()
