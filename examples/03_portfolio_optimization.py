"""Example 03: Efficient frontier, max Sharpe, ERC, Black-Litterman, HRP."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.data.synthetic import generate_correlated_returns
from quantforge.portfolio.markowitz import (
    mean_variance, min_variance, max_sharpe, efficient_frontier,
)
from quantforge.portfolio.risk_parity import equal_risk_contribution
from quantforge.portfolio.black_litterman import black_litterman
from quantforge.portfolio.hrp import hierarchical_risk_parity


def main() -> None:
    symbols = ["TECH", "HEALTH", "ENERGY", "BONDS", "GOLD"]
    rng = np.random.default_rng(42)
    cov_factor = rng.normal(0, 0.01, (5, 5))
    cov = 0.5 * (cov_factor @ cov_factor.T) + np.diag([0.040, 0.030, 0.060, 0.005, 0.020])
    returns = generate_correlated_returns(n=504, symbols=symbols, cov=cov / 252, seed=42)
    mu_hat = returns.mean().values * 252
    cov_hat = returns.cov().values * 252

    print("=" * 70)
    print("Annualized expected returns and volatilities")
    print("=" * 70)
    vols = np.sqrt(np.diag(cov_hat))
    summary = pd.DataFrame({"exp_return": mu_hat, "vol": vols, "sharpe": mu_hat / vols}, index=symbols)
    print(summary.round(4))

    print("\n" + "=" * 70)
    print("Classic long-only optimizers")
    print("=" * 70)
    w_minvar = min_variance(cov_hat)
    w_ms = max_sharpe(mu_hat, cov_hat, risk_free=0.02)
    w_mv = mean_variance(mu_hat, cov_hat, gamma=3.0)
    w_erc = equal_risk_contribution(cov_hat)
    w_hrp = hierarchical_risk_parity(cov_hat)

    wdf = pd.DataFrame({
        "MinVar": w_minvar,
        "MaxSharpe": w_ms,
        "MeanVar(g=3)": w_mv,
        "ERC": w_erc,
        "HRP": w_hrp,
    }, index=symbols)
    print(wdf.round(3))

    print("\n" + "=" * 70)
    print("Portfolio stats")
    print("=" * 70)
    for name, w in wdf.items():
        p_ret = w.values @ mu_hat
        p_vol = float(np.sqrt(w.values @ cov_hat @ w.values))
        print(f"  {name:>14} ret={p_ret:+.4f}  vol={p_vol:.4f}  sharpe={(p_ret-0.02)/p_vol:+.3f}")

    print("\n" + "=" * 70)
    print("Black-Litterman with investor view: TECH will outperform HEALTH by 3%")
    print("=" * 70)
    caps = np.array([1500, 800, 400, 2000, 300])
    # TECH - HEALTH = 0.03
    P = np.zeros((1, len(symbols)))
    P[0, symbols.index("TECH")] = 1
    P[0, symbols.index("HEALTH")] = -1
    Q = np.array([0.03])
    bl_mu, bl_cov = black_litterman(cov_hat, caps, risk_aversion=2.5, P=P, Q=Q)
    bl_df = pd.DataFrame({"prior_implied": (2.5 * cov_hat @ (caps / caps.sum())), "posterior": bl_mu}, index=symbols)
    print(bl_df.round(4))

    print("\n" + "=" * 70)
    print("Efficient frontier (10 points)")
    print("=" * 70)
    ef = efficient_frontier(mu_hat, cov_hat, n_points=10)
    print(ef[["target_return", "risk", "return"]].round(4))


if __name__ == "__main__":
    main()
