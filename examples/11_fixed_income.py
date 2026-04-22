"""Example 11: Fixed-income analytics (bonds + yield curve + CVaR optimization)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.fixed_income.bond import (
    bond_price, bond_ytm, macaulay_duration, modified_duration, convexity, dv01,
)
from quantforge.fixed_income.yield_curve import (
    NelsonSiegel, NelsonSiegelSvensson, bootstrap_zero_curve,
    discount_factor, forward_rate, zero_to_par_yield,
)
from quantforge.portfolio.cvar_opt import minimize_cvar, historical_cvar_from_scenarios


def main() -> None:
    print("=" * 70)
    print("Bond analytics: 10y 5% coupon, face=100")
    print("=" * 70)
    for y in [0.03, 0.04, 0.05, 0.06]:
        p = bond_price(100, 0.05, y, 10, 2)
        mac = macaulay_duration(100, 0.05, y, 10, 2)
        mod = modified_duration(100, 0.05, y, 10, 2)
        conv = convexity(100, 0.05, y, 10, 2)
        print(f"  YTM={y:.2%} -> price={p:7.4f}  Mac={mac:.3f}  Mod={mod:.3f}  Conv={conv:.3f}")

    print("\n" + "=" * 70)
    print("Duration hedge verification")
    print("=" * 70)
    p0 = bond_price(100, 0.05, 0.04, 10, 2)
    md = modified_duration(100, 0.05, 0.04, 10, 2)
    cv = convexity(100, 0.05, 0.04, 10, 2)
    for bp in [1, 10, 50, 100]:
        shock = bp * 1e-4
        p1 = bond_price(100, 0.05, 0.04 + shock, 10, 2)
        linear_pred = p0 * (1 - md * shock)
        taylor_pred = p0 * (1 - md * shock + 0.5 * cv * shock**2)
        print(f"  +{bp}bp: actual={p1:.4f}  linear={linear_pred:.4f}  taylor(2nd)={taylor_pred:.4f}")

    print("\n" + "=" * 70)
    print("Nelson-Siegel yield-curve fit")
    print("=" * 70)
    mats = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    # Humped curve with a peak at ~7y
    obs = np.array([0.025, 0.028, 0.033, 0.039, 0.043, 0.047, 0.049, 0.048, 0.046, 0.044])
    ns = NelsonSiegel().fit(mats, obs)
    print(f"  Fitted parameters: beta0={ns.beta0:.4f}  beta1={ns.beta1:.4f}  beta2={ns.beta2:.4f}  tau={ns.tau:.3f}")
    rows = []
    for m, y in zip(mats, obs):
        rows.append({"maturity": m, "observed": y, "fitted": ns(m).item()})
    print(pd.DataFrame(rows).round(5).to_string(index=False))

    nss = NelsonSiegelSvensson().fit(mats, obs)
    print(f"\n  NSS: beta0={nss.beta0:.4f} b1={nss.beta1:.4f} b2={nss.beta2:.4f} b3={nss.beta3:.4f} tau1={nss.tau1:.3f} tau2={nss.tau2:.3f}")
    preds_nss = [float(nss(m)) for m in mats]
    rmse_ns = float(np.sqrt(np.mean((np.array([ns(m).item() for m in mats]) - obs) ** 2)))
    rmse_nss = float(np.sqrt(np.mean((np.array(preds_nss) - obs) ** 2)))
    print(f"  RMSE:  NS={rmse_ns:.6f}  NSS={rmse_nss:.6f}")

    print("\n" + "=" * 70)
    print("Forward rates derived from NS curve")
    print("=" * 70)
    for (t1, t2) in [(1, 2), (2, 5), (5, 10), (10, 20)]:
        f = forward_rate(float(ns(t1)), t1, float(ns(t2)), t2)
        print(f"  {t1}y->{t2}y fwd: {f:.4f}")

    print("\n" + "=" * 70)
    print("CVaR optimization (3 assets)")
    print("=" * 70)
    rng = np.random.default_rng(42)
    # Heavy left-tail distribution to show the point of CVaR
    sims = rng.multivariate_normal([0.10/252, 0.08/252, 0.05/252],
                                    np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]) / 252,
                                    size=2000)
    # inject tail events
    for k in range(60):
        i = rng.integers(0, 2000)
        sims[i] -= rng.exponential(0.04, 3)

    from quantforge.portfolio.markowitz import min_variance, max_sharpe
    mu = sims.mean(axis=0)
    cov = np.cov(sims, rowvar=False)
    w_mv = min_variance(cov)
    w_ms = max_sharpe(mu, cov)
    w_cvar = minimize_cvar(sims, alpha=0.95)

    for name, w in [("Min-Var", w_mv), ("Max-Sharpe", w_ms), ("Min-CVaR(95)", w_cvar)]:
        cvar = historical_cvar_from_scenarios(w, sims, 0.95)
        vol = float(np.sqrt(w @ cov @ w))
        ret = float(w @ mu)
        print(f"  {name:>14}  w={np.round(w, 3)}  ret={ret:.5f}  vol={vol:.4f}  CVaR(95)={cvar:.5f}")


if __name__ == "__main__":
    main()
