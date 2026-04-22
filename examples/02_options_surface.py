"""Example 02: Build an implied-vol surface, verify model convergence, price exotics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.options.black_scholes import bs_call, bs_put, bs_implied_vol
from quantforge.options.greeks import all_greeks
from quantforge.options.binomial import crr_price, crr_american
from quantforge.options.monte_carlo import mc_european, mc_asian, mc_barrier, mc_lookback


def main() -> None:
    S, r, q, T = 100.0, 0.04, 0.0, 1.0

    print("=" * 70)
    print("Black-Scholes verification: put-call parity")
    print("=" * 70)
    for K in [80, 90, 100, 110, 120]:
        sigma = 0.2
        c = bs_call(S, K, T, r, sigma, q)
        p = bs_put(S, K, T, r, sigma, q)
        parity = c - p - (S - K * np.exp(-r * T))
        print(f"  K={K:>3}  call={c:7.4f}  put={p:7.4f}  parity={parity:+.2e}")

    print("\n" + "=" * 70)
    print("Implied Vol smile (synthetic: 18% low strikes, 22% high strikes)")
    print("=" * 70)
    strikes = np.array([80, 90, 95, 100, 105, 110, 120])
    true_vols = 0.20 + 0.02 * (strikes - 100) / 100  # asymmetric smile
    iv_rows = []
    for K, sv in zip(strikes, true_vols):
        c = bs_call(S, K, T, r, sv, q)
        iv = bs_implied_vol(c, S, K, T, r, "call", q)
        iv_rows.append({"K": K, "true_vol": sv, "mkt_price": c, "iv": iv})
    print(pd.DataFrame(iv_rows).round(6).to_string(index=False))

    print("\n" + "=" * 70)
    print("CRR convergence to Black-Scholes (ATM call)")
    print("=" * 70)
    bs = bs_call(100, 100, 1.0, 0.05, 0.2)
    print(f"  Black-Scholes: {bs:.6f}")
    for n in [10, 50, 100, 500, 2000]:
        price = crr_price(100, 100, 1.0, 0.05, 0.2, "call", n)
        print(f"  CRR N={n:>4}: {price:.6f}   (|diff|={abs(price - bs):.6f})")

    print("\n" + "=" * 70)
    print("American vs European put (same parameters)")
    print("=" * 70)
    euro = crr_price(100, 100, 1.0, 0.05, 0.3, "put", 500)
    amer = crr_american(100, 100, 1.0, 0.05, 0.3, "put", 500)
    print(f"  Euro put : {euro:.4f}")
    print(f"  Am put   : {amer:.4f}  (premium = {amer - euro:+.4f})")

    print("\n" + "=" * 70)
    print("Greeks for ATM 1Y 20% vol call")
    print("=" * 70)
    greeks = all_greeks(100, 100, 1.0, 0.05, 0.2, "call")
    for k, v in greeks.items():
        print(f"  {k:>6}: {v:+.4f}")

    print("\n" + "=" * 70)
    print("Monte Carlo exotics")
    print("=" * 70)
    euro, se = mc_european(100, 100, 1.0, 0.05, 0.2, "call", seed=7)
    print(f"  MC European  : {euro:.4f} (+- {se:.4f})  |  BS: {bs_call(100, 100, 1.0, 0.05, 0.2):.4f}")
    asian, se = mc_asian(100, 100, 1.0, 0.05, 0.2, "call", seed=7)
    print(f"  MC Asian     : {asian:.4f} (+- {se:.4f})  (expect < European)")
    barrier, se = mc_barrier(100, 100, 1.0, 0.05, 0.2, barrier=80,
                              barrier_type="down-and-out", option="call", seed=7)
    print(f"  MC Down-&-out: {barrier:.4f} (+- {se:.4f})  (expect < European)")
    lb, se = mc_lookback(100, None, 1.0, 0.05, 0.2, "call", kind="floating", seed=7)
    print(f"  MC Lookback  : {lb:.4f} (+- {se:.4f})  (expect > European)")


if __name__ == "__main__":
    main()
