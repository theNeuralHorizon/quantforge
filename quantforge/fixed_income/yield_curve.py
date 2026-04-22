"""Yield-curve models and bootstrapping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares


@dataclass
class NelsonSiegel:
    """Nelson-Siegel 3-factor yield curve.

    y(t) = b0 + b1*(1-exp(-t/tau))/(t/tau) + b2*((1-exp(-t/tau))/(t/tau) - exp(-t/tau))
    """
    beta0: float = 0.03
    beta1: float = -0.02
    beta2: float = 0.02
    tau: float = 2.0

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        x = np.where(t == 0, 1e-12, t) / self.tau
        decay = (1 - np.exp(-x)) / x
        return self.beta0 + self.beta1 * decay + self.beta2 * (decay - np.exp(-x))

    def fit(self, maturities: Sequence[float], yields: Sequence[float]) -> "NelsonSiegel":
        m = np.asarray(maturities, dtype=float)
        y = np.asarray(yields, dtype=float)

        def residuals(params):
            self.beta0, self.beta1, self.beta2, self.tau = params
            self.tau = max(1e-3, self.tau)
            return self(m) - y

        params0 = [0.03, -0.02, 0.02, 2.0]
        res = least_squares(residuals, params0, bounds=([-0.1, -0.5, -0.5, 0.01], [0.2, 0.5, 0.5, 30]))
        self.beta0, self.beta1, self.beta2, self.tau = res.x
        return self


@dataclass
class NelsonSiegelSvensson:
    """NSS = NS + a second hump term."""
    beta0: float = 0.03
    beta1: float = -0.02
    beta2: float = 0.02
    beta3: float = 0.01
    tau1: float = 2.0
    tau2: float = 5.0

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        x1 = np.where(t == 0, 1e-12, t) / self.tau1
        x2 = np.where(t == 0, 1e-12, t) / self.tau2
        d1 = (1 - np.exp(-x1)) / x1
        d2 = (1 - np.exp(-x2)) / x2
        return (self.beta0 + self.beta1 * d1 + self.beta2 * (d1 - np.exp(-x1))
                + self.beta3 * (d2 - np.exp(-x2)))

    def fit(self, maturities: Sequence[float], yields: Sequence[float]) -> "NelsonSiegelSvensson":
        m = np.asarray(maturities, dtype=float)
        y = np.asarray(yields, dtype=float)

        def residuals(params):
            (self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2) = params
            self.tau1 = max(1e-3, self.tau1)
            self.tau2 = max(1e-3, self.tau2)
            return self(m) - y

        p0 = [0.03, -0.02, 0.02, 0.01, 2.0, 5.0]
        bounds = ([-0.1, -0.5, -0.5, -0.5, 0.01, 0.01], [0.2, 0.5, 0.5, 0.5, 30, 30])
        res = least_squares(residuals, p0, bounds=bounds)
        (self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2) = res.x
        return self


def discount_factor(rate: float, t: float, compounding: str = "continuous") -> float:
    if compounding == "continuous":
        return float(np.exp(-rate * t))
    if compounding == "annual":
        return float((1 + rate) ** (-t))
    if compounding == "semiannual":
        return float((1 + rate / 2) ** (-2 * t))
    raise ValueError(f"Unknown compounding: {compounding}")


def forward_rate(r1: float, t1: float, r2: float, t2: float) -> float:
    """Continuous forward rate between t1 and t2 given zero rates."""
    if t2 <= t1:
        return float("nan")
    return float((r2 * t2 - r1 * t1) / (t2 - t1))


def zero_to_par_yield(zeros: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Convert a list of (maturity, zero_rate) to par yields assuming semi-annual coupons.

    par_yield(T) = (1 - P(T)) / sum_{i=1..2T} P(i/2)   * 2
    """
    pairs = sorted(zeros)
    out = []
    for T_idx, (T, _) in enumerate(pairs):
        halves = np.arange(0.5, T + 1e-9, 0.5)
        pvs = []
        for t in halves:
            # linear interp on (m, r)
            ms = [m for m, _ in pairs]
            rs = [r for _, r in pairs]
            r = float(np.interp(t, ms, rs))
            pvs.append(discount_factor(r, t))
        pT = pvs[-1]
        par = (1 - pT) / sum(pvs) * 2
        out.append((T, par))
    return out


def bootstrap_zero_curve(
    bonds: Sequence[Tuple[float, float, float]]
) -> List[Tuple[float, float]]:
    """Bootstrap zero rates from a sequence of (maturity, coupon_rate, price) with face=100.

    Assumes bonds are ordered by maturity and pay semi-annually.
    Returns list of (maturity, zero_rate).
    """
    zeros: List[Tuple[float, float]] = []
    freq = 2
    for T, c, price in sorted(bonds, key=lambda x: x[0]):
        n = int(round(T * freq))
        cpn = 100 * c / freq
        pv_known = 0.0
        for i in range(1, n):
            t = i / freq
            r = float(np.interp(t, [m for m, _ in zeros], [r for _, r in zeros])) if zeros else 0.03
            pv_known += cpn * np.exp(-r * t)
        remaining = price - pv_known
        if remaining <= 0:
            z = zeros[-1][1] if zeros else 0.03
        else:
            z = float(-np.log(remaining / (cpn + 100)) / T)
        zeros.append((T, z))
    return zeros
