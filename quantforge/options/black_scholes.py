"""Black-Scholes-Merton European option pricing and implied vol."""
from __future__ import annotations

import math
from typing import Literal

from scipy.stats import norm


def d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return float("nan")
    return (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return float("nan")
    return d1(S, K, T, r, sigma, q) - sigma * math.sqrt(T)


def bs_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    D1 = d1(S, K, T, r, sigma, q)
    D2 = D1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm.cdf(D1) - K * math.exp(-r * T) * norm.cdf(D2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
    D1 = d1(S, K, T, r, sigma, q)
    D2 = D1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-D2) - S * math.exp(-q * T) * norm.cdf(-D1)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option: Literal["call", "put"] = "call", q: float = 0.0) -> float:
    return bs_call(S, K, T, r, sigma, q) if option.lower() == "call" else bs_put(S, K, T, r, sigma, q)


def bs_implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option: Literal["call", "put"] = "call",
    q: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Implied volatility via robust bisection bracketing + Newton refinement."""
    if price <= 0 or T <= 0:
        return float("nan")
    intrinsic = max((S - K) if option == "call" else (K - S), 0.0)
    if price < intrinsic * math.exp(-r * T) - 1e-10:
        return float("nan")

    # Bracket
    lo, hi = 1e-6, 5.0
    p_lo = bs_price(S, K, T, r, lo, option, q)
    p_hi = bs_price(S, K, T, r, hi, option, q)
    while p_hi < price and hi < 50.0:
        hi *= 2
        p_hi = bs_price(S, K, T, r, hi, option, q)
    if p_lo > price:
        return float("nan")

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = bs_price(S, K, T, r, mid, option, q)
        if abs(p - price) < tol:
            return mid
        if p < price:
            lo = mid
        else:
            hi = mid

    # Newton polish
    sigma = 0.5 * (lo + hi)
    for _ in range(20):
        p = bs_price(S, K, T, r, sigma, option, q)
        D1 = d1(S, K, T, r, sigma, q)
        v = S * math.exp(-q * T) * norm.pdf(D1) * math.sqrt(T)
        if v < 1e-12:
            break
        diff = p - price
        sigma -= diff / v
        sigma = max(1e-8, min(sigma, 10.0))
        if abs(diff) < tol:
            break
    return sigma
