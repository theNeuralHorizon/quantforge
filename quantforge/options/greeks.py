"""Closed-form Black-Scholes Greeks."""
from __future__ import annotations

import math
from typing import Literal

from scipy.stats import norm

from quantforge.options.black_scholes import d1, d2


def delta(S: float, K: float, T: float, r: float, sigma: float, option: Literal["call", "put"] = "call", q: float = 0.0) -> float:
    D1 = d1(S, K, T, r, sigma, q)
    if option == "call":
        return math.exp(-q * T) * norm.cdf(D1)
    return math.exp(-q * T) * (norm.cdf(D1) - 1)


def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    D1 = d1(S, K, T, r, sigma, q)
    if math.isnan(D1) or sigma <= 0 or T <= 0:
        return 0.0
    return math.exp(-q * T) * norm.pdf(D1) / (S * sigma * math.sqrt(T))


def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Vega per 1 unit of vol (i.e., per 1.00 change). Divide by 100 for per-volpoint."""
    D1 = d1(S, K, T, r, sigma, q)
    if math.isnan(D1):
        return 0.0
    return S * math.exp(-q * T) * norm.pdf(D1) * math.sqrt(T)


def theta(S: float, K: float, T: float, r: float, sigma: float, option: Literal["call", "put"] = "call", q: float = 0.0) -> float:
    """Theta per year. Divide by 365 for per day."""
    if T <= 0:
        return 0.0
    D1 = d1(S, K, T, r, sigma, q)
    D2 = d2(S, K, T, r, sigma, q)
    term1 = -(S * math.exp(-q * T) * norm.pdf(D1) * sigma) / (2 * math.sqrt(T))
    if option == "call":
        term2 = q * S * math.exp(-q * T) * norm.cdf(D1)
        term3 = -r * K * math.exp(-r * T) * norm.cdf(D2)
        return term1 + term2 + term3
    term2 = -q * S * math.exp(-q * T) * norm.cdf(-D1)
    term3 = r * K * math.exp(-r * T) * norm.cdf(-D2)
    return term1 + term2 + term3


def rho(S: float, K: float, T: float, r: float, sigma: float, option: Literal["call", "put"] = "call", q: float = 0.0) -> float:
    if T <= 0:
        return 0.0
    D2 = d2(S, K, T, r, sigma, q)
    if option == "call":
        return K * T * math.exp(-r * T) * norm.cdf(D2)
    return -K * T * math.exp(-r * T) * norm.cdf(-D2)


def all_greeks(S: float, K: float, T: float, r: float, sigma: float, option: Literal["call", "put"] = "call", q: float = 0.0) -> dict[str, float]:
    return {
        "delta": delta(S, K, T, r, sigma, option, q),
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q),
        "theta": theta(S, K, T, r, sigma, option, q),
        "rho": rho(S, K, T, r, sigma, option, q),
    }
