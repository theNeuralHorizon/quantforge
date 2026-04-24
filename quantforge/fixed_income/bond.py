"""Vanilla bond pricing, YTM, duration, convexity, DV01."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def bond_cashflows(face: float, coupon: float, maturity: float, freq: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, amounts) arrays for a plain-vanilla coupon bond.

    `coupon` is the annual coupon *rate* (e.g. 0.05 = 5%); `face` is face value.
    """
    n_periods = int(round(maturity * freq))
    if n_periods <= 0:
        return np.array([]), np.array([])
    times = np.arange(1, n_periods + 1) / freq
    cpn = face * coupon / freq
    amounts = np.full(n_periods, cpn)
    amounts[-1] += face
    return times, amounts


def bond_price(face: float, coupon: float, ytm: float, maturity: float, freq: int = 2) -> float:
    """Clean price given yield to maturity."""
    times, amounts = bond_cashflows(face, coupon, maturity, freq)
    if len(times) == 0:
        return face
    disc = (1 + ytm / freq) ** (-times * freq)
    return float(np.dot(amounts, disc))


def bond_ytm(
    price: float, face: float, coupon: float, maturity: float,
    freq: int = 2, bounds: tuple[float, float] = (-0.5, 2.0),
) -> float:
    """Solve YTM from dirty price via Brent's method."""
    def f(y: float) -> float:
        return bond_price(face, coupon, y, maturity, freq) - price
    try:
        return float(brentq(f, bounds[0], bounds[1], xtol=1e-10))
    except ValueError:
        return float("nan")


def macaulay_duration(face: float, coupon: float, ytm: float, maturity: float, freq: int = 2) -> float:
    times, amounts = bond_cashflows(face, coupon, maturity, freq)
    if len(times) == 0:
        return 0.0
    disc = (1 + ytm / freq) ** (-times * freq)
    pv = amounts * disc
    price = pv.sum()
    if price == 0:
        return 0.0
    return float(np.dot(times, pv) / price)


def modified_duration(face: float, coupon: float, ytm: float, maturity: float, freq: int = 2) -> float:
    mac = macaulay_duration(face, coupon, ytm, maturity, freq)
    return mac / (1 + ytm / freq)


def convexity(face: float, coupon: float, ytm: float, maturity: float, freq: int = 2) -> float:
    times, amounts = bond_cashflows(face, coupon, maturity, freq)
    if len(times) == 0:
        return 0.0
    disc = (1 + ytm / freq) ** (-times * freq)
    pv = amounts * disc
    price = pv.sum()
    if price == 0:
        return 0.0
    t_in_periods = times * freq
    conv_periods = np.dot(t_in_periods * (t_in_periods + 1), pv) / price / (1 + ytm / freq) ** 2
    return float(conv_periods / freq**2)


def dv01(face: float, coupon: float, ytm: float, maturity: float, freq: int = 2) -> float:
    """Dollar value of a 1bp move."""
    p0 = bond_price(face, coupon, ytm, maturity, freq)
    p1 = bond_price(face, coupon, ytm + 1e-4, maturity, freq)
    return float(p0 - p1)


def accrued_interest(face: float, coupon: float, days_since_last: int, days_in_period: int = 182) -> float:
    """Simple straight-line accrued interest."""
    return face * coupon * (days_since_last / days_in_period) / 2
