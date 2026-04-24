"""Multi-leg options strategies: straddles, strangles, spreads, condors, butterflies.

Each strategy is modelled as a list of `Leg`s (quantity, option type, strike);
we price them by summing Black-Scholes prices of the individual legs and
aggregating the Greeks. Break-evens and max-loss/profit are computed
analytically where possible, numerically otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from quantforge.options.black_scholes import bs_call, bs_put
from quantforge.options.greeks import all_greeks


@dataclass(frozen=True)
class Leg:
    quantity: float                              # signed: +1 = long, -1 = short
    option: Literal["call", "put"]
    strike: float
    expiry_years: float                          # T; allows calendar spreads


@dataclass
class StrategyQuote:
    strategy: str
    legs: list[Leg]
    net_premium: float                           # +ve = net debit, -ve = net credit
    greeks: dict                                 # delta, gamma, vega, theta, rho
    max_profit: float | None                  # None = unbounded
    max_loss: float | None                    # None = unbounded (short legs)
    break_evens: list[float]
    payoff: tuple[np.ndarray, np.ndarray]        # (spot_grid, pnl_grid) at expiry

    def summary(self) -> str:
        legs_desc = ", ".join(
            f"{'+'if l.quantity>0 else ''}{l.quantity:g} {l.option} {l.strike:g}" for l in self.legs
        )
        mp = "∞" if self.max_profit is None else f"{self.max_profit:+.4f}"
        ml = "-∞" if self.max_loss is None else f"{self.max_loss:+.4f}"
        be = ", ".join(f"{b:.2f}" for b in self.break_evens) or "n/a"
        return (
            f"{self.strategy}: {legs_desc}\n"
            f"  premium={self.net_premium:+.4f}  max_profit={mp}  max_loss={ml}\n"
            f"  break_evens=[{be}]  delta={self.greeks['delta']:+.3f} vega={self.greeks['vega']:+.2f}"
        )


def _price_leg(leg: Leg, S: float, r: float, sigma: float, q: float) -> float:
    fn = bs_call if leg.option == "call" else bs_put
    return leg.quantity * fn(S, leg.strike, leg.expiry_years, r, sigma, q)


def _sum_greeks(legs: list[Leg], S: float, r: float, sigma: float, q: float) -> dict:
    agg = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    for leg in legs:
        g = all_greeks(S, leg.strike, leg.expiry_years, r, sigma, leg.option, q)
        for k in agg:
            agg[k] += leg.quantity * g[k]
    return agg


def _payoff_at_expiry(legs: list[Leg], net_premium: float,
                       s_min: float, s_max: float, n: int = 201) -> tuple[np.ndarray, np.ndarray]:
    spots = np.linspace(max(0.01, s_min), s_max, n)
    pnl = np.full(n, -net_premium, dtype=float)
    for leg in legs:
        if leg.option == "call":
            pnl += leg.quantity * np.maximum(spots - leg.strike, 0.0)
        else:
            pnl += leg.quantity * np.maximum(leg.strike - spots, 0.0)
    return spots, pnl


def _find_break_evens(spots: np.ndarray, pnl: np.ndarray) -> list[float]:
    """Linear interpolation between sign changes."""
    out: list[float] = []
    for i in range(1, len(pnl)):
        if pnl[i - 1] == 0:
            out.append(float(spots[i - 1]))
        elif np.sign(pnl[i - 1]) != np.sign(pnl[i]):
            # linear interp
            x0, x1 = spots[i - 1], spots[i]
            y0, y1 = pnl[i - 1], pnl[i]
            be = x0 - y0 * (x1 - x0) / (y1 - y0)
            out.append(float(be))
    return out


def _quote(
    strategy_name: str, legs: list[Leg],
    S: float, r: float, sigma: float, q: float,
    max_profit: float | None = None, max_loss: float | None = None,
    grid_range: float = 0.6,
) -> StrategyQuote:
    """Compute net premium, greeks, break-evens, payoff grid. If max_profit /
    max_loss are not supplied, scan the payoff grid for approximate bounds."""
    net_premium = sum(_price_leg(l, S, r, sigma, q) for l in legs)
    greeks = _sum_greeks(legs, S, r, sigma, q)

    s_min = S * (1 - grid_range)
    s_max = S * (1 + grid_range)
    spots, pnl = _payoff_at_expiry(legs, net_premium, s_min, s_max)

    # Heuristic: if grid-end payoffs are finite and not the max, extend the grid
    # to probe for unbounded behaviour.
    if max_profit is None or max_loss is None:
        ext_spots, ext_pnl = _payoff_at_expiry(legs, net_premium,
                                                 S * (1 - 5 * grid_range),
                                                 S * (1 + 5 * grid_range),
                                                 n=501)
        # Compare slopes at far tails; nonzero slope → unbounded
        left_slope = ext_pnl[1] - ext_pnl[0]
        right_slope = ext_pnl[-1] - ext_pnl[-2]
        # Fallback derivations only if caller didn't pin them
        if max_profit is None:
            if right_slope > 1e-6 or left_slope < -1e-6:
                # possibly unbounded
                max_profit = float("inf") if (right_slope > 0 or left_slope < 0) else float(ext_pnl.max())
            else:
                max_profit = float(ext_pnl.max())
        if max_loss is None:
            if right_slope < -1e-6 or left_slope > 1e-6:
                max_loss = float("-inf") if (right_slope < 0 or left_slope > 0) else float(ext_pnl.min())
            else:
                max_loss = float(ext_pnl.min())

    return StrategyQuote(
        strategy=strategy_name, legs=legs, net_premium=float(net_premium),
        greeks={k: float(v) for k, v in greeks.items()},
        max_profit=(None if max_profit == float("inf") else max_profit),
        max_loss=(None if max_loss == float("-inf") else max_loss),
        break_evens=_find_break_evens(spots, pnl),
        payoff=(spots, pnl),
    )


# =============================================================================
# Named strategies
# =============================================================================
def straddle(S: float, K: float, T: float, r: float, sigma: float,
              q: float = 0.0, direction: Literal["long", "short"] = "long") -> StrategyQuote:
    sign = 1 if direction == "long" else -1
    legs = [Leg(sign * 1.0, "call", K, T), Leg(sign * 1.0, "put", K, T)]
    mp = None if direction == "long" else -(bs_call(S, K, T, r, sigma, q) + bs_put(S, K, T, r, sigma, q))
    ml = -(bs_call(S, K, T, r, sigma, q) + bs_put(S, K, T, r, sigma, q)) if direction == "long" else None
    return _quote(f"{direction} straddle", legs, S, r, sigma, q,
                   max_profit=mp, max_loss=ml)


def strangle(S: float, put_strike: float, call_strike: float, T: float,
              r: float, sigma: float, q: float = 0.0,
              direction: Literal["long", "short"] = "long") -> StrategyQuote:
    if put_strike >= call_strike:
        raise ValueError("put_strike must be < call_strike for a strangle")
    sign = 1 if direction == "long" else -1
    legs = [Leg(sign * 1.0, "put", put_strike, T), Leg(sign * 1.0, "call", call_strike, T)]
    return _quote(f"{direction} strangle", legs, S, r, sigma, q)


def bull_call_spread(S: float, lower: float, upper: float, T: float,
                      r: float, sigma: float, q: float = 0.0) -> StrategyQuote:
    if lower >= upper:
        raise ValueError("lower strike must be < upper strike")
    legs = [Leg(+1.0, "call", lower, T), Leg(-1.0, "call", upper, T)]
    debit = bs_call(S, lower, T, r, sigma, q) - bs_call(S, upper, T, r, sigma, q)
    return _quote("bull call spread", legs, S, r, sigma, q,
                   max_profit=(upper - lower) - debit, max_loss=-debit)


def bear_put_spread(S: float, lower: float, upper: float, T: float,
                     r: float, sigma: float, q: float = 0.0) -> StrategyQuote:
    if lower >= upper:
        raise ValueError("lower strike must be < upper strike")
    legs = [Leg(+1.0, "put", upper, T), Leg(-1.0, "put", lower, T)]
    debit = bs_put(S, upper, T, r, sigma, q) - bs_put(S, lower, T, r, sigma, q)
    return _quote("bear put spread", legs, S, r, sigma, q,
                   max_profit=(upper - lower) - debit, max_loss=-debit)


def iron_condor(S: float, p_long: float, p_short: float, c_short: float, c_long: float,
                 T: float, r: float, sigma: float, q: float = 0.0) -> StrategyQuote:
    """Iron condor: sell a put spread + sell a call spread.

    Strikes ordered: p_long < p_short < c_short < c_long.
    """
    if not (p_long < p_short < c_short < c_long):
        raise ValueError("strikes must satisfy p_long < p_short < c_short < c_long")
    legs = [
        Leg(+1.0, "put", p_long, T),   Leg(-1.0, "put", p_short, T),
        Leg(-1.0, "call", c_short, T), Leg(+1.0, "call", c_long, T),
    ]
    -(bs_put(S, p_long, T, r, sigma, q) - bs_put(S, p_short, T, r, sigma, q)
               + bs_call(S, c_long, T, r, sigma, q) - bs_call(S, c_short, T, r, sigma, q))
    # credit is negative of net_premium (net_premium is what you pay);
    # collection = -net_premium, positive if net_credit.
    wing_width = max(p_short - p_long, c_long - c_short)
    max_profit = -sum(_price_leg(l, S, r, sigma, q) for l in legs)   # net credit received
    max_loss = -(wing_width - max_profit)
    return _quote("iron condor", legs, S, r, sigma, q,
                   max_profit=max_profit, max_loss=max_loss)


def butterfly(S: float, lower: float, center: float, upper: float, T: float,
               r: float, sigma: float, q: float = 0.0,
               option: Literal["call", "put"] = "call") -> StrategyQuote:
    """Long butterfly: +1 lower, -2 center, +1 upper."""
    if not (lower < center < upper):
        raise ValueError("strikes must satisfy lower < center < upper")
    if abs((center - lower) - (upper - center)) > 1e-9:
        # asymmetric butterfly — still valid but note
        pass
    legs = [Leg(+1.0, option, lower, T),
            Leg(-2.0, option, center, T),
            Leg(+1.0, option, upper, T)]
    debit = sum(_price_leg(l, S, r, sigma, q) for l in legs)
    wing_width = min(center - lower, upper - center)
    return _quote(f"long {option} butterfly", legs, S, r, sigma, q,
                   max_profit=wing_width - debit, max_loss=-debit)


def calendar_spread(S: float, K: float, T_near: float, T_far: float,
                     r: float, sigma: float, q: float = 0.0,
                     option: Literal["call", "put"] = "call") -> StrategyQuote:
    """Calendar spread: short near-dated, long far-dated at same strike.

    Profits from time decay differential; assumes same vol for both (a
    conservative pricing simplification).
    """
    if T_near >= T_far:
        raise ValueError("T_near must be < T_far")
    legs = [Leg(-1.0, option, K, T_near), Leg(+1.0, option, K, T_far)]
    return _quote(f"{option} calendar spread", legs, S, r, sigma, q)


def collar(S: float, put_strike: float, call_strike: float, T: float,
            r: float, sigma: float, q: float = 0.0) -> StrategyQuote:
    """Stock + long put + short call.

    Models the underlying as an imaginary zero-premium "call at strike 0"
    so we can express it as pure option legs for the payoff math.
    """
    if put_strike > call_strike:
        raise ValueError("put_strike must be <= call_strike for a collar")
    # Use a call at strike 0 as a stock proxy (P(S-0, 0) = S)
    legs = [Leg(+1.0, "call", 0.0001, T),      # ~ stock
            Leg(+1.0, "put", put_strike, T),
            Leg(-1.0, "call", call_strike, T)]
    # Skip actually pricing the stock proxy — just use real S - premiums.
    opt_premium = bs_put(S, put_strike, T, r, sigma, q) - bs_call(S, call_strike, T, r, sigma, q)
    net_premium = S + opt_premium
    greeks = _sum_greeks(
        [Leg(+1.0, "put", put_strike, T), Leg(-1.0, "call", call_strike, T)], S, r, sigma, q
    )
    greeks["delta"] += 1.0  # underlying delta
    spots, pnl = _payoff_at_expiry(
        [Leg(+1.0, "put", put_strike, T), Leg(-1.0, "call", call_strike, T)],
        opt_premium, S * 0.4, S * 1.6,
    )
    pnl = pnl + (spots - S)  # add underlying P&L
    return StrategyQuote(
        strategy="collar", legs=legs, net_premium=float(net_premium),
        greeks=greeks,
        max_profit=float(call_strike - S - opt_premium),
        max_loss=float(put_strike - S - opt_premium),
        break_evens=_find_break_evens(spots, pnl),
        payoff=(spots, pnl),
    )
