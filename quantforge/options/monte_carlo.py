"""Monte Carlo pricing for European, Asian, barrier, and lookback options under GBM."""
from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import numpy as np


def _gbm_paths(
    S: float, T: float, r: float, sigma: float, q: float,
    steps: int, n_paths: int, antithetic: bool, seed: Optional[int],
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = T / steps
    half = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal((steps, half))
    if antithetic:
        z = np.concatenate([z, -z], axis=1)
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * math.sqrt(dt) * z
    log_paths = np.cumsum(drift + diffusion, axis=0)
    paths = S * np.exp(log_paths)
    paths = np.vstack([np.full((1, paths.shape[1]), S), paths])
    return paths


def _stderr(payoffs: np.ndarray, disc: float) -> float:
    return float(disc * payoffs.std(ddof=1) / math.sqrt(len(payoffs)))


def mc_european(
    S: float, K: float, T: float, r: float, sigma: float,
    option: Literal["call", "put"] = "call",
    q: float = 0.0,
    steps: int = 1, n_paths: int = 50_000,
    antithetic: bool = True, seed: Optional[int] = None,
) -> Tuple[float, float]:
    paths = _gbm_paths(S, T, r, sigma, q, steps, n_paths, antithetic, seed)
    ST = paths[-1]
    payoff = np.maximum(ST - K, 0.0) if option == "call" else np.maximum(K - ST, 0.0)
    disc = math.exp(-r * T)
    price = float(disc * payoff.mean())
    return price, _stderr(payoff, disc)


def mc_asian(
    S: float, K: float, T: float, r: float, sigma: float,
    option: Literal["call", "put"] = "call",
    q: float = 0.0,
    steps: int = 52, n_paths: int = 50_000,
    antithetic: bool = True, seed: Optional[int] = None,
    average: Literal["arithmetic", "geometric"] = "arithmetic",
) -> Tuple[float, float]:
    paths = _gbm_paths(S, T, r, sigma, q, steps, n_paths, antithetic, seed)
    if average == "arithmetic":
        avg = paths[1:].mean(axis=0)
    else:
        avg = np.exp(np.log(paths[1:]).mean(axis=0))
    payoff = np.maximum(avg - K, 0.0) if option == "call" else np.maximum(K - avg, 0.0)
    disc = math.exp(-r * T)
    price = float(disc * payoff.mean())
    return price, _stderr(payoff, disc)


def mc_barrier(
    S: float, K: float, T: float, r: float, sigma: float,
    barrier: float,
    barrier_type: Literal["up-and-out", "up-and-in", "down-and-out", "down-and-in"] = "down-and-out",
    option: Literal["call", "put"] = "call",
    q: float = 0.0, rebate: float = 0.0,
    steps: int = 100, n_paths: int = 50_000,
    antithetic: bool = True, seed: Optional[int] = None,
) -> Tuple[float, float]:
    paths = _gbm_paths(S, T, r, sigma, q, steps, n_paths, antithetic, seed)
    hit_up = (paths >= barrier).any(axis=0)
    hit_down = (paths <= barrier).any(axis=0)

    ST = paths[-1]
    vanilla = np.maximum(ST - K, 0.0) if option == "call" else np.maximum(K - ST, 0.0)

    if barrier_type == "up-and-out":
        active = ~hit_up
    elif barrier_type == "up-and-in":
        active = hit_up
    elif barrier_type == "down-and-out":
        active = ~hit_down
    else:
        active = hit_down

    payoff = np.where(active, vanilla, rebate)
    disc = math.exp(-r * T)
    return float(disc * payoff.mean()), _stderr(payoff, disc)


def mc_lookback(
    S: float, K: Optional[float], T: float, r: float, sigma: float,
    option: Literal["call", "put"] = "call",
    kind: Literal["fixed", "floating"] = "floating",
    q: float = 0.0,
    steps: int = 100, n_paths: int = 50_000,
    antithetic: bool = True, seed: Optional[int] = None,
) -> Tuple[float, float]:
    paths = _gbm_paths(S, T, r, sigma, q, steps, n_paths, antithetic, seed)
    smax = paths.max(axis=0)
    smin = paths.min(axis=0)
    ST = paths[-1]
    if kind == "floating":
        payoff = (ST - smin) if option == "call" else (smax - ST)
    else:
        assert K is not None
        payoff = np.maximum(smax - K, 0.0) if option == "call" else np.maximum(K - smin, 0.0)
    disc = math.exp(-r * T)
    return float(disc * payoff.mean()), _stderr(payoff, disc)
