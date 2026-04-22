"""Cox-Ross-Rubinstein binomial trees (European + American)."""
from __future__ import annotations

import math
from typing import Literal

import numpy as np


def crr_price(
    S: float, K: float, T: float, r: float, sigma: float,
    option: Literal["call", "put"] = "call",
    steps: int = 200,
    q: float = 0.0,
) -> float:
    """Cox-Ross-Rubinstein European option price."""
    if T <= 0:
        return max((S - K) if option == "call" else (K - S), 0.0)
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    j = np.arange(steps + 1)
    ST = S * (u ** (steps - j)) * (d ** j)
    if option == "call":
        values = np.maximum(ST - K, 0.0)
    else:
        values = np.maximum(K - ST, 0.0)

    for _ in range(steps):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
    return float(values[0])


def crr_american(
    S: float, K: float, T: float, r: float, sigma: float,
    option: Literal["call", "put"] = "put",
    steps: int = 200,
    q: float = 0.0,
) -> float:
    """American option via CRR with early-exercise check at each node."""
    if T <= 0:
        return max((S - K) if option == "call" else (K - S), 0.0)
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    j = np.arange(steps + 1)
    ST = S * (u ** (steps - j)) * (d ** j)
    if option == "call":
        values = np.maximum(ST - K, 0.0)
    else:
        values = np.maximum(K - ST, 0.0)

    for step in range(steps - 1, -1, -1):
        i = np.arange(step + 1)
        S_node = S * (u ** (step - i)) * (d ** i)
        exercise = np.maximum((S_node - K) if option == "call" else (K - S_node), 0.0)
        continuation = disc * (p * values[:-1] + (1 - p) * values[1:])
        values = np.maximum(continuation, exercise)
    return float(values[0])
