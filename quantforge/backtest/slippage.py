"""Slippage models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class SlippageModel(ABC):
    @abstractmethod
    def adjust(self, price: float, quantity: float, volume: float, side: int) -> float:
        """Return adjusted fill price given mid price, order quantity, bar volume, and side (+1 buy / -1 sell)."""


@dataclass
class NoSlippage(SlippageModel):
    def adjust(self, price: float, quantity: float, volume: float, side: int) -> float:
        return price


@dataclass
class FixedBpsSlippage(SlippageModel):
    bps: float = 5.0

    def adjust(self, price: float, quantity: float, volume: float, side: int) -> float:
        return price * (1 + side * self.bps * 1e-4)


@dataclass
class VolumeImpactSlippage(SlippageModel):
    """Square-root market impact model: impact scales with sqrt(quantity / volume)."""
    eta: float = 0.1
    base_bps: float = 2.0

    def adjust(self, price: float, quantity: float, volume: float, side: int) -> float:
        participation = abs(quantity) / max(volume, 1.0)
        impact = self.eta * np.sqrt(participation)
        total = self.base_bps * 1e-4 + impact
        return price * (1 + side * total)
