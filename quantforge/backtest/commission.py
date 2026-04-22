"""Commission models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class CommissionModel(ABC):
    @abstractmethod
    def charge(self, price: float, quantity: float) -> float:
        ...


@dataclass
class NoCommission(CommissionModel):
    def charge(self, price: float, quantity: float) -> float:
        return 0.0


@dataclass
class FixedBpsCommission(CommissionModel):
    bps: float = 1.0

    def charge(self, price: float, quantity: float) -> float:
        return abs(price * quantity) * self.bps * 1e-4


@dataclass
class PerShareCommission(CommissionModel):
    per_share: float = 0.005
    minimum: float = 1.0

    def charge(self, price: float, quantity: float) -> float:
        return max(self.minimum, abs(quantity) * self.per_share)
