"""Finance constants used across the library."""
from __future__ import annotations

TRADING_DAYS_YEAR: int = 252
TRADING_HOURS_DAY: float = 6.5
SECONDS_PER_TRADING_DAY: int = int(TRADING_HOURS_DAY * 3600)
DEFAULT_RISK_FREE_RATE: float = 0.04
DEFAULT_CAPITAL: float = 100_000.0
BASIS_POINT: float = 1e-4
EPS: float = 1e-12
