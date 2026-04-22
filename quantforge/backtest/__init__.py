"""Event-driven backtest engine."""
from quantforge.backtest.slippage import SlippageModel, FixedBpsSlippage, VolumeImpactSlippage, NoSlippage
from quantforge.backtest.commission import CommissionModel, FixedBpsCommission, PerShareCommission, NoCommission
from quantforge.backtest.broker import SimulatedBroker
from quantforge.backtest.engine import BacktestEngine, BacktestResult
from quantforge.backtest.tca import TCAReport, analyze_trades

__all__ = [
    "SlippageModel",
    "FixedBpsSlippage",
    "VolumeImpactSlippage",
    "NoSlippage",
    "CommissionModel",
    "FixedBpsCommission",
    "PerShareCommission",
    "NoCommission",
    "SimulatedBroker",
    "BacktestEngine",
    "BacktestResult",
    "TCAReport",
    "analyze_trades",
]
