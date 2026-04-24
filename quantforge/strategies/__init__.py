"""Strategy library."""
from quantforge.strategies.base import Strategy
from quantforge.strategies.buy_and_hold import BuyAndHoldStrategy
from quantforge.strategies.cross_sectional_momentum import CrossSectionalMomentum
from quantforge.strategies.dual_momentum import DualMomentum
from quantforge.strategies.factor_strategy import FactorStrategy
from quantforge.strategies.ma_crossover import MACrossoverStrategy
from quantforge.strategies.mean_reversion import BollingerMeanReversion, MeanReversionStrategy
from quantforge.strategies.ml_strategy import MLClassifierStrategy
from quantforge.strategies.momentum import MomentumStrategy
from quantforge.strategies.pairs_trading import PairsTradingStrategy
from quantforge.strategies.regime_switch import RegimeSwitch
from quantforge.strategies.rl_ppo import PPOStrategy
from quantforge.strategies.rsi_reversal import RSIReversalStrategy
from quantforge.strategies.trend_breakout import DonchianBreakout
from quantforge.strategies.vol_target import VolTarget

__all__ = [
    "Strategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "BollingerMeanReversion",
    "PairsTradingStrategy",
    "DonchianBreakout",
    "MACrossoverStrategy",
    "RSIReversalStrategy",
    "MLClassifierStrategy",
    "FactorStrategy",
    "BuyAndHoldStrategy",
    "CrossSectionalMomentum",
    "DualMomentum",
    "VolTarget",
    "RegimeSwitch",
    "PPOStrategy",
]
