"""Advanced statistical methods: GARCH, cointegration, covariance shrinkage."""
from quantforge.stats.garch import garch11_fit, garch11_forecast, GARCHParams
from quantforge.stats.cointegration import (
    engle_granger, johansen_trace,
    rolling_cointegration, half_life_of_mean_reversion,
)
from quantforge.stats.shrinkage import (
    ledoit_wolf_shrinkage, constant_correlation_target,
    oracle_shrinkage, shrunk_covariance,
)
from quantforge.stats.regime import (
    markov_switching_returns, detect_structural_breaks,
)

__all__ = [
    "garch11_fit", "garch11_forecast", "GARCHParams",
    "engle_granger", "johansen_trace", "rolling_cointegration",
    "half_life_of_mean_reversion",
    "ledoit_wolf_shrinkage", "constant_correlation_target",
    "oracle_shrinkage", "shrunk_covariance",
    "markov_switching_returns", "detect_structural_breaks",
]
