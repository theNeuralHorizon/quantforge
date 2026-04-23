"""Risk analytics: VaR, CVaR, stress tests, drawdowns."""
from quantforge.risk.var import (
    historical_var, historical_cvar, parametric_var, parametric_cvar,
    monte_carlo_var, cornish_fisher_var,
)
from quantforge.risk.drawdown import (
    drawdown_series, max_drawdown, drawdown_table, underwater_duration,
)
from quantforge.risk.stress_test import (
    stress_scenarios, shock_portfolio, factor_shock,
)
from quantforge.risk.metrics import (
    sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio,
    tail_ratio, ulcer_index, gain_to_pain, information_ratio,
)
from quantforge.risk.simulation import (
    simulate_portfolio, simulate_portfolio_returns, SimulationResult,
)
from quantforge.risk.kelly import (
    kelly_fraction, kelly_continuous, kelly_vector, fractional_kelly,
    kelly_vector_capped,
)
from quantforge.risk.attribution import (
    volatility_attribution, var_attribution, cvar_attribution,
    risk_budget_deviation, RiskAttributionReport,
)

__all__ = [
    "historical_var", "historical_cvar", "parametric_var", "parametric_cvar",
    "monte_carlo_var", "cornish_fisher_var",
    "drawdown_series", "max_drawdown", "drawdown_table", "underwater_duration",
    "stress_scenarios", "shock_portfolio", "factor_shock",
    "sharpe_ratio", "sortino_ratio", "calmar_ratio", "omega_ratio",
    "tail_ratio", "ulcer_index", "gain_to_pain", "information_ratio",
    "simulate_portfolio", "simulate_portfolio_returns", "SimulationResult",
    "kelly_fraction", "kelly_continuous", "kelly_vector", "fractional_kelly",
    "kelly_vector_capped",
    "volatility_attribution", "var_attribution", "cvar_attribution",
    "risk_budget_deviation", "RiskAttributionReport",
]
