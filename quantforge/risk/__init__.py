"""Risk analytics: VaR, CVaR, stress tests, drawdowns."""
from quantforge.risk.attribution import (
    RiskAttributionReport,
    cvar_attribution,
    risk_budget_deviation,
    var_attribution,
    volatility_attribution,
)
from quantforge.risk.drawdown import (
    drawdown_series,
    drawdown_table,
    max_drawdown,
    underwater_duration,
)
from quantforge.risk.kelly import (
    fractional_kelly,
    kelly_continuous,
    kelly_fraction,
    kelly_vector,
    kelly_vector_capped,
)
from quantforge.risk.metrics import (
    calmar_ratio,
    gain_to_pain,
    information_ratio,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
    ulcer_index,
)
from quantforge.risk.simulation import (
    SimulationResult,
    simulate_portfolio,
    simulate_portfolio_returns,
)
from quantforge.risk.stress_test import (
    factor_shock,
    shock_portfolio,
    stress_scenarios,
)
from quantforge.risk.var import (
    cornish_fisher_var,
    historical_cvar,
    historical_var,
    monte_carlo_var,
    parametric_cvar,
    parametric_var,
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
