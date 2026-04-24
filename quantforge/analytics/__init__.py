"""Performance analytics, tearsheets, attribution."""
from quantforge.analytics.attribution import brinson_attribution, factor_attribution
from quantforge.analytics.benchmark import (
    BenchmarkReport,
    alpha_beta,
    benchmark_report,
    information_ratio,
    tracking_error,
    up_down_capture,
)
from quantforge.analytics.performance import (
    annualized_return,
    annualized_vol,
    avg_win_loss,
    cumulative_returns,
    profit_factor,
    rolling_sharpe,
    summary_stats,
    win_rate,
)
from quantforge.analytics.tearsheet import Tearsheet, tearsheet_markdown, tearsheet_text

__all__ = [
    "annualized_return", "annualized_vol", "cumulative_returns",
    "rolling_sharpe", "win_rate", "profit_factor", "avg_win_loss",
    "summary_stats", "Tearsheet", "tearsheet_text", "tearsheet_markdown",
    "brinson_attribution", "factor_attribution",
    "alpha_beta", "tracking_error", "information_ratio", "up_down_capture",
    "benchmark_report", "BenchmarkReport",
]
