"""Pydantic request/response models — strict input validation for every endpoint."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Options
# =============================================================================
class OptionsPriceRequest(BaseModel):
    S: float = Field(..., gt=0, le=1_000_000, description="Spot price")
    K: float = Field(..., gt=0, le=1_000_000, description="Strike price")
    T: float = Field(..., gt=0, le=50, description="Years to expiry")
    r: float = Field(0.04, ge=-0.5, le=1.0, description="Risk-free rate")
    sigma: float = Field(..., gt=0, le=5.0, description="Volatility")
    option: Literal["call", "put"] = "call"
    q: float = Field(0.0, ge=0, le=0.5, description="Dividend yield")
    steps: int = Field(500, ge=10, le=5000, description="CRR steps")
    n_paths: int = Field(50_000, ge=1_000, le=500_000, description="MC paths")
    seed: Optional[int] = Field(42, ge=0, le=2**31 - 1)


class OptionsPriceResponse(BaseModel):
    black_scholes: float
    crr_european: float
    crr_american: float
    monte_carlo: float
    monte_carlo_stderr: float
    greeks: Dict[str, float]


class ImpliedVolRequest(BaseModel):
    price: float = Field(..., gt=0, le=1_000_000)
    S: float = Field(..., gt=0, le=1_000_000)
    K: float = Field(..., gt=0, le=1_000_000)
    T: float = Field(..., gt=0, le=50)
    r: float = Field(0.04, ge=-0.5, le=1.0)
    option: Literal["call", "put"] = "call"
    q: float = Field(0.0, ge=0, le=0.5)


class ImpliedVolResponse(BaseModel):
    implied_vol: float


# =============================================================================
# Backtest
# =============================================================================
class BacktestRequest(BaseModel):
    strategy: Literal[
        "momentum", "ma_crossover", "bollinger_mr", "donchian",
        "rsi_reversal", "factor", "cross_sec_momentum", "dual_momentum",
        "regime_switch", "vol_target_momentum", "buy_and_hold",
    ] = Field(..., description="Strategy name")
    params: Dict[str, Any] = Field(default_factory=dict)
    tickers: List[str] = Field(default_factory=lambda: ["SPY", "QQQ", "IWM", "TLT", "GLD"],
                                 min_length=1, max_length=20)
    start: str = Field("2015-01-01")
    end: str = Field("2025-01-01")
    capital: float = Field(100_000, gt=0, le=1_000_000_000)
    sizing_fraction: float = Field(0.25, gt=0, le=1.0)
    rebalance: Literal["bar", "weekly", "monthly"] = "monthly"

    @field_validator("tickers")
    @classmethod
    def _uppercase_tickers(cls, v: List[str]) -> List[str]:
        cleaned = []
        for t in v:
            t = t.strip().upper()
            if not t.replace("-", "").replace(".", "").isalnum() or len(t) > 10:
                raise ValueError(f"invalid ticker: {t}")
            cleaned.append(t)
        return cleaned


class BacktestResponse(BaseModel):
    strategy: str
    n_bars: int
    n_trades: int
    final_equity: float
    summary_stats: Dict[str, float]
    equity_curve_sample: Dict[str, float]  # date -> equity (sampled for bandwidth)


# =============================================================================
# Portfolio
# =============================================================================
class PortfolioOptimizeRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=2, max_length=20)
    start: str = Field("2015-01-01")
    end: str = Field("2025-01-01")
    objective: Literal["max_sharpe", "min_variance", "erc", "hrp"] = "max_sharpe"
    risk_free: float = Field(0.02, ge=-0.2, le=0.5)

    @field_validator("tickers")
    @classmethod
    def _up(cls, v: List[str]) -> List[str]:
        out = [t.strip().upper() for t in v]
        for t in out:
            if not t.replace("-", "").replace(".", "").isalnum() or len(t) > 10:
                raise ValueError(f"invalid ticker: {t}")
        return out


class PortfolioOptimizeResponse(BaseModel):
    weights: Dict[str, float]
    expected_return: float
    expected_vol: float
    expected_sharpe: float


# =============================================================================
# Risk
# =============================================================================
class VaRRequest(BaseModel):
    returns: List[float] = Field(..., min_length=20, max_length=100_000)
    confidence: float = Field(0.95, gt=0.5, lt=1.0)
    method: Literal["historical", "parametric", "cornish_fisher", "monte_carlo"] = "historical"


class VaRResponse(BaseModel):
    var: float
    cvar: float
    method: str
    n_observations: int


# =============================================================================
# ML
# =============================================================================
class MLTrainRequest(BaseModel):
    ticker: str = Field("SPY")
    start: str = Field("2010-01-01")
    end: str = Field("2025-01-01")
    n_estimators: int = Field(100, ge=10, le=500)
    max_depth: int = Field(3, ge=2, le=10)
    learning_rate: float = Field(0.05, gt=0, le=0.5)

    @field_validator("ticker")
    @classmethod
    def _tk(cls, v: str) -> str:
        v = v.strip().upper()
        if not v.replace("-", "").replace(".", "").isalnum() or len(v) > 10:
            raise ValueError("invalid ticker")
        return v


class MLTrainResponse(BaseModel):
    train_acc: float
    val_acc: float
    test_acc: float
    train_auc: float
    val_auc: float
    test_auc: float
    baseline_acc: float
    top_features: Dict[str, float]
    n_train: int
    n_val: int
    n_test: int


# =============================================================================
# Market data
# =============================================================================
class MarketDataResponse(BaseModel):
    ticker: str
    n_bars: int
    start_date: str
    end_date: str
    last_close: float
    ytd_return: Optional[float] = None
    bars: List[Dict[str, Any]] = Field(default_factory=list)


# =============================================================================
# Meta
# =============================================================================
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unhealthy"]
    version: str
    uptime_s: float


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: Dict[str, bool]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
