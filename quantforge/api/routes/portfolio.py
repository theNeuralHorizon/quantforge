"""/v1/portfolio routes."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from quantforge.api.auth import verify_api_key
from quantforge.api.schemas import PortfolioOptimizeRequest, PortfolioOptimizeResponse
from quantforge.data.loader import DataLoader
from quantforge.portfolio.hrp import hierarchical_risk_parity
from quantforge.portfolio.markowitz import max_sharpe, min_variance
from quantforge.portfolio.risk_parity import equal_risk_contribution

router = APIRouter(prefix="/v1/portfolio", tags=["portfolio"])


@router.post("/optimize", response_model=PortfolioOptimizeResponse)
def optimize(req: PortfolioOptimizeRequest,
              _key: str = Depends(verify_api_key)) -> PortfolioOptimizeResponse:
    dl = DataLoader(cache_dir="data/cache")
    try:
        data = dl.yfinance_many(req.tickers, req.start, req.end)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"data fetch failed: {e}") from e

    rets = pd.DataFrame(
        {t: df["close"].pct_change() for t, df in data.items()}
    ).dropna()
    if len(rets) < 30:
        raise HTTPException(status_code=422, detail="not enough return history")

    mu = rets.mean().values * 252
    cov = rets.cov().values * 252
    if np.isnan(mu).any() or np.isnan(cov).any():
        raise HTTPException(status_code=422, detail="returns contain NaN")

    if req.objective == "max_sharpe":
        w = max_sharpe(mu, cov, risk_free=req.risk_free)
    elif req.objective == "min_variance":
        w = min_variance(cov)
    elif req.objective == "erc":
        w = equal_risk_contribution(cov)
    else:
        w = hierarchical_risk_parity(cov)

    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ cov @ w))
    sharpe = (port_ret - req.risk_free) / port_vol if port_vol > 1e-9 else 0.0

    return PortfolioOptimizeResponse(
        weights={t: float(wi) for t, wi in zip(req.tickers, w, strict=False)},
        expected_return=port_ret,
        expected_vol=port_vol,
        expected_sharpe=float(sharpe) if math.isfinite(sharpe) else 0.0,
    )
