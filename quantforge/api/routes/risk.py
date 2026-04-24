"""/v1/risk routes."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from quantforge.api.auth import verify_api_key
from quantforge.api.schemas import VaRRequest, VaRResponse
from quantforge.risk.var import (
    cornish_fisher_var,
    historical_cvar,
    historical_var,
    monte_carlo_var,
    parametric_cvar,
    parametric_var,
)

router = APIRouter(prefix="/v1/risk", tags=["risk"])


@router.post("/var", response_model=VaRResponse)
def compute_var(req: VaRRequest, _key: str = Depends(verify_api_key)) -> VaRResponse:
    returns = np.asarray(req.returns, dtype=float)
    if np.isnan(returns).any():
        raise HTTPException(status_code=422, detail="returns contain NaN")
    c = req.confidence

    if req.method == "historical":
        var = float(historical_var(returns, c))
        cvar = float(historical_cvar(returns, c))
    elif req.method == "parametric":
        var = float(parametric_var(returns, c))
        cvar = float(parametric_cvar(returns, c))
    elif req.method == "cornish_fisher":
        var = float(cornish_fisher_var(returns, c))
        cvar = float(historical_cvar(returns, c))  # CF-CVaR less common; fallback
    else:  # monte_carlo
        mu, sigma = float(returns.mean()), float(returns.std(ddof=1))
        var, cvar = monte_carlo_var(mu, sigma, c, seed=42)

    return VaRResponse(var=var, cvar=cvar, method=req.method, n_observations=len(returns))
