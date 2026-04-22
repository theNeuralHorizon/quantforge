"""/v1/options routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from quantforge.api.auth import verify_api_key
from quantforge.api.schemas import (
    ImpliedVolRequest, ImpliedVolResponse,
    OptionsPriceRequest, OptionsPriceResponse,
)
from quantforge.options.binomial import crr_american, crr_price
from quantforge.options.black_scholes import bs_call, bs_implied_vol, bs_put
from quantforge.options.greeks import all_greeks
from quantforge.options.monte_carlo import mc_european


router = APIRouter(prefix="/v1/options", tags=["options"])


@router.post("/price", response_model=OptionsPriceResponse)
def price_option(req: OptionsPriceRequest, _key: str = Depends(verify_api_key)) -> OptionsPriceResponse:
    bs = bs_call(req.S, req.K, req.T, req.r, req.sigma, req.q) if req.option == "call" \
         else bs_put(req.S, req.K, req.T, req.r, req.sigma, req.q)
    crr_eu = crr_price(req.S, req.K, req.T, req.r, req.sigma, req.option, req.steps, req.q)
    crr_am = crr_american(req.S, req.K, req.T, req.r, req.sigma, req.option, req.steps, req.q)
    mc_p, mc_se = mc_european(req.S, req.K, req.T, req.r, req.sigma, req.option,
                                q=req.q, n_paths=req.n_paths, seed=req.seed)
    greeks = all_greeks(req.S, req.K, req.T, req.r, req.sigma, req.option, req.q)
    return OptionsPriceResponse(
        black_scholes=float(bs),
        crr_european=float(crr_eu),
        crr_american=float(crr_am),
        monte_carlo=float(mc_p),
        monte_carlo_stderr=float(mc_se),
        greeks={k: float(v) for k, v in greeks.items()},
    )


@router.post("/iv", response_model=ImpliedVolResponse)
def implied_vol(req: ImpliedVolRequest, _key: str = Depends(verify_api_key)) -> ImpliedVolResponse:
    iv = bs_implied_vol(req.price, req.S, req.K, req.T, req.r, req.option, req.q)
    if iv != iv:  # NaN — no solution
        raise HTTPException(status_code=422, detail="no implied vol solution (price below intrinsic?)")
    return ImpliedVolResponse(implied_vol=float(iv))
