"""/v1/backtest route."""
from __future__ import annotations

import math

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from quantforge.analytics.performance import summary_stats
from quantforge.api.auth import verify_api_key
from quantforge.api.cache import cache_backend, make_key
from quantforge.api.schemas import BacktestRequest, BacktestResponse
from quantforge.backtest import BacktestEngine
from quantforge.data.loader import DataLoader
from quantforge.strategies import (
    BollingerMeanReversion,
    BuyAndHoldStrategy,
    CrossSectionalMomentum,
    DonchianBreakout,
    DualMomentum,
    FactorStrategy,
    MACrossoverStrategy,
    MomentumStrategy,
    RegimeSwitch,
    RSIReversalStrategy,
    VolTarget,
)

router = APIRouter(prefix="/v1/backtest", tags=["backtest"])


_STRATEGY_FACTORY = {
    "momentum":             lambda p: MomentumStrategy(
                                 lookback=int(p.get("lookback", 120)),
                                 allow_short=bool(p.get("allow_short", False))),
    "ma_crossover":         lambda p: MACrossoverStrategy(
                                 fast=int(p.get("fast", 20)), slow=int(p.get("slow", 100))),
    "bollinger_mr":         lambda p: BollingerMeanReversion(
                                 window=int(p.get("window", 20)), k=float(p.get("k", 2.0))),
    "donchian":             lambda p: DonchianBreakout(
                                 entry_window=int(p.get("entry_window", 20)),
                                 exit_window=int(p.get("exit_window", 10)),
                                 allow_short=bool(p.get("allow_short", False))),
    "rsi_reversal":         lambda p: RSIReversalStrategy(
                                 oversold=float(p.get("oversold", 30)),
                                 overbought=float(p.get("overbought", 70))),
    "factor":               lambda p: FactorStrategy(
                                 lookback_mom=int(p.get("lookback_mom", 120)),
                                 lookback_vol=int(p.get("lookback_vol", 60))),
    "cross_sec_momentum":   lambda p: CrossSectionalMomentum(
                                 lookback=int(p.get("lookback", 252)),
                                 skip=int(p.get("skip", 21)),
                                 top_q=float(p.get("top_q", 0.4))),
    "dual_momentum":        lambda p: DualMomentum(
                                 lookback=int(p.get("lookback", 252)),
                                 cash_asset=str(p.get("cash_asset", "TLT"))),
    "regime_switch":        lambda p: RegimeSwitch(
                                 risk_on_asset=str(p.get("risk_on_asset", "SPY")),
                                 risk_off_asset=str(p.get("risk_off_asset", "TLT")),
                                 signal_asset=str(p.get("signal_asset", "SPY"))),
    "vol_target_momentum":  lambda p: VolTarget(
                                 base=MomentumStrategy(lookback=int(p.get("lookback", 120))),
                                 target_vol=float(p.get("target_vol", 0.10))),
    "buy_and_hold":         lambda p: BuyAndHoldStrategy(),
}


def _sample_equity(curve: pd.Series, max_points: int = 180) -> dict[str, float]:
    if len(curve) <= max_points:
        step = 1
    else:
        step = max(1, len(curve) // max_points)
    sampled = curve.iloc[::step]
    return {ts.strftime("%Y-%m-%d"): float(v) for ts, v in sampled.items()}


def _sanitize(obj):
    """Replace NaN / Inf with None so Pydantic + JSON accept the payload."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


@router.post("", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest, _key: str = Depends(verify_api_key)) -> BacktestResponse:
    cache = cache_backend()
    ck = make_key("backtest", strategy=req.strategy, params=req.params, tickers=req.tickers,
                  start=req.start, end=req.end, capital=req.capital,
                  sizing=req.sizing_fraction, rebalance=req.rebalance)
    hit = cache.get(ck)
    if hit:
        import json
        return BacktestResponse(**json.loads(hit))

    try:
        factory = _STRATEGY_FACTORY[req.strategy]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"unknown strategy: {req.strategy}") from None

    try:
        strat = factory(req.params)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f"bad strategy params: {e}") from e

    dl = DataLoader(cache_dir="data/cache")
    try:
        data = dl.yfinance_many(req.tickers, req.start, req.end)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"data fetch failed: {e}") from e

    if any(df.empty for df in data.values()):
        raise HTTPException(status_code=404, detail="no data for one or more tickers")

    engine = BacktestEngine(strategy=strat, data=data, initial_capital=req.capital,
                             sizing_fraction=req.sizing_fraction, rebalance=req.rebalance)
    res = engine.run()

    stats = summary_stats(res.equity_curve, trades=res.trades)
    stats = {k: (None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v))
             for k, v in stats.items()}

    payload = BacktestResponse(
        strategy=req.strategy,
        n_bars=len(res.equity_curve),
        n_trades=int(len(res.trades)),
        final_equity=float(res.equity_curve.iloc[-1]) if len(res.equity_curve) else req.capital,
        summary_stats=_sanitize(stats),
        equity_curve_sample=_sample_equity(res.equity_curve),
    )

    import json
    cache.set(ck, json.dumps(payload.model_dump(), default=str).encode(), ttl_seconds=600)
    return payload
