"""/v1/backtest/walk-forward + /v1/backtest/compare — advanced backtest endpoints."""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from quantforge.analytics.performance import summary_stats
from quantforge.api.auth import verify_api_key
from quantforge.api.routes.backtest import _STRATEGY_FACTORY, _sample_equity, _sanitize
from quantforge.api.schemas import BacktestRequest
from quantforge.backtest import BacktestEngine
from quantforge.data.loader import DataLoader

router = APIRouter(prefix="/v1/backtest", tags=["backtest"])


class WalkForwardRequest(BacktestRequest):
    n_folds: int = Field(5, ge=2, le=20)


class CompareRequest(BaseModel):
    strategies: list[dict[str, Any]] = Field(..., min_length=2, max_length=12,
                                               description="list of BacktestRequest-shaped dicts")


def _run_single(req_dict: dict) -> dict:
    req = BacktestRequest(**req_dict)
    factory = _STRATEGY_FACTORY.get(req.strategy)
    if factory is None:
        raise ValueError(f"unknown strategy: {req.strategy}")
    strat = factory(req.params)
    dl = DataLoader(cache_dir="data/cache")
    data = dl.yfinance_many(req.tickers, req.start, req.end)
    if any(df.empty for df in data.values()):
        raise ValueError("no data for one or more tickers")
    engine = BacktestEngine(
        strategy=strat, data=data, initial_capital=req.capital,
        sizing_fraction=req.sizing_fraction, rebalance=req.rebalance,
    )
    res = engine.run()
    stats = summary_stats(res.equity_curve, trades=res.trades)
    stats = {k: (None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v))
             for k, v in stats.items()}
    return {
        "strategy": req.strategy,
        "n_bars": len(res.equity_curve),
        "n_trades": int(len(res.trades)),
        "final_equity": float(res.equity_curve.iloc[-1]) if len(res.equity_curve) else req.capital,
        "summary_stats": _sanitize(stats),
        "equity_curve_sample": _sample_equity(res.equity_curve),
    }


@router.post("/walk-forward")
def walk_forward(req: WalkForwardRequest, _owner: str = Depends(verify_api_key)) -> dict:
    """Split the date range into N non-overlapping folds, run the same strategy
    on each. Returns per-fold stats and overall average Sharpe/return.
    """
    try:
        start_dt = pd.to_datetime(req.start)
        end_dt = pd.to_datetime(req.end)
    except Exception:
        raise HTTPException(status_code=422, detail="bad start/end") from None
    if end_dt <= start_dt:
        raise HTTPException(status_code=422, detail="end must be > start")

    # Build fold boundaries by calendar-equal slices
    total_days = (end_dt - start_dt).days
    fold_days = total_days // req.n_folds
    if fold_days < 30:
        raise HTTPException(status_code=422, detail="fold too short; reduce n_folds or widen date range")

    fold_requests: list[dict] = []
    for i in range(req.n_folds):
        s = start_dt + pd.Timedelta(days=i * fold_days)
        e = start_dt + pd.Timedelta(days=(i + 1) * fold_days) if i < req.n_folds - 1 else end_dt
        d = req.model_dump()
        d["start"] = s.strftime("%Y-%m-%d")
        d["end"] = e.strftime("%Y-%m-%d")
        fold_requests.append(d)

    folds_out: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(4, req.n_folds)) as pool:
        futures = [pool.submit(_run_single, fr) for fr in fold_requests]
        for i, f in enumerate(futures):
            try:
                r = f.result()
                r["fold"] = i + 1
                r["start"] = fold_requests[i]["start"]
                r["end"] = fold_requests[i]["end"]
                folds_out.append(r)
            except Exception as e:
                folds_out.append({"fold": i + 1, "error": str(e),
                                   "start": fold_requests[i]["start"],
                                   "end": fold_requests[i]["end"]})

    # Aggregate
    ok = [f for f in folds_out if "error" not in f]
    sharpes = [f["summary_stats"].get("sharpe", 0) or 0 for f in ok]
    returns = [f["summary_stats"].get("total_return", 0) or 0 for f in ok]
    return {
        "strategy": req.strategy,
        "n_folds": req.n_folds,
        "folds": folds_out,
        "mean_sharpe": float(np.mean(sharpes)) if sharpes else None,
        "std_sharpe": float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else None,
        "mean_return": float(np.mean(returns)) if returns else None,
        "hit_rate": float(sum(1 for r in returns if r > 0) / len(returns)) if returns else None,
    }


@router.post("/compare")
def compare(req: CompareRequest, _owner: str = Depends(verify_api_key)) -> dict:
    """Run N strategies in parallel and return ranked results by Sharpe."""
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(4, len(req.strategies))) as pool:
        futures = [pool.submit(_run_single, s) for s in req.strategies]
        for i, f in enumerate(futures):
            try:
                r = f.result()
                r["_index"] = i
                results.append(r)
            except Exception as e:
                results.append({"_index": i, "strategy": req.strategies[i].get("strategy"),
                                 "error": str(e)})

    ok = [r for r in results if "error" not in r]
    ok.sort(key=lambda r: r["summary_stats"].get("sharpe", -1e9) or -1e9, reverse=True)
    errors = [r for r in results if "error" in r]
    return {
        "n_submitted": len(req.strategies),
        "ranked": ok,
        "failed": errors,
    }
