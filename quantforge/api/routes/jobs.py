"""/v1/jobs routes — async backtest + ML training."""
from __future__ import annotations

import math
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from quantforge.api.auth import verify_api_key
from quantforge.api.jobs import JobStatus, get_manager
from quantforge.api.routes.backtest import _STRATEGY_FACTORY, _sample_equity, _sanitize
from quantforge.api.schemas import BacktestRequest, MLTrainRequest
from quantforge.analytics.performance import summary_stats
from quantforge.backtest import BacktestEngine
from quantforge.data.loader import DataLoader


router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


def _run_backtest_job(job, params: dict):
    req = BacktestRequest(**params)
    try:
        factory = _STRATEGY_FACTORY[req.strategy]
    except KeyError:
        raise ValueError(f"unknown strategy: {req.strategy}")
    strat = factory(req.params)
    if job._cancel.is_set():
        return None
    job.progress = 0.1
    dl = DataLoader(cache_dir="data/cache")
    data = dl.yfinance_many(req.tickers, req.start, req.end)
    if job._cancel.is_set():
        return None
    job.progress = 0.4
    engine = BacktestEngine(
        strategy=strat, data=data, initial_capital=req.capital,
        sizing_fraction=req.sizing_fraction, rebalance=req.rebalance,
    )
    res = engine.run()
    job.progress = 0.9
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


def _run_ml_train_job(job, params: dict):
    from sklearn.ensemble import GradientBoostingClassifier
    from quantforge.ml.features import build_feature_matrix, target_labels
    from quantforge.ml.trainer import train_classifier

    req = MLTrainRequest(**params)
    dl = DataLoader(cache_dir="data/cache")
    df = dl.yfinance(req.ticker, req.start, req.end)
    if df.empty:
        raise ValueError("no data")
    job.progress = 0.2
    feats = build_feature_matrix(df)
    target = target_labels(df["close"], horizon=1, kind="binary")
    data = feats.join(target.rename("y")).dropna()
    if len(data) < 300:
        raise ValueError("not enough samples")
    X = data.drop(columns="y")
    y = data["y"]
    if job._cancel.is_set():
        return None
    job.progress = 0.5
    hp_grid = [{"n_estimators": req.n_estimators, "max_depth": req.max_depth,
                "learning_rate": req.learning_rate, "random_state": 42}]
    _m, report = train_classifier(X, y, model_cls=GradientBoostingClassifier,
                                    hp_grid=hp_grid, verbose=False)
    job.progress = 0.9
    top = dict(sorted(report.feature_importances.items(),
                       key=lambda kv: kv[1], reverse=True)[:15])
    return {
        "train_acc": float(report.train_acc), "val_acc": float(report.val_acc),
        "test_acc": float(report.test_acc),
        "train_auc": float(report.train_auc), "val_auc": float(report.val_auc),
        "test_auc": float(report.test_auc),
        "baseline_acc": float(report.baseline_acc),
        "top_features": {k: float(v) for k, v in top.items()},
        "n_train": report.n_train, "n_val": report.n_val, "n_test": report.n_test,
    }


@router.post("/backtest", status_code=202)
def submit_backtest_job(req: BacktestRequest, owner: str = Depends(verify_api_key)):
    mgr = get_manager()
    try:
        job = mgr.submit(
            kind="backtest",
            func=lambda j: _run_backtest_job(j, req.model_dump()),
            params=req.model_dump(), owner=owner,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))
    return {"job_id": job.id, "status": job.status.value,
             "poll_url": f"/v1/jobs/{job.id}"}


@router.post("/ml-train", status_code=202)
def submit_ml_job(req: MLTrainRequest, owner: str = Depends(verify_api_key)):
    mgr = get_manager()
    try:
        job = mgr.submit(
            kind="ml_train",
            func=lambda j: _run_ml_train_job(j, req.model_dump()),
            params=req.model_dump(), owner=owner,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))
    return {"job_id": job.id, "status": job.status.value,
             "poll_url": f"/v1/jobs/{job.id}"}


@router.get("/{job_id}")
def get_job(job_id: str, owner: str = Depends(verify_api_key)):
    job = get_manager().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    # privacy: reject foreign jobs
    if job.owner and job.owner != owner and owner != "anonymous":
        raise HTTPException(status_code=403, detail="not your job")
    return job.to_dict()


@router.delete("/{job_id}")
def cancel_job(job_id: str, owner: str = Depends(verify_api_key)):
    ok = get_manager().cancel(job_id, owner=owner)
    if not ok:
        raise HTTPException(status_code=404, detail="job not found or not cancellable")
    return {"status": "cancelled"}


@router.get("")
def list_jobs(
    limit: int = Query(50, ge=1, le=500),
    owner: str = Depends(verify_api_key),
) -> List[dict]:
    jobs = get_manager().list_by_owner(owner, limit=limit)
    return [j.to_dict(include_result=False) for j in jobs]
