"""/v1/ml routes — real training on real data, not stubbed."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from quantforge.api.auth import verify_api_key
from quantforge.api.schemas import MLTrainRequest, MLTrainResponse
from quantforge.data.loader import DataLoader
from quantforge.ml.features import build_feature_matrix, target_labels
from quantforge.ml.trainer import train_classifier


router = APIRouter(prefix="/v1/ml", tags=["ml"])


@router.post("/train", response_model=MLTrainResponse)
def train(req: MLTrainRequest, _key: str = Depends(verify_api_key)) -> MLTrainResponse:
    from sklearn.ensemble import GradientBoostingClassifier

    dl = DataLoader(cache_dir="data/cache")
    try:
        df = dl.yfinance(req.ticker, req.start, req.end)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"data fetch failed: {e}")
    if df.empty:
        raise HTTPException(status_code=404, detail="no data")

    feats = build_feature_matrix(df)
    target = target_labels(df["close"], horizon=1, kind="binary")
    data = feats.join(target.rename("y")).dropna()
    if len(data) < 300:
        raise HTTPException(status_code=422, detail="not enough samples")
    X = data.drop(columns="y")
    y = data["y"]

    hp_grid = [{"n_estimators": req.n_estimators, "max_depth": req.max_depth,
                "learning_rate": req.learning_rate, "random_state": 42}]
    _model, report = train_classifier(
        X, y, model_cls=GradientBoostingClassifier, hp_grid=hp_grid, verbose=False,
    )

    top_feats = dict(sorted(report.feature_importances.items(),
                             key=lambda kv: kv[1], reverse=True)[:15])
    return MLTrainResponse(
        train_acc=float(report.train_acc), val_acc=float(report.val_acc),
        test_acc=float(report.test_acc),
        train_auc=float(report.train_auc), val_auc=float(report.val_auc),
        test_auc=float(report.test_auc),
        baseline_acc=float(report.baseline_acc),
        top_features={k: float(v) for k, v in top_feats.items()},
        n_train=report.n_train, n_val=report.n_val, n_test=report.n_test,
    )
