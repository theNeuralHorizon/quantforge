"""ML utilities for quant: features, regime detection, simple forecasters."""
from quantforge.ml.features import (
    build_feature_matrix,
    cross_sectional_rank,
    price_features,
    target_labels,
    volatility_features,
    volume_features,
)
from quantforge.ml.forecast import (
    ar_forecast,
    ewma_forecast,
    linear_forecast,
    make_sequences,
)
from quantforge.ml.regime import (
    bull_bear_regime,
    hmm_regimes,
    trend_regime,
    vol_regime,
)
from quantforge.ml.trainer import (
    TrainingReport,
    time_order_split,
    train_classifier,
    walk_forward_train,
)

__all__ = [
    "price_features", "volume_features", "volatility_features", "cross_sectional_rank",
    "build_feature_matrix", "target_labels",
    "hmm_regimes", "bull_bear_regime", "vol_regime", "trend_regime",
    "ar_forecast", "ewma_forecast", "linear_forecast", "make_sequences",
    "train_classifier", "walk_forward_train", "time_order_split", "TrainingReport",
]
