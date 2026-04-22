"""ML utilities for quant: features, regime detection, simple forecasters."""
from quantforge.ml.features import (
    price_features, volume_features, volatility_features, cross_sectional_rank,
    build_feature_matrix, target_labels,
)
from quantforge.ml.regime import (
    hmm_regimes, bull_bear_regime, vol_regime, trend_regime,
)
from quantforge.ml.forecast import (
    ar_forecast, ewma_forecast, linear_forecast, make_sequences,
)
from quantforge.ml.trainer import (
    train_classifier, walk_forward_train, time_order_split, TrainingReport,
)

__all__ = [
    "price_features", "volume_features", "volatility_features", "cross_sectional_rank",
    "build_feature_matrix", "target_labels",
    "hmm_regimes", "bull_bear_regime", "vol_regime", "trend_regime",
    "ar_forecast", "ewma_forecast", "linear_forecast", "make_sequences",
    "train_classifier", "walk_forward_train", "time_order_split", "TrainingReport",
]
