"""Machine-learning layer: train a model that learns to combine the engine's
signals into an expected-return (alpha) score.

The models are numpy-only and interpretable (linear/logistic), evaluated with
walk-forward, no-look-ahead cross-validation using the Information Coefficient.
A trained model plugs into the engine as an extra ``ml_alpha`` signal.

Quick start::

    from wharton_ml_engine import SyntheticDataSource
    from wharton_ml_engine.ml import train_alpha_model, AlphaModel

    bundle = SyntheticDataSource().load()
    trained = train_alpha_model(bundle, task="regression")
    print(trained.metrics)          # out-of-sample IC, hit rate, ...
    print(trained.coefficients())   # what the model learned to weight

    alpha = AlphaModel(trained)
    ml_alpha = alpha.score(bundle)  # 0-100 per ticker at the latest date
"""

from .dataset import (
    EXTENDED_FEATURES,
    ML_FEATURES,
    RAW_FUNDAMENTAL_FEATURES,
    build_training_panel,
    feature_columns,
    feature_frame,
)
from .metrics import ic_summary, per_date_ic, rank_ic
from .models import LogisticRegressor, RidgeRegressor, StandardScaler
from .predict import AlphaModel
from .train import TrainedModel, train_alpha_model, walk_forward_evaluate

__all__ = [
    "ML_FEATURES",
    "EXTENDED_FEATURES",
    "RAW_FUNDAMENTAL_FEATURES",
    "feature_columns",
    "feature_frame",
    "build_training_panel",
    "rank_ic",
    "per_date_ic",
    "ic_summary",
    "StandardScaler",
    "RidgeRegressor",
    "LogisticRegressor",
    "TrainedModel",
    "train_alpha_model",
    "walk_forward_evaluate",
    "AlphaModel",
]
