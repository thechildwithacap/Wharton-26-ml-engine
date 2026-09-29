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

from .card import ModelCard, build_model_card, write_model_card
from .dataset import (
    EXTENDED_FEATURES,
    ML_FEATURES,
    RAW_FUNDAMENTAL_FEATURES,
    build_training_panel,
    feature_columns,
    feature_frame,
)
from .explain import Explanation, coefficient_stability, explain, global_explain, permutation_importance, single_feature_ic_table
from .features import REGISTRY, FeatureSpec, family_of, proxy_features
from .features import spec as feature_spec
from .metrics import ic_summary, per_date_ic, rank_ic
from .models import LogisticRegressor, RidgeRegressor, StandardScaler
from .narrate import narrate, narrate_one
from .predict import AlphaModel
from .train import TrainedModel, train_alpha_model, walk_forward_evaluate
from .validation import (
    adjusted_significance,
    final_check,
    ic_breakdown_by_group,
    ic_breakdown_by_regime,
    ic_breakdown_by_year,
    ic_summary_nw,
    log_experiment,
    newey_west_t,
)
from .view import MLView, load_ml_view

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
    # feature registry (W1)
    "REGISTRY",
    "FeatureSpec",
    "feature_spec",
    "family_of",
    "proxy_features",
    # validation protocol (W3)
    "newey_west_t",
    "ic_summary_nw",
    "adjusted_significance",
    "log_experiment",
    "final_check",
    "ic_breakdown_by_year",
    "ic_breakdown_by_group",
    "ic_breakdown_by_regime",
    # interpretability (W5)
    "Explanation",
    "explain",
    "coefficient_stability",
    "permutation_importance",
    "single_feature_ic_table",
    "global_explain",
    "narrate",
    "narrate_one",
    "ModelCard",
    "build_model_card",
    "write_model_card",
    # integration (W6, partial)
    "MLView",
    "load_ml_view",
]
