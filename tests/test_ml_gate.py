"""AlphaModel evidence gate: a model's blend influence tracks its actual
out-of-sample evidence, not just whether one was supplied."""

import numpy as np
import pandas as pd
import pytest

from wharton_ml_engine.data import SyntheticDataSource
from wharton_ml_engine.ml.predict import AlphaModel
from wharton_ml_engine.ml.train import TrainedModel
from wharton_ml_engine.ml.models import RidgeRegressor, StandardScaler


def _fake_trained(mean_ic, nw_t, feature_names=("value", "quality")):
    model = RidgeRegressor(alpha=1.0)
    model.coef_ = np.array([0.3, -0.1][: len(feature_names)])
    model.intercept_ = 0.0
    scaler = StandardScaler(mean_=np.zeros(len(feature_names)),
                            std_=np.ones(len(feature_names)))
    return TrainedModel(
        model=model, scaler=scaler, feature_names=list(feature_names),
        task="regression", target_col="fwd_excess", horizon_days=126,
        metrics={"mean_ic": mean_ic, "nw_t": nw_t, "n_periods": 40},
    )


def test_confidence_zero_for_negative_ic():
    am = AlphaModel(_fake_trained(mean_ic=-0.02, nw_t=3.0))
    assert am.confidence() == 0.0


def test_confidence_zero_for_missing_metrics():
    trained = _fake_trained(mean_ic=0.05, nw_t=2.0)
    trained.metrics = {}
    assert AlphaModel(trained).confidence() == 0.0


def test_confidence_zero_for_weak_t():
    am = AlphaModel(_fake_trained(mean_ic=0.02, nw_t=0.5))
    assert am.confidence() == 0.0


def test_confidence_positive_for_strong_t():
    am = AlphaModel(_fake_trained(mean_ic=0.05, nw_t=3.5))
    assert am.confidence() > 0.5


def test_confidence_monotonic_in_t():
    weak = AlphaModel(_fake_trained(mean_ic=0.03, nw_t=1.5)).confidence()
    strong = AlphaModel(_fake_trained(mean_ic=0.03, nw_t=3.0)).confidence()
    assert 0.0 <= weak < strong <= 1.0


def test_confidence_shrinks_with_more_trials():
    single = AlphaModel(_fake_trained(mean_ic=0.03, nw_t=2.5)).confidence(n_trials=1)
    many = AlphaModel(_fake_trained(mean_ic=0.03, nw_t=2.5)).confidence(n_trials=50)
    assert many <= single


def test_gated_score_collapses_toward_neutral_when_confidence_zero():
    bundle = SyntheticDataSource(n_tickers=30, years=4, seed=17).load()
    weak = AlphaModel(_fake_trained(mean_ic=0.01, nw_t=0.3))
    gated = weak.score(bundle, gate=True)
    # Zero confidence -> every name collapses to the neutral midpoint.
    assert (gated == 50.0).all()


def test_ungated_score_still_varies_across_names():
    bundle = SyntheticDataSource(n_tickers=30, years=4, seed=17).load()
    weak = AlphaModel(_fake_trained(mean_ic=0.01, nw_t=0.3))
    ungated = weak.score(bundle, gate=False)
    # explain.py needs the real, undamped prediction spread even for a weak model.
    assert ungated.nunique() > 1


def test_strong_model_gated_score_is_close_to_ungated():
    bundle = SyntheticDataSource(n_tickers=30, years=4, seed=17).load()
    strong = AlphaModel(_fake_trained(mean_ic=0.06, nw_t=5.0))
    gated = strong.score(bundle, gate=True)
    ungated = strong.score(bundle, gate=False)
    # High confidence -> gating barely moves the score away from the raw one.
    assert (gated - ungated).abs().max() < 15.0


def test_gate_closes_the_previously_fixed_blend_weight_gap():
    """Regression guard for the real gap this closes: previously ANY model
    passed to run_engine got a fixed, undiminished 22% blend weight in
    integrate_scores regardless of its evidence. A weak model must now leave
    the integrated ranking materially unchanged versus not having a model at
    all (because its gated ml_alpha collapses to a constant, contributing no
    differentiating information to the blend)."""
    from wharton_ml_engine import EngineConfig, run_engine, sample_profile

    bundle = SyntheticDataSource(n_tickers=40, years=5, seed=23).load()
    weak = AlphaModel(_fake_trained(mean_ic=0.005, nw_t=0.2,
                                    feature_names=("value", "quality")))
    profile = sample_profile()
    without = run_engine(bundle, profile, run_backtest=False, run_monte_carlo=False)
    with_weak = run_engine(bundle, profile, alpha_model=weak, run_backtest=False,
                           run_monte_carlo=False)
    # Rankings should be nearly identical — a gated-off model must not reshuffle
    # the book just because it was supplied.
    top_without = set(list(without.candidate_weights)[:10]) if without.candidate_weights else set()
    top_with = set(list(with_weak.candidate_weights)[:10]) if with_weak.candidate_weights else set()
    if top_without and top_with:
        overlap = len(top_without & top_with) / len(top_without)
        assert overlap >= 0.7
