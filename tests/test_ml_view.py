"""MLView: one object bundling score, confidence, explanation and reasons —
the interface any downstream consumer (engine, API/UI) should read from."""

import pandas as pd
import pytest

from wharton_ml_engine.data import SyntheticDataSource
from wharton_ml_engine.ml.dataset import build_training_panel
from wharton_ml_engine.ml.train import train_alpha_model
from wharton_ml_engine.ml.validation import log_experiment
from wharton_ml_engine.ml.view import load_ml_view


@pytest.fixture(scope="module")
def bundle():
    return SyntheticDataSource(n_tickers=25, years=5, seed=61).load()


@pytest.fixture(scope="module")
def model_path(bundle, tmp_path_factory):
    panel = build_training_panel(bundle, horizon_days=63)
    trained = train_alpha_model(bundle, task="regression", panel=panel, alpha=10.0,
                                horizon_days=63)
    path = str(tmp_path_factory.mktemp("models") / "test_alpha.json")
    trained.save(path)
    return path


def test_load_ml_view_produces_consistent_pieces(bundle, model_path):
    view = load_ml_view(model_path, bundle)
    assert view.as_of == bundle.dates()[-1]
    assert 0.0 <= view.confidence <= 1.0
    assert set(view.score.index) == set(view.raw_score.index)
    assert set(view.reasons.index) == set(view.explanation.family_rollup.index)
    assert view.card.confidence == view.confidence


def test_gated_off_view_scores_are_all_neutral(bundle, model_path):
    # Force zero confidence by faking away the model's metrics.
    from wharton_ml_engine.ml.predict import AlphaModel
    am = AlphaModel.load(model_path)
    am.trained.metrics = {"mean_ic": -0.01, "nw_t": 0.1}
    am.trained.save(model_path)
    view = load_ml_view(model_path, bundle)
    assert view.confidence == 0.0
    assert (view.score == 50.0).all()
    assert view.card.confidence == 0.0


def test_to_dict_is_json_safe(bundle, model_path):
    import json
    view = load_ml_view(model_path, bundle)
    d = view.to_dict()
    json.dumps(d, default=str)          # must not raise
    assert "confidence" in d and "reasons" in d and "card" in d


def test_registry_family_applies_multiple_testing_correction(bundle, model_path, tmp_path):
    reg = str(tmp_path / "registry.jsonl")
    for i in range(20):
        log_experiment({"variant": i}, {"mean_ic": 0.01}, "some_family", reg)

    import wharton_ml_engine.ml.view as view_mod
    orig = view_mod.count_trials
    view_mod.count_trials = lambda family, registry_path=reg: orig(family, reg)
    try:
        no_family = load_ml_view(model_path, bundle)
        with_family = load_ml_view(model_path, bundle, registry_family="some_family")
        # More trials in the family -> confidence should not increase.
        assert with_family.confidence <= no_family.confidence
    finally:
        view_mod.count_trials = orig
