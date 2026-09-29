"""Model cards: built from a real trained model's own metrics/metadata, never
hand-typed, and round-trip to JSON + Markdown."""

import json
import os

import pytest

from wharton_ml_engine.data import SyntheticDataSource
from wharton_ml_engine.ml.card import build_model_card, write_model_card
from wharton_ml_engine.ml.dataset import ML_FEATURES, build_training_panel
from wharton_ml_engine.ml.predict import AlphaModel
from wharton_ml_engine.ml.train import train_alpha_model
from wharton_ml_engine.ml.validation import final_check, log_experiment


@pytest.fixture(scope="module")
def bundle():
    return SyntheticDataSource(n_tickers=30, years=6, seed=51).load()


@pytest.fixture(scope="module")
def alpha_model(bundle):
    panel = build_training_panel(bundle, horizon_days=63)
    trained = train_alpha_model(bundle, task="regression", panel=panel, alpha=10.0,
                                horizon_days=63)
    return AlphaModel(trained)


def test_card_reflects_the_actual_model(alpha_model):
    card = build_model_card(alpha_model, model_id="test_model")
    assert card.horizon_days == 63
    assert card.task == "regression"
    assert set(card.feature_names) == set(ML_FEATURES)
    assert card.model_type == "ridge"
    assert "analyst" in card.proxy_features


def test_card_confidence_matches_alpha_model(alpha_model):
    card = build_model_card(alpha_model, model_id="test_model", n_trials=3)
    assert card.confidence == alpha_model.confidence(n_trials=3)


def test_card_limitations_flag_proxies_and_gate_state(alpha_model):
    card = build_model_card(alpha_model, model_id="test_model")
    joined = " ".join(card.known_limitations)
    assert "analyst" in joined
    if card.confidence <= 0.05:
        assert "gate" in joined.lower() or "GATED" in joined or "constant" in joined


def test_holdout_unchecked_by_default(alpha_model):
    card = build_model_card(alpha_model, model_id="test_model")
    assert card.holdout["checked"] is False
    assert "Not yet checked" in card.to_markdown()


def test_holdout_result_reflected_in_card(bundle, tmp_path):
    panel = build_training_panel(bundle, horizon_days=63)
    from wharton_ml_engine.ml.models import RidgeRegressor, StandardScaler
    from wharton_ml_engine.ml.dataset import ML_FEATURES

    def fit_fn(research_panel):
        scaler = StandardScaler().fit(research_panel[ML_FEATURES].to_numpy())
        X = scaler.transform(research_panel[ML_FEATURES].to_numpy())
        m = RidgeRegressor(alpha=10.0).fit(X, research_panel["fwd_excess"].to_numpy())
        return (scaler, m)

    def predict_fn(model, holdout_panel):
        scaler, m = model
        X = scaler.transform(holdout_panel[ML_FEATURES].to_numpy())
        return m.predict(X)

    result = final_check({"model": "card_test"}, panel, fit_fn, predict_fn,
                         family="card_test_fam", holdout_fraction=0.2,
                         holdout_log_path=str(tmp_path / "holdout_log.jsonl"))
    trained = train_alpha_model(bundle, task="regression", panel=panel, alpha=10.0,
                                horizon_days=63)
    am = AlphaModel(trained)
    card = build_model_card(am, model_id="test_model", holdout_result=result)
    assert card.holdout["checked"] is True
    assert card.holdout["n_holdout_dates"] == result.n_holdout_dates


def test_write_model_card_produces_both_files(alpha_model, tmp_path):
    card = build_model_card(alpha_model, model_id="test_model")
    paths = write_model_card(card, str(tmp_path / "my_model"))
    assert os.path.exists(paths["json"])
    assert os.path.exists(paths["md"])
    with open(paths["json"]) as fh:
        data = json.load(fh)
    assert data["model_id"] == "test_model"
    with open(paths["md"]) as fh:
        md = fh.read()
    assert "# Model card: test_model" in md
    assert "## Known limitations" in md


def test_write_model_card_handles_json_suffix_in_path(alpha_model, tmp_path):
    card = build_model_card(alpha_model, model_id="test_model")
    paths = write_model_card(card, str(tmp_path / "my_model.json"))
    assert paths["json"].endswith("my_model.card.json")
    assert os.path.exists(paths["json"]) and os.path.exists(paths["md"])
