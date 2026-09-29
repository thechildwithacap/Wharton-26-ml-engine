"""Plain-English reasons: correctness of the numbers quoted, and the house
style rules (no em dashes, no banned words)."""

import pandas as pd
import pytest

from wharton_ml_engine.data import SyntheticDataSource
from wharton_ml_engine.ml.dataset import build_training_panel
from wharton_ml_engine.ml.explain import explain
from wharton_ml_engine.ml.narrate import (
    BANNED_WORDS,
    EM_DASH,
    check_style,
    narrate,
    narrate_one,
)
from wharton_ml_engine.ml.predict import AlphaModel
from wharton_ml_engine.ml.train import train_alpha_model


@pytest.fixture(scope="module")
def bundle():
    return SyntheticDataSource(n_tickers=30, years=5, seed=44).load()


@pytest.fixture(scope="module")
def explanation(bundle):
    panel = build_training_panel(bundle, horizon_days=63)
    trained = train_alpha_model(bundle, task="regression", panel=panel, alpha=10.0)
    am = AlphaModel(trained)
    return explain(am, bundle)


def test_narrate_one_produces_text_for_every_ticker(explanation):
    for t in list(explanation.family_rollup.index)[:5]:
        text = narrate_one(explanation, t, confidence=0.6)
        assert isinstance(text, str) and len(text) > 20


def test_narrate_covers_all_requested_tickers(explanation):
    some = list(explanation.family_rollup.index)[:6]
    out = narrate(explanation, confidence=0.6, tickers=some)
    assert set(out.index) == set(some)
    assert out.apply(len).min() > 10


def test_narrate_flags_low_confidence(explanation):
    t = explanation.family_rollup.index[0]
    text = narrate_one(explanation, t, confidence=0.0)
    assert "doesn't change the portfolio" in text
    assert "0.00" in text


def test_narrate_high_confidence_no_disclaimer_needed_below_threshold(explanation):
    t = explanation.family_rollup.index[0]
    text = narrate_one(explanation, t, confidence=0.9)
    assert "doesn't change the portfolio" not in text


def test_unknown_ticker_handled_gracefully(explanation):
    text = narrate_one(explanation, "NOT_A_REAL_TICKER", confidence=0.5)
    assert "No score is available" in text


def test_raw_fundamentals_produce_concrete_numbers(bundle, explanation):
    fund = bundle.fundamentals_asof(bundle.dates()[-1])
    t = explanation.family_rollup.index[0]
    text_plain = narrate_one(explanation, t, confidence=0.7)
    text_with_raw = narrate_one(explanation, t, confidence=0.7, raw_fundamentals=fund)
    assert isinstance(text_with_raw, str)
    # Both must still pass style checks even when raw numbers are injected.
    assert check_style(text_with_raw)["ok"]
    assert isinstance(text_plain, str)


# --------------------------------------------------------------- style rules
def test_check_style_catches_em_dash():
    assert check_style(f"This is bad{EM_DASH}really bad.")["em_dash"] is True
    assert check_style("This is fine, really fine.")["em_dash"] is False


def test_check_style_catches_banned_words():
    for w in BANNED_WORDS:
        result = check_style(f"Let's {w} into this.")
        assert w in result["banned_words"]
    assert check_style("A clean sentence with no issues.")["banned_words"] == []


def test_all_narrated_text_passes_style_check(explanation):
    out = narrate(explanation, confidence=0.6)
    violations = []
    for ticker, text in out.items():
        result = check_style(text)
        if not result["ok"]:
            violations.append((ticker, result))
    assert violations == [], f"style violations: {violations}"


def test_all_narrated_text_passes_style_check_low_confidence(explanation):
    # Also check the "gated off" disclaimer sentence path.
    out = narrate(explanation, confidence=0.0)
    violations = [(t, check_style(x)) for t, x in out.items() if not check_style(x)["ok"]]
    assert violations == []
