"""Tests for the ML-driven vs rule-only portfolio backtest (and leak control)."""

import numpy as np
import pandas as pd
import pytest

from wharton_ml_engine import SyntheticDataSource, sample_profile
from wharton_ml_engine.engine import backtest_ml_vs_rules, format_comparison
from wharton_ml_engine.engine.backtest_portfolio import _train_bundle
from wharton_ml_engine.ml import AlphaModel, train_alpha_model


@pytest.fixture(scope="module")
def bundle():
    return SyntheticDataSource(n_tickers=40, years=8, seed=5).load()


@pytest.fixture(scope="module")
def train_end(bundle):
    return bundle.dates()[int(len(bundle.dates()) * 0.6)]


@pytest.fixture(scope="module")
def result(bundle, train_end):
    return backtest_ml_vs_rules(bundle, sample_profile(), train_end=train_end)


# --- bundle truncation / leak control ------------------------------------

def test_before_truncates_to_asof(bundle):
    mid = bundle.dates()[len(bundle.dates()) // 2]
    sub = bundle.before(mid)
    assert sub.prices.index.max() <= mid
    assert sub.benchmarks.index.max() <= mid
    assert sub.fundamentals.index.get_level_values(0).max() <= mid
    assert set(sub.meta) == set(bundle.meta)


def test_training_is_leak_free(bundle, train_end):
    trained = train_alpha_model(bundle.before(train_end), task="regression")
    # No training label may come from on/after the test boundary.
    last_train_date = pd.Timestamp(trained.meta["date_range"][1])
    assert last_train_date <= train_end


# --- comparison structure -------------------------------------------------

def test_comparison_has_both_strategies_and_benchmark(result):
    assert set(result.strategies) == {"rule_only", "ml_driven"}
    assert set(result.summary.index) == {"rule_only", "ml_driven", "benchmark"}
    for col in ["ann_return", "ann_vol", "sharpe", "max_drawdown"]:
        assert col in result.summary.columns


def test_returns_and_metrics_are_finite(result):
    for name in ["rule_only", "ml_driven"]:
        r = result.strategies[name].returns
        assert not r.empty
        assert np.isfinite(result.strategies[name].metrics["sharpe"])
        assert np.isfinite(result.strategies[name].metrics["ann_return"])


def test_backtest_dates_are_out_of_sample(result, train_end):
    assert result.test_start >= train_end
    for name in result.strategies:
        for d in result.strategies[name].weights_history:
            assert d >= train_end


def test_equity_curves_shape(result):
    curves = result.equity_curves()
    assert set(curves.columns) == {"rule_only", "ml_driven", "benchmark"}
    assert len(curves) > 50


def test_format_runs(result):
    text = format_comparison(result)
    assert "ML-DRIVEN vs RULE-ONLY" in text
    assert "benchmark" in text


def test_accepts_pretrained_model(bundle, train_end):
    trained = train_alpha_model(bundle.before(train_end), task="regression")
    res = backtest_ml_vs_rules(bundle, sample_profile(), train_end=train_end,
                               alpha_model=AlphaModel(trained))
    assert not res.strategies["ml_driven"].returns.empty
    assert res.retrain_dates == []               # fixed model -> no retraining


# --- walk-forward retraining ---------------------------------------------

def test_train_bundle_expanding_and_rolling(bundle):
    d = bundle.dates()[int(len(bundle.dates()) * 0.7)]
    exp = _train_bundle(bundle, d, None)
    assert exp.prices.index.max() <= d                       # leak-free
    assert exp.prices.index.min() == bundle.prices.index.min()  # expanding
    roll = _train_bundle(bundle, d, lookback_days=365 * 2)
    assert roll.prices.index.max() <= d
    assert roll.prices.index.min() > exp.prices.index.min()  # rolling is shorter


def test_walk_forward_retrains_leak_free(bundle):
    # Shorter test window + coarse cadence keeps this fast.
    te = bundle.dates()[int(len(bundle.dates()) * 0.7)]
    res = backtest_ml_vs_rules(bundle, sample_profile(), train_end=te,
                               retrain_every=12)
    assert len(res.retrain_dates) >= 2
    assert res.ml_metrics.get("n_retrains") == len(res.retrain_dates)
    # Every refit happens inside the test window (never on future data).
    assert all(d >= res.test_start for d in res.retrain_dates)
    assert res.retrain_dates == sorted(res.retrain_dates)
    assert not res.strategies["ml_driven"].returns.empty
    assert "WALK-FORWARD" in format_comparison(res)
