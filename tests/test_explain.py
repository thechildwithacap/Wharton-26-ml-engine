"""Interpretability: local contributions sum to the raw score exactly; global
explanation views (coefficient stability, permutation importance, single-
feature IC) run and return sane structures."""

import numpy as np
import pandas as pd
import pytest

from wharton_ml_engine.data import SyntheticDataSource
from wharton_ml_engine.ml.dataset import ML_FEATURES, build_training_panel
from wharton_ml_engine.ml.explain import (
    coefficient_stability,
    explain,
    global_explain,
    permutation_importance,
    single_feature_ic_table,
)
from wharton_ml_engine.ml.predict import AlphaModel
from wharton_ml_engine.ml.train import train_alpha_model


@pytest.fixture(scope="module")
def bundle():
    return SyntheticDataSource(n_tickers=35, years=6, seed=31).load()


@pytest.fixture(scope="module")
def panel(bundle):
    return build_training_panel(bundle, horizon_days=63)


@pytest.fixture(scope="module")
def reg_model(bundle, panel):
    trained = train_alpha_model(bundle, task="regression", panel=panel, alpha=10.0)
    return AlphaModel(trained)


@pytest.fixture(scope="module")
def clf_model(bundle, panel):
    trained = train_alpha_model(bundle, task="classification", panel=panel)
    return AlphaModel(trained)


# --------------------------------------------------------------- local
def test_contributions_sum_to_raw_score_ridge(reg_model, bundle):
    exp = explain(reg_model, bundle)
    raw = reg_model.raw_score(bundle)
    total_from_rollup = exp.family_rollup["total"]
    common = total_from_rollup.index.intersection(raw.index)
    assert len(common) > 0
    np.testing.assert_allclose(total_from_rollup.loc[common].to_numpy(),
                               raw.loc[common].to_numpy(), atol=1e-6)


def test_contributions_sum_to_logit_classification(clf_model, bundle):
    exp = explain(clf_model, bundle)
    assert exp.is_classification is True
    # For classification, family_rollup's "total" is the logit; sigmoid(logit)
    # must equal the model's actual predict_proba output.
    trained = clf_model.trained
    ext = trained.feature_names
    from wharton_ml_engine.ml.dataset import feature_frame, is_extended
    feats = feature_frame(bundle, bundle.dates()[-1], extended=is_extended(ext))
    feats = feats.reindex(exp.family_rollup.index).dropna()
    proba = trained.raw_predict(feats)
    logit = exp.family_rollup["total"].reindex(feats.index)
    sigmoid = 1.0 / (1.0 + np.exp(-logit))
    np.testing.assert_allclose(sigmoid.to_numpy(), proba.reindex(feats.index).to_numpy(),
                               atol=1e-6)


def test_family_rollup_sums_match_per_feature_sums(reg_model, bundle):
    exp = explain(reg_model, bundle)
    per_ticker_feature_sum = exp.local.groupby("ticker")["contribution"].sum()
    family_cols = [c for c in exp.family_rollup.columns if c not in ("intercept", "total")]
    per_ticker_family_sum = exp.family_rollup[family_cols].sum(axis=1)
    common = per_ticker_feature_sum.index.intersection(per_ticker_family_sum.index)
    np.testing.assert_allclose(per_ticker_feature_sum.loc[common].to_numpy(),
                               per_ticker_family_sum.loc[common].to_numpy(), atol=1e-6)


def test_explain_restricts_to_requested_tickers(reg_model, bundle):
    some = bundle.tickers[:5]
    exp = explain(reg_model, bundle, tickers=some)
    assert set(exp.local["ticker"].unique()).issubset(set(some))


def test_top_contributors_helper(reg_model, bundle):
    exp = explain(reg_model, bundle)
    ticker = exp.family_rollup.index[0]
    top = exp.top_contributors(ticker, n=2, positive=True)
    assert len(top) <= 2
    assert (top["contribution"] > 0).all()


def test_proxy_features_flagged_in_meta(reg_model, bundle):
    exp = explain(reg_model, bundle)
    assert "analyst" in exp.meta["proxy_features"]


# --------------------------------------------------------------- global
def test_coefficient_stability_reports_every_feature(panel):
    out = coefficient_stability(panel, task="regression", feature_names=ML_FEATURES,
                                n_splits=4, min_train_periods=8, alpha=10.0)
    assert set(out.index) == set(ML_FEATURES)
    assert (out["sign_stability"].dropna() >= 0).all()
    assert (out["sign_stability"].dropna() <= 1).all()


def test_permutation_importance_by_family_runs(panel):
    out = permutation_importance(panel, task="regression", feature_names=ML_FEATURES,
                                 n_splits=3, min_train_periods=8, n_repeats=2,
                                 by_family=True, alpha=10.0)
    assert not out.empty
    assert "mean_ic_drop" in out.columns and "baseline_ic" in out.columns
    assert out["family"].nunique() >= 1


def test_single_feature_ic_table_flags_sign_disagreement():
    out = single_feature_ic_table(
        pd.DataFrame({"date": pd.to_datetime(["2020-01-01"] * 20 + ["2020-02-01"] * 20),
                     "ticker": [f"T{i}" for i in range(20)] * 2,
                     "value": list(range(20)) * 2,
                     "fwd_return": list(-np.arange(20) * 0.01) * 2}),
        feature_names=["value"])
    row = out.loc["value"]
    assert row["expected_sign"] == 1          # value registered as +1 expected
    assert row["realised_sign"] == -1         # this synthetic data has it inverted
    assert row["sign_matches_expected"] == False


def test_global_explain_bundles_everything(panel, bundle):
    out = global_explain(panel, task="regression", feature_names=ML_FEATURES,
                         bundle=bundle, n_splits=3, min_train_periods=8, alpha=10.0)
    for key in ("coefficient_stability", "permutation_importance_by_family",
               "single_feature_ic"):
        assert key in out and not out[key].empty
    assert "ic_by_year" in out
    assert "ic_by_regime" in out
