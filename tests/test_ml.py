import numpy as np
import pandas as pd
import pytest

from wharton_ml_engine import run_engine, sample_profile
from wharton_ml_engine.ml import (
    AlphaModel,
    LogisticRegressor,
    RidgeRegressor,
    StandardScaler,
    build_training_panel,
    feature_frame,
    train_alpha_model,
    walk_forward_evaluate,
)
from wharton_ml_engine.ml.dataset import ML_FEATURES


@pytest.fixture(scope="module")
def ml_bundle():
    from wharton_ml_engine import SyntheticDataSource
    return SyntheticDataSource(n_tickers=45, years=8, seed=13).load()


@pytest.fixture(scope="module")
def panel(ml_bundle):
    return build_training_panel(ml_bundle, horizon_days=21)


@pytest.fixture(scope="module")
def trained(ml_bundle, panel):
    return train_alpha_model(ml_bundle, task="regression", panel=panel, alpha=10.0)


# --- numpy models --------------------------------------------------------

def test_standard_scaler_roundtrip():
    X = np.random.RandomState(0).randn(50, 4) * 3 + 7
    sc = StandardScaler().fit(X)
    Xt = sc.transform(X)
    assert np.allclose(Xt.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(Xt.std(axis=0), 1, atol=1e-9)
    sc2 = StandardScaler.from_dict(sc.to_dict())
    assert np.allclose(sc.transform(X), sc2.transform(X))


def test_ridge_recovers_linear_signal():
    rng = np.random.RandomState(1)
    X = rng.randn(400, 3)
    true_w = np.array([2.0, -1.0, 0.5])
    y = X @ true_w + 4.0 + rng.randn(400) * 0.01
    m = RidgeRegressor(alpha=1e-6).fit(X, y)
    assert np.allclose(m.coef_, true_w, atol=0.05)
    assert abs(m.intercept_ - 4.0) < 0.05


def test_logistic_learns_separation():
    rng = np.random.RandomState(2)
    X = rng.randn(300, 2)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    m = LogisticRegressor(l2=0.01, lr=0.5, epochs=800).fit(X, y)
    acc = (m.predict(X) == y).mean()
    assert acc > 0.85
    assert m.loss_history_[-1] < m.loss_history_[0]


# --- dataset / no look-ahead --------------------------------------------

def test_feature_frame_is_0_100(ml_bundle):
    f = feature_frame(ml_bundle, ml_bundle.dates()[-1])
    assert list(f.columns) == ML_FEATURES
    v = f.dropna()
    assert v.to_numpy().min() >= 0 and v.to_numpy().max() <= 100


def test_panel_labels_are_relative(panel):
    assert not panel.empty
    # fwd_excess is cross-sectionally demeaned => ~0 mean per date.
    per_date_mean = panel.groupby("date")["fwd_excess"].mean().abs().max()
    assert per_date_mean < 1e-9
    # outperform is a median split => ~half ones.
    assert 0.4 < panel["outperform"].mean() < 0.6


def test_panel_has_no_lookahead(ml_bundle, panel):
    # The last usable feature date must leave a full forward window of prices.
    last_date = panel["date"].max()
    idx = list(ml_bundle.prices.index)
    assert idx.index(last_date) + 21 < len(idx)


# --- training / evaluation ----------------------------------------------

def test_walk_forward_returns_ic(panel):
    ev = walk_forward_evaluate(panel, task="regression", n_splits=4, alpha=10.0)
    assert "summary" in ev
    assert ev["summary"]["n_periods"] > 0
    assert not ev["oos"].empty
    assert {"date", "ticker", "pred", "fwd_return"}.issubset(ev["oos"].columns)


def test_trained_model_metrics_and_coeffs(trained):
    assert "mean_ic" in trained.metrics
    assert len(trained.coefficients()) == len(ML_FEATURES)


def test_save_load_roundtrip_preserves_predictions(trained, ml_bundle, tmp_path):
    path = tmp_path / "m.json"
    trained.save(str(path))
    from wharton_ml_engine.ml import TrainedModel
    reloaded = TrainedModel.load(str(path))
    f = feature_frame(ml_bundle, ml_bundle.dates()[-1])
    a = trained.raw_predict(f)
    b = reloaded.raw_predict(f)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_alpha_model_score_bounds(trained, ml_bundle):
    s = AlphaModel(trained).score(ml_bundle)
    v = s.dropna()
    assert v.min() >= 0 and v.max() <= 100
    assert s.notna().sum() > 0


# --- engine integration --------------------------------------------------

def test_engine_uses_ml_alpha(trained, ml_bundle):
    profile = sample_profile()
    report = run_engine(ml_bundle, profile, alpha_model=AlphaModel(trained),
                        run_backtest=False)
    assert "ml_alpha" in report.signals.columns
    assert report.ml_metrics is not None
    assert report.mandate.passed
