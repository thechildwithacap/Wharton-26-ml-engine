"""Macro-regime factor and portfolio Monte Carlo simulation."""

import numpy as np
import pandas as pd
import pytest

from wharton_ml_engine.data import SyntheticDataSource
from wharton_ml_engine.ml.dataset import ML_FEATURES, feature_frame
from wharton_ml_engine.risk import simulate_from_bundle, simulate_portfolio
from wharton_ml_engine.signals.macro import macro_factor_model, macro_regime


@pytest.fixture(scope="module")
def bundle():
    return SyntheticDataSource(n_tickers=40, years=6, seed=7).load()


# --- macro regime + factor ------------------------------------------------
def test_regime_fields_are_sane(bundle):
    reg = macro_regime(bundle, bundle.dates()[-1])
    assert 0.0 <= reg.risk_on <= 1.0
    assert reg.vol_annual > 0
    assert reg.label() in ("risk-on", "risk-off", "neutral")
    assert reg.drawdown <= 0.0


def test_macro_fit_is_cross_sectional(bundle):
    m = macro_factor_model(bundle, bundle.dates()[-1])
    assert "macro_fit" in m.columns
    assert m["macro_fit"].notna().all()
    # It must vary across names (not a single market-wide constant).
    assert m["macro_fit"].nunique() > 5
    assert m["macro_fit"].between(0, 100).all()


def test_macro_fit_alignment_matches_regime(bundle):
    """macro_fit rewards beta in the direction the regime tilts: the sign of
    corr(beta, macro_fit) must match the sign of the regime tilt (2*risk_on-1)."""
    as_of = bundle.dates()[-1]
    m = macro_factor_model(bundle, as_of)
    reg = macro_regime(bundle, as_of)
    tilt = 2.0 * reg.risk_on - 1.0
    corr = m["beta"].corr(m["macro_fit"])
    if abs(tilt) > 0.2:                      # only assert when the regime is decisive
        assert np.sign(corr) == np.sign(tilt)


def test_macro_fit_in_features(bundle):
    assert "macro_fit" in ML_FEATURES
    ff = feature_frame(bundle, bundle.dates()[-1])
    assert "macro_fit" in ff.columns and ff["macro_fit"].notna().all()


# --- Monte Carlo ----------------------------------------------------------
def _returns(seed=0, n=500, k=6):
    rng = np.random.default_rng(seed)
    cols = [f"S{i}" for i in range(k)]
    mkt = rng.normal(0.0004, 0.01, n)
    data = {c: 0.6 * mkt + rng.normal(0.0002, 0.012, n) for c in cols}
    return pd.DataFrame(data)


def test_mc_basic_distribution():
    rets = _returns()
    w = pd.Series(1.0 / rets.shape[1], index=rets.columns)
    r = simulate_portfolio(rets, w, horizon_days=63, n_sims=4000, method="block_bootstrap", seed=1)
    assert 0.0 <= r.prob_loss <= 1.0
    # percentiles are ordered
    p = r.percentiles
    assert p["p05"] < p["p50"] < p["p95"]
    assert r.var_95 >= 0 and r.cvar_95 >= r.var_95 - 1e-9   # CVaR at least as bad as VaR
    assert r.max_drawdown_median <= 0.0


def test_mc_t_has_fatter_tails_than_normal():
    rets = _returns(seed=2)
    w = [1.0 / rets.shape[1]] * rets.shape[1]
    n = simulate_portfolio(rets, w, horizon_days=63, n_sims=8000, method="normal", seed=3)
    t = simulate_portfolio(rets, w, horizon_days=63, n_sims=8000, method="t", t_df=4, seed=3)
    # Student-t tail loss should exceed the Gaussian tail loss.
    assert t.var_95 > n.var_95


def test_mc_drift_override_shifts_mean():
    rets = _returns(seed=5)
    w = pd.Series(1.0 / rets.shape[1], index=rets.columns)
    base = simulate_portfolio(rets, w, horizon_days=126, n_sims=6000, method="bootstrap", seed=4)
    hi = simulate_portfolio(rets, w, horizon_days=126, n_sims=6000, method="bootstrap",
                            annual_drift=0.30, seed=4)
    assert hi.median_return > base.median_return


def test_mc_from_bundle(bundle):
    held = bundle.tickers[:8]
    w = pd.Series(1.0 / len(held), index=held)
    r = simulate_from_bundle(bundle, w, horizon_days=63, n_sims=3000, seed=0)
    assert r.n_sims == 3000 and r.horizon_days == 63
    d = r.to_json()
    assert "percentiles" in d and "var_95" in d


def test_mc_needs_history():
    rets = _returns(n=10)
    with pytest.raises(ValueError):
        simulate_portfolio(rets, [1, 1, 1, 1, 1, 1], horizon_days=20, n_sims=100)
