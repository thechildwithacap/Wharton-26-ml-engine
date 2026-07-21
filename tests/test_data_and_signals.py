import numpy as np
import pandas as pd

from wharton_ml_engine.client import compute_client_fit
from wharton_ml_engine.signals import compute_signals
from wharton_ml_engine.utils import (
    max_drawdown,
    pct_rank,
    sharpe_ratio,
    weighted_blend,
)


def test_bundle_shapes_and_no_nan_prices(bundle):
    assert bundle.prices.shape[1] == 40
    assert not bundle.prices.isna().any().any()
    assert (bundle.prices > 0).all().all()


def test_fundamentals_asof_has_no_lookahead(bundle):
    dates = bundle.dates()
    mid = dates[len(dates) // 2]
    fund = bundle.fundamentals_asof(mid)
    # Every snapshot used must be dated on or before the as-of date.
    used_dates = bundle.fundamentals.index.get_level_values(0)
    assert used_dates.min() <= mid
    assert not fund.empty
    assert set(fund.index) == set(bundle.tickers)


def test_restrict_universe(bundle):
    keep = bundle.tickers[:10]
    sub = bundle.restrict_universe(keep)
    assert list(sub.prices.columns) == keep
    assert set(sub.meta) == set(keep)


def test_pct_rank_bounds():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    r = pct_rank(s, ascending=True)
    assert r.min() >= 0 and r.max() <= 100
    assert r.idxmax() == 3  # largest value ranks highest when ascending


def test_weighted_blend_handles_missing():
    a = pd.Series([100.0, np.nan], index=["x", "y"])
    b = pd.Series([0.0, 50.0], index=["x", "y"])
    out = weighted_blend({"a": a, "b": b}, {"a": 0.5, "b": 0.5})
    # y has only b present -> equals b; x is the average.
    assert out["y"] == 50.0
    assert out["x"] == 50.0


def test_signals_are_0_100(bundle, profile, as_of):
    sig = compute_signals(bundle, profile, as_of)
    for col in ["value", "quality", "growth", "garp", "income", "momentum",
                "low_vol", "factor", "macro_tilt", "analyst", "hybrid_alpha"]:
        v = sig[col].dropna()
        assert v.min() >= 0.0 and v.max() <= 100.0, col


def test_client_fit_excludes_avoided_sector(bundle, profile, as_of):
    fit = compute_client_fit(bundle, profile, as_of)
    excluded = fit[fit["hard_excluded"]]
    for t in excluded.index:
        assert bundle.meta[t].sector in profile.sector_avoidances
        assert fit.loc[t, "client_fit"] == 0.0


def test_risk_metric_helpers():
    rets = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    assert max_drawdown(rets) <= 0
    assert np.isfinite(sharpe_ratio(rets))
