"""Tests for the benchmark-tilt and concentration construction knobs."""

import numpy as np
import pandas as pd
import pytest

from wharton_ml_engine import EngineConfig, SyntheticDataSource, run_engine, sample_profile
from wharton_ml_engine.config import PortfolioConstraints


@pytest.fixture(scope="module")
def bundle():
    return SyntheticDataSource(n_tickers=50, years=5, seed=4).load()


def _run(bundle, profile, tilt=0.0, conc=1.0):
    cfg = EngineConfig()
    cfg.constraints.benchmark_tilt = tilt
    cfg.constraints.concentration = conc
    cfg.constraints.validate()
    return run_engine(bundle, profile, config=cfg, as_of=bundle.dates()[-1],
                      run_backtest=False)


def _wavg_cap(weights, mcap):
    w = pd.Series(weights)
    return float((w * mcap.reindex(w.index)).sum() / w.sum())


def _hhi(weights):
    w = np.array(list(weights.values()))
    return float((w ** 2).sum())


def test_benchmark_tilt_shifts_toward_large_caps(bundle):
    profile = sample_profile()
    mcap = bundle.fundamentals_asof(bundle.dates()[-1])["market_cap"]
    low = _run(bundle, profile, tilt=0.0).candidate_weights
    high = _run(bundle, profile, tilt=0.9).candidate_weights
    # A high tilt should hold materially larger-cap names on average.
    assert _wavg_cap(high, mcap) > _wavg_cap(low, mcap) * 1.15


def test_benchmark_tilt_default_is_neutral(bundle):
    profile = sample_profile()
    # tilt=0 must reproduce the untilted book exactly (deterministic).
    a = _run(bundle, profile, tilt=0.0).candidate_weights
    b = _run(bundle, profile, tilt=0.0).candidate_weights
    assert a == b


def test_concentration_increases_top_heaviness(bundle):
    profile = sample_profile()
    flat = _run(bundle, profile, conc=0.5).candidate_weights
    peaked = _run(bundle, profile, conc=3.0).candidate_weights
    # More concentration -> higher Herfindahl (fewer effective names).
    assert _hhi(peaked) >= _hhi(flat)


def test_tilted_book_still_respects_caps(bundle):
    profile = sample_profile()
    report = _run(bundle, profile, tilt=0.8, conc=2.0)
    cons = profile.apply_to_constraints(EngineConfig().constraints)
    w = report.candidate_weights
    assert max(w.values()) <= cons.max_weight_per_stock + 1e-6
    assert report.mandate.passed, report.mandate.violations


def test_constraints_validate_rejects_bad_knobs():
    with pytest.raises(AssertionError):
        PortfolioConstraints(benchmark_tilt=1.5).validate()
    with pytest.raises(AssertionError):
        PortfolioConstraints(concentration=0.1).validate()
