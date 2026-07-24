"""Tests for the point-in-time S&P 500 universe (Task E)."""

import pandas as pd

from wharton_ml_engine import SyntheticDataSource
from wharton_ml_engine.config import UniverseFilter
from wharton_ml_engine.data import (
    SP500_REMOVALS,
    estimate_survivorship_bias,
    sp500_pit_metadata,
)


def test_pit_metadata_marks_removed_names():
    meta = sp500_pit_metadata(["AAPL", "MSFT"], SP500_REMOVALS)
    assert meta["AAPL"]["delisting_date"] is None          # survivor
    assert meta["SIVB"]["delisting_date"] == "2023-03-15"  # removed (bank failure)
    assert meta["SIVB"]["outcome"] == "failed"


def test_removals_include_the_canonical_failures():
    failures = {r["ticker"] for r in SP500_REMOVALS if r["outcome"] == "failed"}
    assert {"SIVB", "SBNY", "FRC"} <= failures             # the 2023 bank failures


def test_bias_estimate_is_finite_and_documented():
    est = estimate_survivorship_bias(n_survivors=500, years=6.0)
    d = est.as_dict()
    assert d["n_removed"] == len(SP500_REMOVALS)
    assert sum(d["removal_counts"].values()) == len(SP500_REMOVALS)
    assert isinstance(d["est_survivorship_bias_annual"], float)
    assert "documented outcome returns" in d["method"]


def test_framework_reuse_drops_removed_after_date():
    # The S&P delisting dates plug straight into the existing eligible_asof.
    b = SyntheticDataSource(n_tickers=10, years=6, seed=1).load()
    t = b.tickers[0]
    b.meta[t].delisting_date = "2023-03-15"                # pretend it was removed
    uf = UniverseFilter(min_price=0.0, min_adv_usd=0.0)
    assert t in b.eligible_asof("2023-01-01", uf)          # before removal
    assert t not in b.eligible_asof("2023-06-01", uf)      # after removal
