"""Tests for the point-in-time / survivorship-safe universe framework."""

import numpy as np
import pandas as pd
import pytest

from wharton_ml_engine import EngineConfig, SyntheticDataSource, run_engine, sample_profile
from wharton_ml_engine.config import UniverseFilter


def test_synthetic_delistings_created():
    b = SyntheticDataSource(n_tickers=60, years=8, seed=7, delist_frac=0.3).load()
    delisted = [t for t, m in b.meta.items() if m.delisting_date]
    assert len(delisted) > 0
    # a delisted name has NaN prices after its delisting date
    t = delisted[0]
    dd = pd.Timestamp(b.meta[t].delisting_date)
    after = b.prices.loc[b.prices.index > dd, t]
    assert after.isna().all()


def test_eligible_asof_excludes_after_delisting():
    b = SyntheticDataSource(n_tickers=60, years=8, seed=7, delist_frac=0.3).load()
    uf = UniverseFilter(min_price=0.0, min_adv_usd=0.0)
    delisted = [(t, pd.Timestamp(m.delisting_date)) for t, m in b.meta.items()
                if m.delisting_date]
    t, dd = delisted[0]
    before = b.dates()[b.dates() < dd][-1]
    after = b.dates()[b.dates() > dd][0]
    assert t in b.eligible_asof(before, uf)      # listed before delisting
    assert t not in b.eligible_asof(after, uf)   # gone after delisting


def test_eligible_shrinks_over_time():
    b = SyntheticDataSource(n_tickers=80, years=8, seed=7, delist_frac=0.3).load()
    uf = UniverseFilter(min_price=0.0, min_adv_usd=0.0)
    early = len(b.eligible_asof(b.dates()[300], uf))
    late = len(b.eligible_asof(b.dates()[-1], uf))
    assert late < early                          # names delisted along the way


def test_liquidity_and_type_floors():
    b = SyntheticDataSource(n_tickers=40, years=5, seed=1).load()
    # An impossibly high ADV floor should exclude everyone.
    assert b.eligible_asof(b.dates()[-1], UniverseFilter(min_adv_usd=1e15)) == []
    # A non-common security type should be excluded.
    t = b.tickers[0]
    b.meta[t].security_type = "preferred"
    elig = b.eligible_asof(b.dates()[-1], UniverseFilter(min_price=0, min_adv_usd=0))
    assert t not in elig


def test_engine_respects_universe_filter():
    b = SyntheticDataSource(n_tickers=60, years=6, seed=2).load()
    # Mark half the names ineligible via security type; engine must not hold them.
    banned = set(b.tickers[:30])
    for t in banned:
        b.meta[t].security_type = "adr"
    cfg = EngineConfig(universe_filter=UniverseFilter(min_price=0, min_adv_usd=0))
    report = run_engine(b, sample_profile(), config=cfg, run_backtest=False)
    assert all(t not in banned for t in report.candidate_weights)
