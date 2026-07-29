"""Merging cached bundles into one wider universe (large + mid + small caps)."""

import numpy as np
import pandas as pd
import pytest

from wharton_ml_engine.data import SyntheticDataSource, merge_bundles
from wharton_ml_engine.data.indices import MID, SMALL, available_universe
from wharton_ml_engine.signals import compute_signals
from wharton_ml_engine.client import sample_profile


def _bundle(seed, n=12, years=4):
    return SyntheticDataSource(n_tickers=n, years=years, seed=seed).load()


def _rename(b, prefix):
    """Relabel a synthetic bundle's tickers so two bundles don't collide."""
    from wharton_ml_engine.data.source import DataBundle, SecurityMeta
    mapping = {t: f"{prefix}{i}" for i, t in enumerate(b.prices.columns)}
    prices = b.prices.rename(columns=mapping)
    f = b.fundamentals.reset_index()
    f["ticker"] = f["ticker"].map(mapping)
    fundamentals = f.dropna(subset=["ticker"]).set_index(["date", "ticker"]).sort_index()
    meta = {mapping[t]: SecurityMeta(mapping[t], m.name, m.sector)
            for t, m in b.meta.items() if t in mapping}
    return DataBundle(prices=prices, benchmarks=b.benchmarks,
                      fundamentals=fundamentals, meta=meta)


def test_merge_widens_the_universe():
    a = _rename(_bundle(1), "LC")
    b = _rename(_bundle(2), "SC")
    m = merge_bundles([a, b])
    assert len(m.tickers) == len(a.tickers) + len(b.tickers)
    assert set(a.tickers).issubset(m.tickers)
    assert set(b.tickers).issubset(m.tickers)
    # fundamentals from both batches survive
    tick = m.fundamentals.index.get_level_values(1)
    assert tick.str.startswith("LC").any() and tick.str.startswith("SC").any()


def test_merge_keeps_metadata_and_runs_signals():
    m = merge_bundles([_rename(_bundle(3), "A"), _rename(_bundle(4), "B")])
    assert set(m.meta) == set(m.tickers)
    sig = compute_signals(m, sample_profile())
    all_nan = [c for c in sig.columns if sig[c].isna().all()]
    assert all_nan == []


def test_duplicate_tickers_resolve_to_first_bundle():
    a = _rename(_bundle(5), "X")
    b = _rename(_bundle(6), "X")          # same names, different data
    m = merge_bundles([a, b])
    assert len(m.tickers) == len(a.tickers)      # no duplicate columns
    # the first bundle's prices win
    common = a.prices.columns[0]
    joined = m.prices[common].reindex(a.prices.index).dropna()
    assert np.allclose(joined.values, a.prices[common].reindex(joined.index).values)


def test_no_price_invented_before_a_names_first_observation():
    a = _rename(_bundle(7, years=5), "OLD")
    b = _rename(_bundle(8, years=2), "NEW")      # shorter history
    m = merge_bundles([a, b])
    newcol = [c for c in m.prices.columns if c.startswith("NEW")][0]
    first_real = b.prices[b.prices.columns[0]].first_valid_index()
    before = m.prices.loc[m.prices.index < first_real, newcol]
    assert before.isna().all()                   # no back-filled fake history


def test_smid_index_definitions_registered():
    tickers = ["CALM", "CROX", "AAON", "AAPL", "ZZZZ"]
    sectors = {t: "Industrials" for t in tickers}
    small = available_universe("SMALL", tickers, sectors)
    mid = available_universe("MID", tickers, sectors)
    smid = available_universe("SMID", tickers, sectors)
    assert "CALM" in small and "CALM" not in mid
    assert "AAON" in mid and "CROX" in mid
    assert "CALM" in smid and "AAON" in smid
    assert "AAPL" not in smid and "ZZZZ" not in smid
    assert len(SMALL) > 100 and len(MID) > 30


# ------------------------------------------------- free-tier rate limiting
def test_rate_limit_derives_throttle_and_records_failures():
    """Regression guard: a 228-name fetch once returned only 17 names because
    the default throttle (0.15s ~ 400 req/min) blew through TwelveData's free
    8/min cap and every throttled response was silently discarded."""
    from wharton_ml_engine.data.web_source import WebDataSource

    src = WebDataSource(tickers=["AAA"], start="2020-01-01", end="2020-12-31",
                        price_api_key="k", requests_per_minute=8)
    # 8/min => at least ~7.5s between calls, not sub-second.
    assert src.throttle_s >= 7.0
    assert src.failed == []

    # An explicit throttle still wins (used by tests / paid tiers).
    fast = WebDataSource(tickers=["AAA"], start="2020-01-01", end="2020-12-31",
                         price_api_key="k", throttle_s=0.0)
    assert fast.throttle_s == 0.0


def test_failed_symbols_are_reported_not_silently_dropped():
    from wharton_ml_engine.data.web_source import WebDataSource

    dates = pd.bdate_range("2020-01-02", periods=120)

    def price_get(url):
        # GOOD returns a usable series; BAD always fails (simulating throttling).
        if "GOOD" in url:
            return {"values": [{"datetime": d.strftime("%Y-%m-%d"), "close": "10.0",
                                "volume": "1000000"} for d in dates]}
        return {"status": "error", "message": "rate limit"}

    src = WebDataSource(tickers=["GOOD", "BAD"], start="2020-01-02",
                        end="2020-06-30", price_api_key="k", throttle_s=0.0,
                        max_retries=1, price_get=price_get,
                        sec_fetch=lambda url: {})
    try:
        src.load()
    except Exception:
        pass                       # SEC side is stubbed out; we only assert bookkeeping
    assert "BAD" in src.failed     # the miss is recorded, not hidden
    assert "GOOD" not in src.failed
