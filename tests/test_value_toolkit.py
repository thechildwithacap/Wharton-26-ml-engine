"""Piotroski F-score, earnings-event overreaction, and SEC Form 4 insider data.

All three are verified offline: the F-score and event model on constructed
frames, and the insider adapter against a fixture shaped like SEC's
``submissions`` JSON plus a Form 4 XML document.
"""

import numpy as np
import pandas as pd
import pytest

from wharton_ml_engine.data.insider import (
    fetch_insider_activity,
    insider_score,
)
from wharton_ml_engine.data import SyntheticDataSource
from wharton_ml_engine.signals.events import earnings_event_model
from wharton_ml_engine.signals.piotroski import F_TESTS, piotroski_f_score


# --------------------------------------------------------------- Piotroski
_CUR = pd.DataFrame({
    "roic": [0.15, 0.02, -0.05], "roe": [0.18, 0.03, -0.06],
    "fcf_yield": [0.08, 0.01, -0.02], "accruals": [-0.02, 0.04, 0.09],
    "debt_equity": [0.4, 1.2, 2.5], "interest_coverage": [15.0, 4.0, 1.2],
    "net_issuance": [-0.03, 0.01, 0.06], "gross_margin": [0.45, 0.30, 0.22],
    "revenue_growth": [0.12, 0.02, -0.05], "asset_growth": [0.04, 0.06, 0.10],
}, index=["STRONG", "MID", "TRAP"])
_PRIOR = pd.DataFrame({
    "roic": [0.10, 0.03, 0.02], "roe": [0.12, 0.04, 0.03],
    "debt_equity": [0.6, 1.0, 2.0], "interest_coverage": [10.0, 5.0, 3.0],
    "gross_margin": [0.40, 0.31, 0.28],
}, index=["STRONG", "MID", "TRAP"])


def test_f_score_ranks_strength():
    f = piotroski_f_score(_CUR, _PRIOR)
    assert f.loc["STRONG", "f_score"] == 9          # passes every test
    assert f.loc["TRAP", "f_score"] == 0            # fails every test
    assert f.loc["STRONG", "f_score"] > f.loc["MID", "f_score"] > f.loc["TRAP", "f_score"]
    assert bool(f.loc["STRONG", "f_strong"]) is True
    assert bool(f.loc["TRAP", "f_weak"]) is True


def test_f_score_counts_all_nine_tests():
    f = piotroski_f_score(_CUR, _PRIOR)
    assert list(f.columns[:len(F_TESTS)]) == F_TESTS
    assert (f["f_evaluated"] == 9).all()
    assert (f["f_missing"] == 0).all()


def test_missing_data_is_not_scored_as_failure():
    # Without a prior frame the four YoY tests cannot be evaluated; they must be
    # reported missing rather than counted as failed.
    f = piotroski_f_score(_CUR, prior=None)
    assert (f["f_evaluated"] < 9).all()
    assert (f["f_missing"] > 0).all()
    # STRONG still passes every test that *can* be evaluated.
    assert f.loc["STRONG", "f_score"] == f.loc["STRONG", "f_evaluated"]
    assert f.loc["STRONG", "f_score_pct"] == pytest.approx(1.0)


# ------------------------------------------------------------- event model
def test_earnings_event_model_shape():
    b = SyntheticDataSource(n_tickers=25, years=4, seed=11).load()
    ev = earnings_event_model(b, b.dates()[-1])
    for c in ("days_since_event", "event_reaction", "earnings_drift",
              "fundamental_trend", "overreaction"):
        assert c in ev.columns
    assert ev["overreaction"].between(0, 100).all()
    assert ev["fundamental_trend"].between(0, 100).all()


def test_overreaction_rewards_punished_but_improving():
    b = SyntheticDataSource(n_tickers=40, years=4, seed=5).load()
    ev = earnings_event_model(b, b.dates()[-1]).dropna(
        subset=["event_reaction", "overreaction"])
    if len(ev) < 10:
        pytest.skip("not enough event data in this synthetic sample")
    # Among names with a similar fundamental trend, a worse event reaction must
    # not score lower — the signal is contrarian by construction.
    top = ev.nlargest(10, "fundamental_trend")
    corr = top["event_reaction"].corr(top["overreaction"])
    assert corr <= 0.5          # not a momentum signal in disguise


# ---------------------------------------------------------------- insider
_SUBMISSIONS = {
    "filings": {"recent": {
        "form": ["4", "4", "10-K", "4", "8-K"],
        "filingDate": ["2025-11-10", "2025-10-02", "2025-09-30",
                       "2024-01-05", "2025-11-01"],
        "accessionNumber": ["0000000001-25-000001", "0000000001-25-000002",
                            "0000000001-25-000003", "0000000001-24-000004",
                            "0000000001-25-000005"],
    }}
}
_FORM4_BUY = """
<ownershipDocument>
 <transactionCode>P</transactionCode>
 <transactionShares><value>1000</value></transactionShares>
 <transactionPricePerShare><value>50.00</value></transactionPricePerShare>
</ownershipDocument>
"""
_FORM4_SELL = """
<ownershipDocument>
 <transactionCode>S</transactionCode>
 <transactionShares><value>400</value></transactionShares>
 <transactionPricePerShare><value>50.00</value></transactionPricePerShare>
</ownershipDocument>
"""


def _fetch(url):
    if url.endswith(".json"):
        return _SUBMISSIONS
    # Route on the accession suffix: ...-000001 is the buy, ...-000002 the sale.
    return _FORM4_BUY if url.split("/")[-1].startswith(
        "0000000001-25-000001") else _FORM4_SELL


def test_insider_counts_are_point_in_time():
    act = fetch_insider_activity(
        ["AAA"], {"AAA": "0000000001"}, pd.Timestamp("2025-12-01"),
        lookback_days=180, fetch=_fetch)
    # Two Form 4s fall inside the window; the 2024 one and the non-Form-4s don't.
    assert act.counts.loc["AAA", "filings"] == 2
    assert act.has_detail is False           # no detail requested


def test_insider_detail_parses_buys_and_sells():
    act = fetch_insider_activity(
        ["AAA"], {"AAA": "0000000001"}, pd.Timestamp("2025-12-01"),
        lookback_days=180, fetch=_fetch, fetch_detail=True)
    assert act.has_detail is True
    assert act.counts.loc["AAA", "buys"] == 1
    assert act.counts.loc["AAA", "sells"] == 1
    assert act.counts.loc["AAA", "buy_value"] == pytest.approx(50_000.0)
    assert act.counts.loc["AAA", "sell_value"] == pytest.approx(20_000.0)


def test_insider_score_flags_whether_detail_was_used():
    shallow = fetch_insider_activity(["AAA"], {"AAA": "0000000001"},
                                     pd.Timestamp("2025-12-01"), fetch=_fetch)
    s = insider_score(shallow)
    assert bool(s.loc["AAA", "detail"]) is False     # filing-count proxy only
    assert np.isnan(s.loc["AAA", "insider_net"])

    deep = fetch_insider_activity(["AAA"], {"AAA": "0000000001"},
                                  pd.Timestamp("2025-12-01"), fetch=_fetch,
                                  fetch_detail=True)
    d = insider_score(deep)
    assert bool(d.loc["AAA", "detail"]) is True
    # buy value exceeds sell value -> positive net buying
    assert d.loc["AAA", "insider_net"] > 0
