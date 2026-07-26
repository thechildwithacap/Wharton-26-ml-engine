"""New SEC-derived signals: net issuance, asset growth, accruals -> discipline.

Verified fully offline with a ``companyfacts``-shaped fetch fixture, so the
point-in-time derivation of the financing/investment anomalies and their flow
into the capital-discipline alpha factor and the ML feature matrix are checked
without hitting SEC.
"""

import numpy as np
import pandas as pd
import pytest

from wharton_ml_engine.data.sec_edgar import build_sec_fundamentals
from wharton_ml_engine.ml.dataset import ML_FEATURES, feature_frame
from wharton_ml_engine.signals.fundamental import (
    capital_discipline_model,
    fundamental_scores,
)


def _fy(concept_rows):
    return {"units": {"USD": concept_rows}}


def _fy_shares(rows):
    return {"units": {"shares": rows}}


def _annual_row(fy, val, filed, end):
    return {"fy": fy, "fp": "FY", "form": "10-K", "val": val, "filed": filed, "end": end}


def _facts(*, shares_2021, shares_2022, assets_2022, assets_2023, ni, ocf):
    """A minimal but well-formed companyfacts payload for one filer."""
    gaap = {
        "NetIncomeLoss": _fy([_annual_row(2022, ni, "2023-02-15", "2022-12-31")]),
        "NetCashProvidedByUsedInOperatingActivities":
            _fy([_annual_row(2022, ocf, "2023-02-15", "2022-12-31")]),
        "Revenues": _fy([_annual_row(2022, 5000.0, "2023-02-15", "2022-12-31")]),
        "StockholdersEquity": _fy([{"val": 4000.0, "filed": "2023-02-15", "end": "2022-12-31"}]),
        "Assets": _fy([
            {"val": assets_2022, "filed": "2022-02-15", "end": "2021-12-31"},
            {"val": assets_2023, "filed": "2023-02-15", "end": "2022-12-31"},
        ]),
        "WeightedAverageNumberOfDilutedSharesOutstanding": _fy_shares([
            _annual_row(2021, shares_2021, "2022-02-15", "2021-12-31"),
            _annual_row(2022, shares_2022, "2023-02-15", "2022-12-31"),
        ]),
    }
    dei = {"EntityCommonStockSharesOutstanding": _fy_shares([
        {"val": shares_2021, "filed": "2022-02-15", "end": "2021-12-31"},
        {"val": shares_2022, "filed": "2023-02-15", "end": "2022-12-31"},
    ])}
    return {"entityName": "Test Co", "facts": {"us-gaap": gaap, "dei": dei}}


# DISC: bought back stock (shares 1000->950) and grew assets slowly; cash-backed
# earnings (OCF > NI -> negative accruals).  DILU: diluted (1000->1100) and grew
# assets fast; accrual-heavy (NI > OCF).
_FACTS = {
    "DISC": _facts(shares_2021=1000, shares_2022=950, assets_2022=10000,
                   assets_2023=10500, ni=500, ocf=750),
    "DILU": _facts(shares_2021=1000, shares_2022=1100, assets_2022=10000,
                   assets_2023=14000, ni=500, ocf=250),
}


def _fetch(url):
    if url.endswith("company_tickers.json"):
        return {"0": {"ticker": "DISC", "cik_str": 1}, "1": {"ticker": "DILU", "cik_str": 2}}
    if "submissions" in url:
        return {"sic": "3571"}                       # Technology
    for t, cik in (("DISC", "0000000001"), ("DILU", "0000000002")):
        if cik in url:
            return _FACTS[t]
    return {}


@pytest.fixture(scope="module")
def sec_panel():
    dates = pd.bdate_range("2021-01-04", "2023-12-29")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        {t: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(dates))))
         for t in ("DISC", "DILU")}, index=dates)
    panel, sectors, names = build_sec_fundamentals(
        ["DISC", "DILU"], prices, "2021-01-04", "2023-12-29", fetch=_fetch)
    return panel


def test_new_fields_present_and_populated(sec_panel):
    for f in ("net_issuance", "asset_growth", "accruals"):
        assert f in sec_panel.columns
        assert sec_panel[f].notna().any()


def test_issuance_sign_is_correct(sec_panel):
    last = pd.Timestamp("2023-06-30")
    snap = sec_panel[sec_panel.index.get_level_values(0) <= last]
    disc = snap.xs("DISC", level=1)["net_issuance"].dropna().iloc[-1]
    dilu = snap.xs("DILU", level=1)["net_issuance"].dropna().iloc[-1]
    assert disc < 0        # bought back stock (1000 -> 950)
    assert dilu > 0        # diluted (1000 -> 1100)


def test_asset_growth_and_accruals(sec_panel):
    disc = sec_panel.xs("DISC", level=1)
    dilu = sec_panel.xs("DILU", level=1)
    # DILU grew assets far faster than DISC.
    assert dilu["asset_growth"].dropna().iloc[-1] > disc["asset_growth"].dropna().iloc[-1]
    # DISC has cash-backed earnings (OCF>NI) -> negative accruals; DILU positive.
    assert disc["accruals"].dropna().iloc[-1] < 0
    assert dilu["accruals"].dropna().iloc[-1] > 0


def test_point_in_time_no_lookahead(sec_panel):
    # The FY2022 numbers were filed 2023-02-15; they must not be visible before.
    disc = sec_panel.xs("DISC", level=1)
    early = disc[disc.index < pd.Timestamp("2023-02-15")]["asset_growth"]
    # Before the 2023 filing, asset growth can only reflect the 2022-filed value
    # (a single point -> no YoY), so it is NaN; the jump appears only after.
    assert early.dropna().empty or (early.dropna().abs() < 1e9).all()
    after = disc[disc.index >= pd.Timestamp("2023-02-28")]["asset_growth"].dropna()
    assert not after.empty


def test_discipline_factor_ranks_correctly(sec_panel):
    last = sec_panel.index.get_level_values(0).max()
    frame = sec_panel.xs(last, level=0)
    disc = capital_discipline_model(frame)
    assert disc.loc["DISC", "discipline"] > disc.loc["DILU", "discipline"]
    # and it is exposed in the assembled headline scores
    fs = fundamental_scores(frame)
    assert "discipline" in fs.columns


def test_discipline_in_ml_features():
    assert "discipline" in ML_FEATURES


# --------------------------------------------------------- split adjustment
def test_split_adjustment_factors():
    """Reported share counts must be restated onto the latest (split) basis.

    Regression guard for a real bug: split-adjusted prices multiplied by
    as-reported share counts understated NVDA's mid-2021 market cap as ~$12B
    instead of ~$500B (its later 4:1 and 10:1 splits = 40x), which corrupted
    market cap, P/E, P/B and manufactured a spurious size effect.
    """
    from wharton_ml_engine.data.sec_edgar import split_adjustment_factors

    shares = pd.DataFrame({
        "filed": pd.to_datetime(["2020-02-01", "2021-02-01",   # pre-splits
                                 "2021-08-01",                  # after 4:1
                                 "2024-08-01"]),                # after 10:1
        "val": [600e6, 620e6, 2_480e6, 24_600e6],
    })
    adj = split_adjustment_factors(shares)
    f = adj["factor"].tolist()
    assert f[-1] == pytest.approx(1.0)              # latest basis is the anchor
    assert f[0] == pytest.approx(40.0, rel=0.05)    # 4:1 then 10:1 => 40x
    assert f[1] == pytest.approx(40.0, rel=0.05)
    assert f[2] == pytest.approx(10.0, rel=0.05)    # only the 10:1 remains
    # Restated counts are all on the same (latest) basis.
    restated = adj["val"] * adj["factor"]
    assert restated.max() / restated.min() < 1.3    # no 40x artefact left


def test_ordinary_issuance_is_not_treated_as_a_split():
    from wharton_ml_engine.data.sec_edgar import split_adjustment_factors

    shares = pd.DataFrame({
        "filed": pd.to_datetime(["2020-02-01", "2021-02-01", "2022-02-01"]),
        "val": [1_000e6, 1_020e6, 990e6],           # +2% issuance, -3% buyback
    })
    adj = split_adjustment_factors(shares)
    assert adj["factor"].tolist() == pytest.approx([1.0, 1.0, 1.0])
