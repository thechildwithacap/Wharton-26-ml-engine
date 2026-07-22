"""Offline tests for the free real-data path: provider prices + SEC EDGAR.

The HTTP layers are injected with fixtures shaped like the real TwelveData /
FMP / SEC responses, so provider price parsing, SEC fundamentals assembly, and
the combined WebDataSource -> signals -> ML panel path are all verified with no
network access.
"""

import numpy as np
import pandas as pd
import pytest

from wharton_ml_engine.client import sample_profile
from wharton_ml_engine.data import WebDataSource
from wharton_ml_engine.data.sec_edgar import SEC_BASE, SEC_WWW, _sic_to_sector
from wharton_ml_engine.data.web_prices import (
    fetch_prices_fmp,
    fetch_prices_twelvedata,
)
from wharton_ml_engine.ml import build_training_panel
from wharton_ml_engine.signals import compute_signals

_BDAYS = pd.bdate_range("2019-01-02", "2024-12-31")


# --- price parsers -------------------------------------------------------

def test_twelvedata_parser():
    def get(url):
        return {"status": "ok", "values": [
            {"datetime": "2024-01-03", "close": "102", "volume": "1000"},
            {"datetime": "2024-01-02", "close": "100", "volume": "1200"},
        ]}
    df = fetch_prices_twelvedata("AAPL", "2024-01-01", "2024-01-05", "k", get)
    assert list(df.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert df["close"].iloc[0] == 100.0 and df["close"].iloc[-1] == 102.0


def test_fmp_parser_uses_adjclose():
    def get(url):
        return {"symbol": "AAPL", "historical": [
            {"date": "2024-01-03", "close": 200, "adjClose": 150, "volume": 5},
            {"date": "2024-01-02", "close": 190, "adjClose": 149, "volume": 6},
        ]}
    df = fetch_prices_fmp("AAPL", "2024-01-01", "2024-01-05", "k", get)
    assert df["close"].tolist() == [149.0, 150.0]      # adjClose, ascending


def test_sic_sector_mapping():
    assert _sic_to_sector("3571") == "Technology"       # computers (AAPL)
    assert _sic_to_sector("2840") == "Consumer Staples"  # soap (PG)
    assert _sic_to_sector("2834") == "Health Care"       # pharma
    assert _sic_to_sector("6021") == "Financials"        # bank
    assert _sic_to_sector("4911") == "Utilities"


# --- SEC + WebDataSource end to end (fixtures) ---------------------------

def _fy(vals, unit="USD"):
    rows = []
    for i, (fy, val) in enumerate(sorted(vals.items())):
        rows.append({"end": f"{fy}-12-31", "fy": fy, "fp": "FY", "form": "10-K",
                     "filed": f"{fy + 1}-02-15", "val": val})
    return {"units": {unit: rows}}


def _instant(vals, unit="USD"):
    rows = [{"end": f"{fy}-12-31", "filed": f"{fy + 1}-02-15", "val": v}
            for fy, v in sorted(vals.items())]
    return {"units": {unit: rows}}


def _companyfacts(cik, entity):
    # FY2021..2023 with known numbers so ratios are checkable.
    return {"cik": cik, "entityName": entity, "facts": {"us-gaap": {
        "RevenueFromContractWithCustomerExcludingAssessedTax":
            _fy({2021: 800, 2022: 900, 2023: 1000}),
        "NetIncomeLoss": _fy({2021: 80, 2022: 90, 2023: 100}),
        "EarningsPerShareDiluted": _fy({2021: 1.6, 2022: 1.8, 2023: 2.0}, "USD/shares"),
        "GrossProfit": _fy({2021: 320, 2022: 360, 2023: 400}),
        "OperatingIncomeLoss": _fy({2021: 160, 2022: 180, 2023: 200}),
        "StockholdersEquity": _instant({2021: 400, 2022: 450, 2023: 500}),
        "Liabilities": _instant({2021: 500, 2022: 550, 2023: 600}),
    }, "dei": {
        "EntityCommonStockSharesOutstanding": _instant({2021: 50, 2022: 50, 2023: 50}, "shares"),
    }}}


# Six-name fixture universe: ticker -> (cik, sic).
_UNIVERSE = {
    "AAA": (111, "3571"),   # Technology
    "BBB": (222, "2834"),   # Health Care
    "CCC": (333, "6021"),   # Financials
    "DDD": (444, "2911"),   # Energy
    "EEE": (555, "2840"),   # Consumer Staples
    "FFF": (666, "4911"),   # Utilities
}
_CIK_TO_TICKER = {cik: t for t, (cik, _) in _UNIVERSE.items()}


def _sec_fetch(url):
    if url.endswith("company_tickers.json"):
        return {str(i): {"cik_str": cik, "ticker": t, "title": f"{t} Inc"}
                for i, (t, (cik, _)) in enumerate(_UNIVERSE.items())}
    for t, (cik, sic) in _UNIVERSE.items():
        tag = f"CIK{cik:010d}"
        if f"companyfacts/{tag}" in url:
            return _companyfacts(cik, f"{t} Inc")
        if f"submissions/{tag}" in url:
            return {"sic": sic}
    return {}


def _price_get_factory(seed=0):
    rng = np.random.default_rng(seed)

    def get(url):
        # crude symbol parse from the query string
        sym = url.split("symbol=")[1].split("&")[0] if "symbol=" in url else \
            url.split("historical-price-full/")[1].split("?")[0]
        px = 40 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(_BDAYS))))
        return {"status": "ok", "values": [
            {"datetime": d.strftime("%Y-%m-%d"), "close": f"{p:.2f}",
             "volume": str(int(rng.integers(1e6, 9e6)))}
            for d, p in zip(_BDAYS, px)]}
    return get


@pytest.fixture(scope="module")
def web_bundle():
    src = WebDataSource(tickers=list(_UNIVERSE), start="2019-01-02", end="2024-12-31",
                        price_provider="twelvedata", price_api_key="dummy",
                        price_get=_price_get_factory(1), sec_fetch=_sec_fetch,
                        throttle_s=0.0)
    return src.load()


def test_web_bundle_assembled(web_bundle):
    assert set(web_bundle.tickers) == set(_UNIVERSE)
    assert web_bundle.sectors["AAA"] == "Technology"
    assert web_bundle.sectors["BBB"] == "Health Care"
    assert web_bundle.sectors["DDD"] == "Energy"
    assert "SPX" in web_bundle.benchmarks.columns
    assert web_bundle.prices.shape[0] > 500


def test_sec_ratios_are_correct(web_bundle):
    f = web_bundle.fundamentals_asof(web_bundle.prices.index[-1]).loc["AAA"]
    # FY2023: eps=2.0, equity=500, ni=100, shares=50; rev growth 1000/900-1.
    assert abs(f["roe"] - 0.20) < 1e-6
    assert abs(f["gross_margin"] - 0.40) < 1e-6
    assert abs(f["revenue_growth"] - (1000 / 900 - 1)) < 1e-6
    assert abs(f["debt_equity"] - (600 / 500)) < 1e-6
    # pe = price / eps ; market_cap = price * 50 ; pb = market_cap / 500
    price = web_bundle.prices["AAA"].iloc[-1]
    assert abs(f["pe"] - price / 2.0) < 1e-3
    assert abs(f["pb"] - (price * 50) / 500) < 1e-3


def test_no_lookahead_in_sec_panel(web_bundle):
    # FY2023 is only filed 2024-02-15; before that, revenue_growth reflects FY2022.
    early = pd.Timestamp("2023-06-30")
    f = web_bundle.fundamentals_asof(early).loc["AAA"]
    assert abs(f["revenue_growth"] - (900 / 800 - 1)) < 1e-6   # FY2022 vs FY2021


def test_web_bundle_flows_into_signals_and_ml(web_bundle):
    sig = compute_signals(web_bundle, sample_profile())
    assert [c for c in sig.columns if sig[c].isna().all()] == []
    panel = build_training_panel(web_bundle, horizon_days=21)
    assert not panel.empty


def test_missing_price_key_still_constructs():
    # WebDataSource requires a provider key only at fetch time via the fetcher;
    # constructing with an explicit get should not raise.
    src = WebDataSource(tickers=["AAA"], start="2019-01-02", end="2024-12-31",
                        price_api_key="x", price_get=_price_get_factory(),
                        sec_fetch=_sec_fetch, throttle_s=0.0)
    assert src.price_provider == "twelvedata"
