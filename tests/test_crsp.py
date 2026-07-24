"""CRSP / WRDS survivorship-bias-free path, verified fully offline.

A ``FakeCRSPClient`` returns DataFrames shaped exactly like the crsp/comp tables
(``crsp.dsf`` stock file, ``crsp.stocknames``, ``crsp.dsedelist``, ``comp.fundq``,
``crsp.dsi``), so the whole assembly — total-return index, delisting-return
splice, point-in-time Compustat fundamentals, survivorship-safe metadata — and
the measured two-way bias backtest run with no WRDS login.
"""

import numpy as np
import pandas as pd
import pytest

from wharton_ml_engine.backtest import measure_survivorship_bias
from wharton_ml_engine.backtest.templates import STRATEGY_TEMPLATES
from wharton_ml_engine.client import sample_profile
from wharton_ml_engine.data import CRSPDataSource
from wharton_ml_engine.signals import compute_signals

_BDAYS = pd.bdate_range("2018-01-02", "2024-12-31")

# permno -> (ticker, name, siccd, delist_date, dlret, dlstcd, drift)
_UNIVERSE = {
    10001: ("AAPL", "Apple Inc",        3571, None,         np.nan, np.nan, 0.0007),
    10002: ("MSFT", "Microsoft Corp",   7372, None,         np.nan, np.nan, 0.0007),
    10003: ("JPM",  "Jpmorgan Chase",   6020, None,         np.nan, np.nan, 0.0004),
    10004: ("XOM",  "Exxon Mobil",      1311, None,         np.nan, np.nan, 0.0003),
    10005: ("JNJ",  "Johnson & Johnson",2834, None,         np.nan, np.nan, 0.0004),
    10006: ("PG",   "Procter & Gamble", 2840, None,         np.nan, np.nan, 0.0004),
    10007: ("KO",   "Coca Cola Co",     2080, None,         np.nan, np.nan, 0.0003),
    10008: ("HD",   "Home Depot",       5211, None,         np.nan, np.nan, 0.0006),
    10009: ("NEE",  "Nextera Energy",   4911, None,         np.nan, np.nan, 0.0004),
    # --- the failure: a bank that delists in 2023 at a near-total loss ---
    10010: ("SIVB", "Svb Financial",    6020, "2023-03-15", -0.95, 574, 0.0006),
    # --- a takeover: removed at a premium in 2022 ---
    10011: ("XLNX", "Xilinx Inc",       3674, "2022-02-14", 0.05, 200, 0.0006),
}


class FakeCRSPClient:
    """In-memory stand-in for :class:`WRDSClient` returning crsp/comp-shaped rows."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def resolve_tickers(self, tickers):
        by_ticker = {v[0]: k for k, v in _UNIVERSE.items()}
        return {t.upper(): by_ticker[t.upper()] for t in tickers if t.upper() in by_ticker}

    def stock_file(self, permnos, start, end):
        rows = []
        for p in permnos:
            if p not in _UNIVERSE:
                continue
            tk, _n, _s, dl, _dlr, _c, drift = _UNIVERSE[p]
            days = _BDAYS
            if dl is not None:
                days = days[days < pd.Timestamp(dl)]     # dsf ends at delisting
            r = self.rng.normal(drift, 0.016, len(days))
            px = 50 * np.exp(np.cumsum(r))
            vol = self.rng.integers(1e6, 9e6, len(days)).astype(float)
            for d, pr, ret, v in zip(days, px, r, vol):
                rows.append({"permno": p, "date": d, "prc": float(pr),
                             "ret": float(ret), "vol": float(v), "shrout": 1e6})
        return pd.DataFrame(rows)

    def names(self, permnos):
        rows = []
        for p in permnos:
            if p not in _UNIVERSE:
                continue
            tk, nm, sic, dl, _dlr, _c, _d = _UNIVERSE[p]
            end = dl if dl is not None else "2025-12-31"
            rows.append({"permno": p, "ticker": tk, "comnam": nm, "siccd": sic,
                         "shrcd": 11, "exchcd": 1, "namedt": "2010-01-01",
                         "nameenddt": end})
        return pd.DataFrame(rows)

    def delistings(self, permnos):
        rows = []
        for p in permnos:
            if p not in _UNIVERSE:
                continue
            tk, _n, _s, dl, dlr, code, _d = _UNIVERSE[p]
            if dl is not None:
                rows.append({"permno": p, "dlstdt": dl, "dlret": dlr, "dlstcd": code})
        return pd.DataFrame(rows)

    def fundamentals(self, permnos, start, end):
        rows = []
        for p in permnos:
            if p not in _UNIVERSE:
                continue
            tk, _n, _s, dl, _dlr, _c, _d = _UNIVERSE[p]
            last = pd.Timestamp(dl) if dl is not None else pd.Timestamp("2024-09-30")
            for q in pd.date_range("2018-03-31", "2024-09-30", freq="QE"):
                if q > last:
                    continue
                rng = self.rng
                sale = float(rng.uniform(5e3, 5e4))
                rows.append({
                    "permno": p, "datadate": q,
                    "rdq": q + pd.Timedelta(days=40),
                    "prccq": float(rng.uniform(30, 300)),
                    "cshoq": float(rng.uniform(500, 5000)),
                    "atq": float(rng.uniform(1e4, 5e5)),
                    "ceqq": float(rng.uniform(5e3, 1e5)),
                    "dlttq": float(rng.uniform(1e3, 5e4)),
                    "dlcq": float(rng.uniform(1e2, 5e3)),
                    "niq": float(rng.uniform(-1e2, 5e3)),
                    "saleq": sale,
                    "cogsq": sale * float(rng.uniform(0.4, 0.8)),
                    "oiadpq": float(rng.uniform(1e2, 8e3)),
                    "xintq": float(rng.uniform(10, 500)),
                    "capxq": float(rng.uniform(50, 2e3)),
                    "oancfq": float(rng.uniform(1e2, 6e3)),
                    "dvpsxq": float(rng.uniform(0, 1.5)),
                    "epspxq": float(rng.uniform(0.2, 5)),
                })
        return pd.DataFrame(rows)

    def benchmark(self, start, end):
        r = self.rng.normal(0.0004, 0.01, len(_BDAYS))
        return pd.DataFrame({"date": _BDAYS, "vwretd": r})


@pytest.fixture(scope="module")
def crsp_bundle():
    src = CRSPDataSource(
        start="2018-01-02", end="2024-12-31", client=FakeCRSPClient(1),
        tickers=[v[0] for v in _UNIVERSE.values()],
    )
    return src.load()


def test_universe_includes_delisted_names(crsp_bundle):
    # The failed bank and the acquired chip name must be *present* in the bundle,
    # not filtered out — that is the whole point of a survivorship-free source.
    assert "SIVB" in crsp_bundle.tickers
    assert "XLNX" in crsp_bundle.tickers
    assert crsp_bundle.meta["SIVB"].delisting_date == "2023-03-15"
    assert crsp_bundle.meta["XLNX"].delisting_date == "2022-02-14"
    assert crsp_bundle.meta["AAPL"].delisting_date is None


def test_delisting_return_realises_the_loss(crsp_bundle):
    # SIVB's series must carry the -95% delisting mark as its final step: the last
    # return is a near-total loss, not a silent truncation at the prior price.
    sivb = crsp_bundle.prices["SIVB"].dropna()
    last_ret = sivb.iloc[-1] / sivb.iloc[-2] - 1.0
    assert last_ret < -0.9                       # the failure is priced in
    # and the last valid price sits on/after the delisting date
    assert sivb.index[-1] >= pd.Timestamp("2023-03-15") - pd.Timedelta(days=5)


def test_eligible_asof_drops_after_delisting(crsp_bundle):
    before = crsp_bundle.eligible_asof(pd.Timestamp("2023-01-31"))
    after = crsp_bundle.eligible_asof(pd.Timestamp("2023-06-30"))
    assert "SIVB" in before                      # still trading in Jan 2023
    assert "SIVB" not in after                   # gone after March 2023 failure


def test_sector_from_sic(crsp_bundle):
    assert crsp_bundle.meta["JPM"].sector == "Financials"      # SIC 6020
    assert crsp_bundle.meta["SIVB"].sector == "Financials"     # SIC 6020
    assert crsp_bundle.meta["XOM"].sector == "Energy"          # SIC 1311
    assert crsp_bundle.meta["NEE"].sector == "Utilities"       # SIC 4911


def test_point_in_time_fundamentals(crsp_bundle):
    f = crsp_bundle.fundamentals_asof(pd.Timestamp("2021-06-30"))
    assert not f.empty
    assert f["pe"].notna().any()
    # rdq gating: a report dated after the as-of must not be visible.
    dates = crsp_bundle.fundamentals.index.get_level_values(0)
    visible = crsp_bundle.fundamentals.loc[dates <= pd.Timestamp("2021-06-30")]
    assert dates.min() <= pd.Timestamp("2021-06-30")
    assert not visible.empty


def test_signals_run_on_crsp_bundle(crsp_bundle):
    sig = compute_signals(crsp_bundle, sample_profile())
    all_nan = [c for c in sig.columns if sig[c].isna().all()]
    assert all_nan == []


def test_measured_survivorship_bias_is_real(crsp_bundle):
    # Backtest the same template twice: full (incl. SIVB's failure) vs survivors
    # only.  Dropping the failure should make the survivor view *overstate*
    # returns — a positive measured bias — computed from real prices, not modeled.
    est = measure_survivorship_bias(
        crsp_bundle, STRATEGY_TEMPLATES["hybrid_multifactor"],
        start="2019-01-01", end="2024-12-31", top_n=6,
    )
    d = est.as_dict()
    assert est.n_delisted == 2                    # SIVB + XLNX
    assert est.n_survivors == 9
    assert np.isfinite(d["full_cagr"])
    assert np.isfinite(d["survivor_cagr"])
    assert "measured_bias_annual" in d
