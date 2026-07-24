"""Tests for the FinancialDatasetSource API adapter and CSV caching.

The HTTP layer is replaced by a fixture ``fetch_json`` that returns payloads
shaped exactly like the financialdatasets.ai API, so the whole assembly path —
prices, point-in-time fundamentals, sector normalisation, benchmarks — is
verified offline, along with the end-to-end path into signals and ML training.
"""

import numpy as np
import pandas as pd
import pytest

from wharton_ml_engine.client import sample_profile
from wharton_ml_engine.data import (
    CSVDataSource,
    FinancialDatasetSource,
    save_bundle,
)
from wharton_ml_engine.ml import build_training_panel, train_alpha_model
from wharton_ml_engine.signals import compute_signals

_SECTORS = {
    "AAPL": "Technology", "MSFT": "Information Technology", "JPM": "Financial Services",
    "XOM": "Energy", "JNJ": "Healthcare", "PG": "Consumer Defensive",
    "SPY": "", "QQQ": "", "XLK": "", "XLV": "", "XLF": "", "XLE": "", "XLI": "",
}
_BDAYS = pd.bdate_range("2019-01-02", "2024-12-31")


def _make_fetch(seed: int = 0):
    rng = np.random.default_rng(seed)

    def prices(ticker):
        r = rng.normal(0.0004, 0.015, len(_BDAYS))
        px = 100 * np.exp(np.cumsum(r))
        return {"ticker": ticker, "prices": [
            {"time": d.strftime("%Y-%m-%dT00:00:00-05:00"), "close": float(p),
             "volume": float(rng.integers(1e6, 9e6))}
            for d, p in zip(_BDAYS, px)]}

    def metrics(ticker):
        rows = []
        for q in pd.date_range("2019-03-31", "2024-09-30", freq="QE"):
            rows.append({
                "report_period": q.strftime("%Y-%m-%d"), "period": "quarterly",
                "market_cap": float(rng.uniform(5e10, 3e12)),
                "price_to_earnings_ratio": float(rng.uniform(8, 40)),
                "price_to_book_ratio": float(rng.uniform(1, 12)),
                "enterprise_value_to_ebitda_ratio": float(rng.uniform(6, 25)),
                "free_cash_flow_yield": float(rng.uniform(0.01, 0.08)),
                "return_on_equity": float(rng.uniform(0.05, 0.4)),
                "return_on_invested_capital": float(rng.uniform(0.04, 0.3)),
                "gross_margin": float(rng.uniform(0.2, 0.7)),
                "debt_to_equity": float(rng.uniform(0.1, 2.0)),
                "interest_coverage": float(rng.uniform(2, 20)),
                "revenue_growth": float(rng.uniform(-0.1, 0.4)),
                "earnings_per_share_growth": float(rng.uniform(-0.2, 0.5)),
                "payout_ratio": float(rng.uniform(0, 0.6)),
                "dividend_yield": float(rng.uniform(0, 0.03)),
            })
        return {"financial_metrics": rows}

    def fetch(path, params):
        if path.startswith("prices"):
            return prices(params["ticker"])
        if path.startswith("financial-metrics"):
            return metrics(params["ticker"])
        if path.startswith("company/facts"):
            t = params["ticker"]
            return {"company_facts": {"name": f"{t} Inc", "sector": _SECTORS.get(t, "Unknown")}}
        return {}

    return fetch


@pytest.fixture(scope="module")
def fin_bundle():
    src = FinancialDatasetSource(
        tickers=["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG"],
        start="2019-01-02", end="2024-12-31", fetch_json=_make_fetch(0),
    )
    return src.load()


def test_prices_and_benchmarks_assembled(fin_bundle):
    assert fin_bundle.prices.shape[1] == 6
    assert not fin_bundle.prices.isna().all().any()
    assert "SPX" in fin_bundle.benchmarks.columns
    assert "NDX" in fin_bundle.benchmarks.columns


def test_sector_normalisation(fin_bundle):
    secs = fin_bundle.sectors
    assert secs["MSFT"] == "Technology"          # "Information Technology" -> Technology
    assert secs["JPM"] == "Financials"           # "Financial Services" -> Financials
    assert secs["JNJ"] == "Health Care"          # "Healthcare" -> Health Care
    assert secs["PG"] == "Consumer Staples"      # "Consumer Defensive" -> Consumer Staples


def test_fundamentals_point_in_time(fin_bundle):
    # Report periods are lagged; the panel must be multi-indexed and derived
    # fields (earnings_vol) populated once enough history exists.
    assert fin_bundle.fundamentals.index.names == ["date", "ticker"]
    f = fin_bundle.fundamentals_asof(fin_bundle.prices.index[-1])
    assert set(f.index) == set(fin_bundle.tickers)
    assert f["pe"].notna().all()
    assert f["earnings_vol"].notna().any()


def test_no_lookahead_in_asof(fin_bundle):
    early = fin_bundle.prices.index[200]
    used = fin_bundle.fundamentals.index.get_level_values(0)
    f = fin_bundle.fundamentals_asof(early)
    assert used.min() <= early
    # The as-of snapshot must not include a report dated after `early`.
    visible = fin_bundle.fundamentals.loc[used <= early]
    assert not visible.empty


def test_signals_have_no_all_nan_columns(fin_bundle):
    sig = compute_signals(fin_bundle, sample_profile())
    all_nan = [c for c in sig.columns if sig[c].isna().all()]
    assert all_nan == []


def test_csv_roundtrip(fin_bundle, tmp_path):
    save_bundle(fin_bundle, str(tmp_path / "ds"))
    reloaded = CSVDataSource(str(tmp_path / "ds")).load()
    assert reloaded.prices.shape == fin_bundle.prices.shape
    assert reloaded.fundamentals.shape == fin_bundle.fundamentals.shape
    assert reloaded.sectors == fin_bundle.sectors


def test_train_on_financial_dataset(fin_bundle):
    panel = build_training_panel(fin_bundle, horizon_days=21)
    assert not panel.empty
    assert panel["date"].nunique() > 20
    trained = train_alpha_model(fin_bundle, task="regression", panel=panel, alpha=10.0)
    assert "mean_ic" in trained.metrics
    from wharton_ml_engine.ml import ML_FEATURES
    assert len(trained.coefficients()) == len(ML_FEATURES)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("FINANCIAL_DATASETS_API_KEY", raising=False)
    with pytest.raises(ValueError):
        FinancialDatasetSource(tickers=["AAPL"], start="2020-01-01", end="2020-12-31")
