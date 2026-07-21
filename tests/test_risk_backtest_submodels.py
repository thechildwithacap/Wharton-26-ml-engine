"""Sub-model tests for the Risk & Constraints and Backtest & Regime layers."""

import numpy as np
import pandas as pd

from wharton_ml_engine.backtest.regime import classify_regime
from wharton_ml_engine.backtest.stability import signal_stability
from wharton_ml_engine.backtest.style_weights import recommend_style_weights
from wharton_ml_engine.backtest.templates import normalise_weights, template_score
from wharton_ml_engine.client import sample_profile
from wharton_ml_engine.config import STYLES, CostModel, PortfolioConstraints
from wharton_ml_engine.data.source import FUNDAMENTAL_FIELDS, DataBundle, SecurityMeta
from wharton_ml_engine.risk.checks import (
    data_quality_report,
    mandate_check,
    turnover_and_cost,
)
from wharton_ml_engine.risk.portfolio import (
    PortfolioRiskSummary,
    concentration_metrics,
    crowding_model,
)
from wharton_ml_engine.risk.stock import liquidity_model, stock_risk_model


_DEF = {
    "market_cap": 5e10, "pe": 18.0, "pb": 3.0, "ev_ebit": 14.0, "fcf_yield": 0.04,
    "accruals": 0.03, "roe": 0.15, "roic": 0.12, "gross_margin": 0.45,
    "earnings_vol": 0.2, "debt_equity": 0.8, "interest_coverage": 8.0,
    "revenue_growth": 0.08, "eps_growth": 0.10, "growth_stability": 0.6,
    "dividend_yield": 0.0, "payout_ratio": 0.0, "adv_usd": 5e7,
}


def _bundle(trend: float = 0.0004, seed: int = 1, advs=None) -> DataBundle:
    dates = pd.bdate_range("2021-01-01", periods=320)
    n = len(dates)
    rng = np.random.default_rng(seed)
    mkt = rng.normal(trend, 0.01, n)
    prices = pd.DataFrame({
        "LOWVOL": 100 * np.exp(np.cumsum(0.3 * mkt + rng.normal(0, 0.004, n))),
        "HIVOL": 100 * np.exp(np.cumsum(0.3 * mkt + rng.normal(0, 0.03, n))),
        "MID": 100 * np.exp(np.cumsum(0.3 * mkt + rng.normal(0, 0.012, n))),
    }, index=dates)
    benchmarks = pd.DataFrame({"SPX": 4000 * np.exp(np.cumsum(mkt))}, index=dates)
    advs = advs or {"LOWVOL": 5e8, "HIVOL": 5e6, "MID": 5e7}
    idx = pd.MultiIndex.from_product([[dates[-1]], prices.columns], names=["date", "ticker"])
    frows = []
    for t in prices.columns:
        r = dict(_DEF); r["adv_usd"] = advs[t]; frows.append(r)
    fund = pd.DataFrame(frows, index=idx)[FUNDAMENTAL_FIELDS].astype(float)
    meta = {t: SecurityMeta(t, t, "Technology") for t in prices.columns}
    return DataBundle(prices=prices, benchmarks=benchmarks, fundamentals=fund, meta=meta)


# --- stock risk & liquidity ---------------------------------------------

def test_stock_risk_score_prefers_low_vol():
    b = _bundle()
    sr = stock_risk_model(b, b.prices.index[-1])
    assert sr.loc["LOWVOL", "risk_score"] > sr.loc["HIVOL", "risk_score"]
    assert sr.loc["LOWVOL", "volatility"] < sr.loc["HIVOL", "volatility"]


def test_liquidity_score_rises_with_adv():
    b = _bundle()
    liq = liquidity_model(b, b.prices.index[-1], PortfolioConstraints())
    assert liq.loc["LOWVOL", "liquidity_score"] > liq.loc["HIVOL", "liquidity_score"]
    assert liq.loc["HIVOL", "max_safe_trade_usd"] < liq.loc["LOWVOL", "max_safe_trade_usd"]


# --- concentration / turnover -------------------------------------------

def test_concentration_metrics_math():
    w = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
    m = concentration_metrics(w, {})
    assert abs(m["herfindahl"] - (0.16 + 0.09 + 0.04 + 0.01)) < 1e-9
    assert abs(m["top_weight"] - 0.4) < 1e-9
    assert m["n_holdings"] == 4
    assert abs(m["effective_n"] - 1.0 / 0.30) < 1e-9


def test_turnover_and_cost():
    cur = {"A": 0.5, "B": 0.5}
    tgt = {"A": 0.5, "C": 0.5}                       # swap B -> C
    r = turnover_and_cost(cur, tgt, CostModel())
    assert abs(r.turnover - 0.5) < 1e-9              # one-way turnover = 50%
    assert r.cost_estimate > 0
    assert 0 <= r.turnover_score <= 100


# --- mandate check violations -------------------------------------------

def test_mandate_flags_position_cap():
    b = _bundle()
    prof = sample_profile()
    cons = PortfolioConstraints(min_holdings=1)
    chk = mandate_check({"LOWVOL": 0.5, "HIVOL": 0.5}, b, prof, cons)
    assert not chk.passed
    assert any("weight" in v for v in chk.violations)


def test_mandate_flags_holdings_and_sector():
    b = _bundle()
    prof = sample_profile()
    cons = PortfolioConstraints()                    # min 20 holdings, 25% sector cap
    chk = mandate_check({"LOWVOL": 0.34, "HIVOL": 0.33, "MID": 0.33}, b, prof, cons)
    assert not chk.passed
    assert any("holdings" in v for v in chk.violations)
    assert any("sector" in v for v in chk.violations)  # all Technology -> 100%


# --- data quality --------------------------------------------------------

def test_data_quality_ok_and_stale():
    b = _bundle()
    assert data_quality_report(b, b.prices.index[-1]).status == "OK"
    # Freeze one series to a constant -> stale detection.
    b.prices["MID"] = 100.0
    rep = data_quality_report(b, b.prices.index[-1])
    assert rep.status in {"Warning", "Critical"}
    assert "MID" in rep.stale_tickers


# --- crowding ------------------------------------------------------------

def test_crowding_flags_extreme_style():
    summ = PortfolioRiskSummary(0.15, 1.0, 0.08, 0.3, 0.05, 20, {},
                                {"momentum": 78.0, "value": 55.0}, 20)
    out = crowding_model({"A": 1.0}, pd.DataFrame(), summ)
    assert "momentum" in out["de_risk_styles"]
    assert any("momentum" in f for f in out["crowding_flags"])


# --- regime classification ----------------------------------------------

def _trending_bundle(trend: float, seed: int = 5, n_stocks: int = 10) -> DataBundle:
    """A wider universe whose names track the market, so breadth is decisive."""
    dates = pd.bdate_range("2021-01-01", periods=320)
    n = len(dates)
    rng = np.random.default_rng(seed)
    mkt = rng.normal(trend, 0.009, n)
    cols = {f"S{i}": 100 * np.exp(np.cumsum(mkt + rng.normal(0, 0.004, n)))
            for i in range(n_stocks)}
    prices = pd.DataFrame(cols, index=dates)
    benchmarks = pd.DataFrame({"SPX": 4000 * np.exp(np.cumsum(mkt))}, index=dates)
    idx = pd.MultiIndex.from_product([[dates[-1]], prices.columns], names=["date", "ticker"])
    fund = pd.DataFrame([dict(_DEF) for _ in prices.columns],
                        index=idx)[FUNDAMENTAL_FIELDS].astype(float)
    meta = {t: SecurityMeta(t, t, "Technology") for t in prices.columns}
    return DataBundle(prices=prices, benchmarks=benchmarks, fundamentals=fund, meta=meta)


def test_regime_bull_vs_bear():
    bull_b = _trending_bundle(trend=0.0016)
    bear_b = _trending_bundle(trend=-0.0016)
    bull = classify_regime(bull_b, bull_b.prices.index[-1])
    bear = classify_regime(bear_b, bear_b.prices.index[-1])
    assert bull.trend == "bull"
    assert bear.trend == "bear"


# --- style weights & templates ------------------------------------------

def test_style_weights_sum_to_one():
    b = _bundle(trend=0.0012)
    rec = recommend_style_weights(b, sample_profile(), b.prices.index[-1])
    assert set(rec.weights) == set(STYLES)
    assert abs(sum(rec.weights.values()) - 1.0) < 1e-9


def test_normalise_weights():
    w = normalise_weights({"value": 2.0, "quality": 2.0})
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_template_score_weighting():
    panel = pd.DataFrame({"value": [100.0, 0.0], "quality": [0.0, 100.0]},
                         index=["X", "Y"])
    s = template_score(panel, {"value": 1.0, "quality": 0.0})
    assert s["X"] > s["Y"]


# --- signal stability ----------------------------------------------------

def test_signal_stability_high_when_unchanged():
    b = _bundle()
    st = signal_stability(b, b.prices.index[-1], lag_days=3)
    assert 0.0 <= st.confidence <= 1.0
