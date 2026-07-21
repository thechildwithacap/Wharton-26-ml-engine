"""Granular tests for the sub-models and their sub-signals (sub-sub-models).

Each signal model is a composite of named sub-signals (e.g. the Value Model =
valuation + balance-sheet + cash-flow).  These tests pin down that every
sub-signal moves in the right direction in isolation, and that the composites
aggregate them correctly.
"""

import numpy as np
import pandas as pd
import pytest

from wharton_ml_engine.data.source import FUNDAMENTAL_FIELDS, DataBundle, SecurityMeta
from wharton_ml_engine.signals.fundamental import (
    growth_garp_model,
    income_model,
    quality_model,
    value_model,
)
from wharton_ml_engine.signals.hybrid import (
    analyst_overlay_model,
    hybrid_alpha_model,
    theme_tilt_model,
)
from wharton_ml_engine.signals.quant import macro_sensitivity_model, price_factor_model


# --- helpers -------------------------------------------------------------

_DEFAULTS = {
    "market_cap": 5e10, "pe": 18.0, "pb": 3.0, "ev_ebit": 14.0, "fcf_yield": 0.04,
    "accruals": 0.03, "roe": 0.15, "roic": 0.12, "gross_margin": 0.45,
    "earnings_vol": 0.2, "debt_equity": 0.8, "interest_coverage": 8.0,
    "revenue_growth": 0.08, "eps_growth": 0.10, "growth_stability": 0.6,
    "dividend_yield": 0.0, "payout_ratio": 0.0, "adv_usd": 5e7,
}


def make_fund(rows: dict) -> pd.DataFrame:
    """rows: {ticker: {field: value}} — unspecified fields take neutral defaults."""
    data = {}
    for t, over in rows.items():
        rec = dict(_DEFAULTS)
        rec.update(over)
        data[t] = rec
    return pd.DataFrame(data).T[FUNDAMENTAL_FIELDS].astype(float)


def _monotonic_increasing(s: pd.Series, order: list) -> bool:
    vals = [s[t] for t in order]
    return all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))


# =========================================================================
# VALUE MODEL sub-signals
# =========================================================================

def test_value_valuation_prefers_cheaper_pe():
    order = ["A", "B", "C", "D", "E"]                  # increasing P/E
    fund = make_fund({t: {"pe": pe} for t, pe in zip(order, [8, 14, 20, 28, 40])})
    v = value_model(fund)
    # Lower P/E -> higher valuation sub-score (order is worst->best reversed).
    assert _monotonic_increasing(v["valuation"], list(reversed(order)))


def test_value_balance_sheet_prefers_low_leverage():
    order = ["A", "B", "C", "D"]
    fund = make_fund({t: {"debt_equity": de, "interest_coverage": ic}
                      for t, de, ic in zip(order, [0.2, 0.8, 1.5, 2.5], [20, 10, 5, 2])})
    v = value_model(fund)
    assert _monotonic_increasing(v["balance_sheet"], list(reversed(order)))


def test_value_cash_flow_prefers_high_fcf_yield():
    order = ["A", "B", "C", "D"]
    fund = make_fund({t: {"fcf_yield": y} for t, y in zip(order, [0.01, 0.03, 0.06, 0.10])})
    v = value_model(fund)
    assert _monotonic_increasing(v["cash_flow"], order)


def test_margin_of_safety_flag():
    fund = make_fund({
        "CHEAP_SOUND": {"pe": 6, "pb": 0.8, "ev_ebit": 5, "debt_equity": 0.2,
                        "interest_coverage": 20},
        "CHEAP_WEAK": {"pe": 6, "pb": 0.8, "ev_ebit": 5, "debt_equity": 2.8,
                       "interest_coverage": 1.5},
        "EXPENSIVE": {"pe": 40, "pb": 12, "ev_ebit": 30},
    })
    v = value_model(fund)
    assert bool(v.loc["CHEAP_SOUND", "margin_of_safety"]) is True
    assert bool(v.loc["EXPENSIVE", "margin_of_safety"]) is False


# =========================================================================
# QUALITY MODEL sub-signals
# =========================================================================

def test_quality_profitability_rises_with_returns():
    order = ["A", "B", "C", "D"]
    fund = make_fund({t: {"roe": r, "roic": r * 0.8, "gross_margin": gm}
                      for t, r, gm in zip(order, [0.05, 0.12, 0.22, 0.35],
                                          [0.25, 0.4, 0.55, 0.7])})
    q = quality_model(fund)
    assert _monotonic_increasing(q["profitability"], order)


def test_quality_stability_prefers_low_earnings_vol():
    order = ["A", "B", "C", "D"]
    fund = make_fund({t: {"earnings_vol": ev, "growth_stability": gs}
                      for t, ev, gs in zip(order, [0.05, 0.15, 0.30, 0.55],
                                           [0.9, 0.7, 0.5, 0.3])})
    q = quality_model(fund)
    # Lower earnings vol (A) -> higher stability.
    assert _monotonic_increasing(q["stability"], list(reversed(order)))


# =========================================================================
# GROWTH / GARP sub-signals
# =========================================================================

def test_growth_rises_with_growth_rates():
    order = ["A", "B", "C", "D"]
    fund = make_fund({t: {"revenue_growth": g, "eps_growth": g + 0.02}
                      for t, g in zip(order, [-0.05, 0.05, 0.20, 0.40])})
    gg = growth_garp_model(fund)
    assert _monotonic_increasing(gg["growth"], order)


def test_garp_rewards_cheap_growth():
    # Same growth, different P/E: cheaper name should have higher GARP.
    fund = make_fund({
        "CHEAP": {"pe": 10, "eps_growth": 0.20, "revenue_growth": 0.18},
        "MID": {"pe": 20, "eps_growth": 0.20, "revenue_growth": 0.18},
        "PRICEY": {"pe": 40, "eps_growth": 0.20, "revenue_growth": 0.18},
    })
    gg = growth_garp_model(fund)
    assert gg.loc["CHEAP", "garp"] > gg.loc["MID", "garp"] > gg.loc["PRICEY", "garp"]


# =========================================================================
# INCOME MODEL sub-signals
# =========================================================================

def test_income_rewards_yield_among_payers():
    fund = make_fund({
        "LOW": {"dividend_yield": 0.01, "payout_ratio": 0.3, "fcf_yield": 0.05},
        "MID": {"dividend_yield": 0.03, "payout_ratio": 0.3, "fcf_yield": 0.05},
        "HIGH": {"dividend_yield": 0.05, "payout_ratio": 0.3, "fcf_yield": 0.05},
    })
    inc = income_model(fund)
    assert inc.loc["HIGH", "income"] > inc.loc["MID", "income"] > inc.loc["LOW", "income"]


def test_income_penalises_unsustainable_payout():
    fund = make_fund({
        "SUSTAIN": {"dividend_yield": 0.04, "payout_ratio": 0.4, "fcf_yield": 0.06},
        "STRETCHED": {"dividend_yield": 0.04, "payout_ratio": 0.95, "fcf_yield": 0.06},
        "OTHER": {"dividend_yield": 0.02, "payout_ratio": 0.3, "fcf_yield": 0.06},
    })
    inc = income_model(fund)
    assert inc.loc["SUSTAIN", "income"] > inc.loc["STRETCHED", "income"]


def test_non_payer_gets_low_income():
    fund = make_fund({
        "PAYER": {"dividend_yield": 0.04, "payout_ratio": 0.4, "fcf_yield": 0.05},
        "NONPAYER": {"dividend_yield": 0.0, "payout_ratio": 0.0},
        "OTHER": {"dividend_yield": 0.02, "payout_ratio": 0.3, "fcf_yield": 0.05},
    })
    inc = income_model(fund)
    assert inc.loc["NONPAYER", "income"] < inc.loc["PAYER", "income"]


# =========================================================================
# ANALYST / HYBRID / THEME (needs a tiny bundle)
# =========================================================================

def _tiny_bundle() -> DataBundle:
    dates = pd.bdate_range("2021-01-01", periods=320)
    n = len(dates)
    rng = np.random.default_rng(3)
    mkt = rng.normal(0.0004, 0.01, n)
    series = {
        "MOM": 100 * np.exp(np.cumsum(np.full(n, 0.0015) + 0.2 * mkt)),   # steady up
        "DOWN": 100 * np.exp(np.cumsum(np.full(n, -0.0015) + 0.2 * mkt)), # steady down
        "FLAT": 100 * np.exp(np.cumsum(0.2 * mkt + rng.normal(0, 0.003, n))),
        "VOLA": 100 * np.exp(np.cumsum(0.2 * mkt + rng.normal(0, 0.03, n))),
        "HIBETA": 100 * np.exp(np.cumsum(2.0 * mkt)),
        "LOBETA": 100 * np.exp(np.cumsum(0.2 * mkt)),
    }
    prices = pd.DataFrame(series, index=dates)
    benchmarks = pd.DataFrame({"SPX": 4000 * np.exp(np.cumsum(mkt))}, index=dates)

    caps = {"MOM": 1e11, "DOWN": 1e11, "FLAT": 1e11, "VOLA": 1e11,
            "HIBETA": 5e9, "LOBETA": 9e11}      # SMALL vs BIG for size test
    idx = pd.MultiIndex.from_product([[dates[-1]], list(series)], names=["date", "ticker"])
    frows = []
    for t in series:
        rec = dict(_DEFAULTS)
        rec["market_cap"] = caps[t]
        frows.append(rec)
    fundamentals = pd.DataFrame(frows, index=idx)[FUNDAMENTAL_FIELDS].astype(float)

    meta = {t: SecurityMeta(t, f"{t} Inc", "Technology",
                            themes=(["AI"] if t == "MOM" else []),
                            pays_dividend=False) for t in series}
    return DataBundle(prices=prices, benchmarks=benchmarks,
                      fundamentals=fundamentals, meta=meta)


def test_price_factor_momentum_and_lowvol_and_size():
    b = _tiny_bundle()
    pf = price_factor_model(b, b.prices.index[-1])
    assert pf.loc["MOM", "momentum"] > pf.loc["DOWN", "momentum"]
    assert pf.loc["FLAT", "low_vol"] > pf.loc["VOLA", "low_vol"]
    assert pf.loc["HIBETA", "size"] > pf.loc["LOBETA", "size"]     # smaller cap -> higher


def test_macro_tilt_prefers_low_beta():
    b = _tiny_bundle()
    macro = macro_sensitivity_model(b, b.prices.index[-1])
    assert macro.loc["HIBETA", "beta"] > macro.loc["LOBETA", "beta"]
    assert macro.loc["LOBETA", "macro_tilt"] > macro.loc["HIBETA", "macro_tilt"]


def test_analyst_overlay_uses_supplied_ratings():
    b = _tiny_bundle()
    ratings = {"MOM": {"moat": 5, "management": 5, "industry_structure": 5},
               "DOWN": {"moat": 1, "management": 1, "industry_structure": 1}}
    a = analyst_overlay_model(b, b.prices.index[-1], ratings=ratings)
    assert a["MOM"] > 80 and a["DOWN"] < 20


def test_analyst_overlay_robust_to_missing_accruals():
    fund = make_fund({"A": {}, "B": {"roic": 0.3}, "C": {"roic": 0.05}})
    fund["accruals"] = np.nan                       # data source lacks accruals
    idx = pd.MultiIndex.from_product([[pd.Timestamp("2024-01-31")], fund.index],
                                     names=["date", "ticker"])
    fund.index = pd.Index(list(fund.index), name="ticker")
    bundle = DataBundle(
        prices=pd.DataFrame({t: [100, 101] for t in fund.index},
                            index=pd.bdate_range("2024-01-01", periods=2)),
        benchmarks=pd.DataFrame({"SPX": [4000, 4010]},
                                index=pd.bdate_range("2024-01-01", periods=2)),
        fundamentals=fund.set_index(idx),
        meta={t: SecurityMeta(t, t, "Technology") for t in fund.index},
    )
    a = analyst_overlay_model(bundle, bundle.prices.index[-1])
    assert a.notna().all()                          # no NaN despite missing accruals


def test_hybrid_alpha_rewards_agreement():
    idx = ["AGREE", "DISAGREE"]
    style = pd.DataFrame({"value": [80, 80], "quality": [80, 80], "growth": [80, 80]},
                         index=idx)
    quant = pd.DataFrame({"factor": [80, 20]}, index=idx)
    analyst = pd.Series({"AGREE": 80, "DISAGREE": 20})
    h = hybrid_alpha_model(style, analyst, quant)
    assert h["AGREE"] > h["DISAGREE"]


def test_theme_fit_excludes_and_rewards():
    b = _tiny_bundle()
    from wharton_ml_engine.client import sample_profile
    prof = sample_profile()
    prof.excluded_themes = ["AI"]
    tf = theme_tilt_model(b, prof, b.prices.index[-1])
    assert tf["MOM"] < 30           # MOM carries the excluded AI theme
    assert tf["FLAT"] == 50         # no theme -> neutral
