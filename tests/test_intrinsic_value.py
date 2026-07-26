"""Intrinsic value / absolute margin-of-safety (Graham + FCF earnings power)."""

import numpy as np
import pandas as pd

from wharton_ml_engine.signals.fundamental import fundamental_scores, intrinsic_value_model


def _fund(rows):
    return pd.DataFrame(rows).T


def test_graham_margin_of_safety_sign():
    f = _fund({
        "CHEAP":  {"pe": 9.0, "pb": 1.1, "fcf_yield": 0.09},   # PE*PB=9.9 < 22.5
        "FAIR":   {"pe": 18.0, "pb": 3.0, "fcf_yield": 0.04},  # PE*PB=54
        "PRICEY": {"pe": 45.0, "pb": 12.0, "fcf_yield": 0.01},
    })
    iv = intrinsic_value_model(f)
    assert iv.loc["CHEAP", "margin_of_safety_pct"] > 0        # below Graham fair value
    assert iv.loc["FAIR", "margin_of_safety_pct"] < 0
    assert iv.loc["PRICEY", "margin_of_safety_pct"] < iv.loc["FAIR", "margin_of_safety_pct"]


def test_graham_pass_screen():
    f = _fund({
        "PASS": {"pe": 12.0, "pb": 1.3, "fcf_yield": 0.06},   # PE<=15 and PB<=1.5
        "FAIL_PE": {"pe": 25.0, "pb": 1.2, "fcf_yield": 0.05},
        "FAIL_PB": {"pe": 10.0, "pb": 4.0, "fcf_yield": 0.05},
    })
    iv = intrinsic_value_model(f)
    assert bool(iv.loc["PASS", "graham_pass"]) is True
    assert bool(iv.loc["FAIL_PE", "graham_pass"]) is False
    assert bool(iv.loc["FAIL_PB", "graham_pass"]) is False


def test_intrinsic_score_ranks_cheaper_higher():
    f = _fund({
        "CHEAP":  {"pe": 8.0, "pb": 1.0, "fcf_yield": 0.10},
        "PRICEY": {"pe": 40.0, "pb": 10.0, "fcf_yield": 0.01},
    })
    iv = intrinsic_value_model(f)
    assert iv.loc["CHEAP", "intrinsic"] > iv.loc["PRICEY", "intrinsic"]


def test_handles_negative_and_missing_ratios():
    # Loss-making (PE<=0) or missing P/B must not crash; MoS is NaN, not garbage.
    f = _fund({
        "LOSS": {"pe": -5.0, "pb": 2.0, "fcf_yield": 0.02},
        "NOPB": {"pe": 12.0, "pb": np.nan, "fcf_yield": 0.03},
    })
    iv = intrinsic_value_model(f)
    assert np.isnan(iv.loc["LOSS", "graham_ratio"])
    assert bool(iv.loc["LOSS", "graham_pass"]) is False


def test_exposed_in_fundamental_scores():
    f = _fund({
        "A": {"pe": 10, "pb": 1.2, "ev_ebit": 6, "fcf_yield": 0.07, "accruals": 0.02,
              "roe": 0.15, "roic": 0.12, "gross_margin": 0.4, "earnings_vol": 0.2,
              "debt_equity": 0.5, "interest_coverage": 10, "revenue_growth": 0.05,
              "eps_growth": 0.06, "growth_stability": 0.6, "net_issuance": -0.01,
              "asset_growth": 0.03, "dividend_yield": 0.02, "payout_ratio": 0.3,
              "adv_usd": 5e7, "market_cap": 5e10},
        "B": {"pe": 30, "pb": 8, "ev_ebit": 22, "fcf_yield": 0.02, "accruals": 0.05,
              "roe": 0.25, "roic": 0.2, "gross_margin": 0.6, "earnings_vol": 0.3,
              "debt_equity": 1.2, "interest_coverage": 6, "revenue_growth": 0.25,
              "eps_growth": 0.3, "growth_stability": 0.5, "net_issuance": 0.02,
              "asset_growth": 0.2, "dividend_yield": 0.0, "payout_ratio": 0.0,
              "adv_usd": 5e7, "market_cap": 5e10},
    })
    fs = fundamental_scores(f)
    for col in ("intrinsic", "margin_of_safety_pct", "graham_pass"):
        assert col in fs.columns
    # The cheap, sound name carries a larger margin of safety than the pricey one.
    assert fs.loc["A", "margin_of_safety_pct"] > fs.loc["B", "margin_of_safety_pct"]
