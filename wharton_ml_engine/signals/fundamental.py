"""Fundamental style pod: Value, Quality, Growth/GARP, Income (PRD 6.1.1).

Every function takes an as-of fundamentals frame (ticker-indexed, produced by
``DataBundle.fundamentals_asof``) and returns 0-100 scores plus, where the PRD
calls for them, flags.  All scoring is cross-sectional ranking so the outputs
are comparable across styles and dates.
"""

from __future__ import annotations

import pandas as pd

from ..utils import (
    clip_score,
    score_higher_is_better,
    score_lower_is_better,
    weighted_blend,
)


def value_model(fund: pd.DataFrame) -> pd.DataFrame:
    """Value Score with a margin-of-safety flag."""
    valuation = weighted_blend(
        {
            "pe": score_lower_is_better(fund["pe"]),
            "pb": score_lower_is_better(fund["pb"]),
            "ev_ebit": score_lower_is_better(fund["ev_ebit"]),
        },
        {"pe": 0.4, "pb": 0.25, "ev_ebit": 0.35},
    )
    balance_sheet = weighted_blend(
        {
            "debt_equity": score_lower_is_better(fund["debt_equity"]),
            "interest_coverage": score_higher_is_better(fund["interest_coverage"]),
        },
        {"debt_equity": 0.5, "interest_coverage": 0.5},
    )
    cash_flow = weighted_blend(
        {
            "fcf_yield": score_higher_is_better(fund["fcf_yield"]),
            "accruals": score_lower_is_better(fund["accruals"]),
        },
        {"fcf_yield": 0.65, "accruals": 0.35},
    )
    value = weighted_blend(
        {"valuation": valuation, "balance_sheet": balance_sheet, "cash_flow": cash_flow},
        {"valuation": 0.55, "balance_sheet": 0.2, "cash_flow": 0.25},
    )
    out = pd.DataFrame(index=fund.index)
    out["value"] = clip_score(value)
    # Margin of safety: cheap AND financially sound (not a value trap).
    out["margin_of_safety"] = (valuation >= 70.0) & (balance_sheet >= 50.0)
    return out


def quality_model(fund: pd.DataFrame) -> pd.Series:
    profitability = weighted_blend(
        {
            "roe": score_higher_is_better(fund["roe"]),
            "roic": score_higher_is_better(fund["roic"]),
            "gross_margin": score_higher_is_better(fund["gross_margin"]),
        },
        {"roe": 0.35, "roic": 0.4, "gross_margin": 0.25},
    )
    stability = weighted_blend(
        {
            "earnings_vol": score_lower_is_better(fund["earnings_vol"]),
            "growth_stability": score_higher_is_better(fund["growth_stability"]),
        },
        {"earnings_vol": 0.5, "growth_stability": 0.5},
    )
    quality = weighted_blend(
        {"profitability": profitability, "stability": stability},
        {"profitability": 0.6, "stability": 0.4},
    )
    return clip_score(quality).rename("quality")


def growth_garp_model(fund: pd.DataFrame, value_score: pd.Series) -> pd.DataFrame:
    growth = weighted_blend(
        {
            "revenue_growth": score_higher_is_better(fund["revenue_growth"]),
            "eps_growth": score_higher_is_better(fund["eps_growth"]),
            "growth_stability": score_higher_is_better(fund["growth_stability"]),
        },
        {"revenue_growth": 0.4, "eps_growth": 0.4, "growth_stability": 0.2},
    )
    # GARP: growth at a reasonable price.  A PEG-style ratio = P/E divided by
    # EPS growth (%); lower is better, and we only reward genuine growers.
    peg = fund["pe"] / (fund["eps_growth"].clip(lower=0.01) * 100.0)
    garp_cheap = score_lower_is_better(peg)
    garp = weighted_blend(
        {"garp_cheap": garp_cheap, "growth": growth},
        {"garp_cheap": 0.6, "growth": 0.4},
    )
    # Growers with collapsing valuations (very low value_score) get a haircut.
    out = pd.DataFrame(index=fund.index)
    out["growth"] = clip_score(growth)
    out["garp"] = clip_score(garp)
    return out


def income_model(fund: pd.DataFrame) -> pd.Series:
    payers = fund["dividend_yield"] > 0
    yield_score = score_higher_is_better(fund["dividend_yield"].where(payers))
    # Sustainability: a moderate payout ratio is best (too high is risky, too
    # low means little income); reward payout coverage by FCF.
    payout = fund["payout_ratio"].where(payers)
    payout_penalty = (payout - 0.6).clip(lower=0) * 60.0     # penalise > 60% payout
    coverage = score_higher_is_better(fund["fcf_yield"].where(payers))
    income = weighted_blend(
        {"yield": yield_score, "coverage": coverage},
        {"yield": 0.7, "coverage": 0.3},
    )
    income = (income - payout_penalty).where(payers)
    # Non-payers get a low-but-nonzero income score (they may still be great
    # stocks on other styles).
    income = income.fillna(15.0)
    return clip_score(income).rename("income")


def fundamental_scores(fund: pd.DataFrame) -> pd.DataFrame:
    """Assemble all fundamental style scores into one frame."""
    val = value_model(fund)
    qual = quality_model(fund)
    gg = growth_garp_model(fund, val["value"])
    inc = income_model(fund)

    out = pd.DataFrame(index=fund.index)
    out["value"] = val["value"]
    out["margin_of_safety"] = val["margin_of_safety"]
    out["quality"] = qual
    out["growth"] = gg["growth"]
    out["garp"] = gg["garp"]
    out["income"] = inc
    return out
