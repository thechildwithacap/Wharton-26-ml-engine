"""Hybrid & thematic pod (PRD 6.1.3).

* Analyst Overlay — the team's qualitative ratings (moat, management, industry
  structure).  Real ratings are supplied as a dict; when absent we synthesise a
  deterministic placeholder from fundamentals so the pipeline still runs.
* Hybrid Alpha — blends the analyst view with the quant/fundamental scores and
  is *highest when the two agree* (agreement bonus).
* Theme & Sector Tilt — tags each stock to themes and scores alignment with the
  client's preferences.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..client.profile import ClientProfile
from ..data.source import DataBundle
from ..utils import clip_score, score_higher_is_better, weighted_blend


# Team ratings are 1-5 per axis; converted to 0-100.
ANALYST_AXES = ["moat", "management", "industry_structure"]


def analyst_overlay_model(
    bundle: DataBundle,
    as_of: pd.Timestamp,
    ratings: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.Series:
    """Return an Analyst Score (0-100) per ticker.

    ``ratings`` maps ``ticker -> {axis: 1..5}``.  Missing tickers/axes fall
    back to a fundamentals-derived proxy so the model degrades gracefully.
    """
    fund = bundle.fundamentals_asof(as_of)
    tickers = list(fund.index)
    ratings = ratings or {}

    # Proxy from fundamentals: a wide moat looks like high ROIC + margins;
    # good management like low accruals + steady growth; industry structure
    # like high gross margin.  These are placeholders, clearly marked as such.
    proxy_moat = score_higher_is_better(fund["roic"])
    proxy_mgmt = 0.5 * score_higher_is_better(fund["growth_stability"]) + \
        0.5 * clip_score(100 - score_higher_is_better(fund["accruals"]))
    proxy_industry = score_higher_is_better(fund["gross_margin"])
    proxy = {"moat": proxy_moat, "management": proxy_mgmt,
             "industry_structure": proxy_industry}

    scores = {}
    for t in tickers:
        vals = []
        for axis in ANALYST_AXES:
            r = ratings.get(t, {}).get(axis)
            if r is not None:
                vals.append((float(r) - 1.0) / 4.0 * 100.0)   # 1..5 -> 0..100
            else:
                vals.append(float(proxy[axis].get(t, 50.0)))
        scores[t] = float(np.mean(vals))
    return pd.Series(scores, name="analyst").reindex(tickers)


def hybrid_alpha_model(
    style_scores: pd.DataFrame,
    analyst: pd.Series,
    quant: pd.DataFrame,
) -> pd.Series:
    """Blend fundamental + quant + analyst, rewarding agreement.

    Base = weighted blend of the core views.  Agreement bonus = how tightly the
    fundamental composite and the quant/analyst composite line up (both high or
    both low), so conviction is highest when independent lenses concur.
    """
    fundamental_composite = weighted_blend(
        {
            "value": style_scores["value"],
            "quality": style_scores["quality"],
            "growth": style_scores["growth"],
        },
        {"value": 0.34, "quality": 0.4, "growth": 0.26},
    )
    quant_composite = weighted_blend(
        {"factor": quant["factor"], "analyst": analyst},
        {"factor": 0.5, "analyst": 0.5},
    )

    base = weighted_blend(
        {"fundamental": fundamental_composite, "quant": quant_composite},
        {"fundamental": 0.6, "quant": 0.4},
    )

    # Agreement: 100 when both composites sit on the same side of 50 and close
    # together; drops when they disagree.
    disagreement = (fundamental_composite - quant_composite).abs()
    agreement = (100.0 - disagreement).clip(lower=0, upper=100)
    both_high = ((fundamental_composite > 55) & (quant_composite > 55)).astype(float)

    hybrid = base + 0.10 * (agreement - 50.0) + 4.0 * both_high
    return clip_score(hybrid).rename("hybrid_alpha")


def theme_tilt_model(
    bundle: DataBundle,
    profile: ClientProfile,
    as_of: pd.Timestamp,
) -> pd.Series:
    """Theme Fit: alignment of a stock's themes with client preferences."""
    fund = bundle.fundamentals_asof(as_of)
    tickers = list(fund.index)
    excluded = set(profile.excluded_themes)

    scores = {}
    for t in tickers:
        themes = set(bundle.meta[t].themes)
        if themes & excluded:
            scores[t] = 20.0
        elif themes:
            # Themed names get a mild positive tilt; more themes -> a bit higher.
            scores[t] = min(50.0 + 12.0 * len(themes), 85.0)
        else:
            scores[t] = 50.0
    return pd.Series(scores, name="theme_fit").reindex(tickers)
