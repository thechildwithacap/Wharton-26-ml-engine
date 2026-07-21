"""Integrated Score Calculation (PRD 7.2).

Combines the style scores using the regime-dependent style weights, layers in
conviction (hybrid alpha) and theme fit, then applies penalties for risk,
liquidity, crowding and weak client fit.  Models with low robustness are
downweighted.  Output is a single ranked, ticker-indexed table.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..client.profile import ClientProfile
from ..config import STYLES
from ..utils import clip_score, weighted_blend


def _robust_style_weights(style_weights: Dict[str, float],
                          robustness: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not robustness:
        return dict(style_weights)
    adj = {s: style_weights.get(s, 0.0) * (0.5 + 0.5 * robustness.get(s, 50.0) / 100.0)
           for s in style_weights}
    total = sum(adj.values())
    return {s: (v / total if total else 0.0) for s, v in adj.items()}


def integrate_scores(
    signals: pd.DataFrame,
    client_fit: pd.DataFrame,
    stock_risk: pd.DataFrame,
    liquidity: pd.DataFrame,
    style_weights: Dict[str, float],
    profile: ClientProfile,
    robustness: Optional[Dict[str, float]] = None,
    de_risk_styles: Optional[List[str]] = None,
) -> pd.DataFrame:
    tickers = signals.index
    out = pd.DataFrame(index=tickers)

    # 1. Core style score under (robustness-adjusted) regime style weights.
    eff_weights = _robust_style_weights(style_weights, robustness)
    if de_risk_styles:                       # crowding: fade the crowded styles
        for s in de_risk_styles:
            eff_weights[s] = eff_weights.get(s, 0.0) * 0.5
        tot = sum(eff_weights.values())
        eff_weights = {s: (v / tot if tot else 0.0) for s, v in eff_weights.items()}

    style_components = {s: signals[s] for s in STYLES if s in signals.columns}
    core = weighted_blend(style_components, eff_weights)
    out["style_score"] = clip_score(core)

    # 2. Blend in conviction (hybrid alpha), the learned ML alpha (if present)
    #    and theme fit.  When a trained model is supplied, it earns a share of
    #    the blend at the expense of the hand-built core.
    blend_inputs = {
        "core": out["style_score"],
        "hybrid_alpha": signals.get("hybrid_alpha", pd.Series(50.0, index=tickers)),
        "theme_fit": signals.get("theme_fit", pd.Series(50.0, index=tickers)),
    }
    if "ml_alpha" in signals.columns:
        blend_inputs["ml_alpha"] = signals["ml_alpha"]
        blend_weights = {"core": 0.52, "ml_alpha": 0.22, "hybrid_alpha": 0.18,
                         "theme_fit": 0.08}
    else:
        blend_weights = {"core": 0.68, "hybrid_alpha": 0.24, "theme_fit": 0.08}
    integrated = weighted_blend(blend_inputs, blend_weights)

    # 3. Client-fit scaling: poor fit shrinks the score toward zero.
    fit = client_fit["client_fit"].reindex(tickers).fillna(50.0)
    fit_factor = 0.5 + 0.5 * (fit / 100.0)          # fit 0 -> x0.5, fit 100 -> x1.0
    integrated = integrated * fit_factor

    # 4. Risk & liquidity penalties (risk aversion scales with the mandate).
    risk_score = stock_risk["risk_score"].reindex(tickers).fillna(50.0)
    aversion = {1: 0.35, 2: 0.28, 3: 0.20, 4: 0.12, 5: 0.06}[profile.risk_tolerance]
    risk_penalty = aversion * (50.0 - risk_score) / 50.0 * 100.0   # penalise risky names
    integrated = integrated - risk_penalty.clip(lower=-15, upper=40)

    liq_score = liquidity["liquidity_score"].reindex(tickers).fillna(50.0)
    liq_penalty = 0.10 * (50.0 - liq_score).clip(lower=0) / 50.0 * 100.0
    integrated = integrated - liq_penalty

    # 5. Tail-risk and illiquidity flags.
    tail = stock_risk["tail_risk_flag"].reindex(tickers).fillna(False)
    integrated = integrated - tail.astype(float) * 6.0

    out["client_fit"] = fit
    out["risk_score"] = risk_score
    out["liquidity_score"] = liq_score
    out["integrated_score"] = clip_score(integrated)

    # 6. Eligibility: exclusions and hard operational vetoes.
    excluded = client_fit["hard_excluded"].reindex(tickers).fillna(False)
    illiquid = liquidity["illiquid_flag"].reindex(tickers).fillna(False)
    out["eligible"] = (~excluded) & (~illiquid)
    out.loc[~out["eligible"], "integrated_score"] = np.nan

    reason = pd.Series("", index=tickers)
    reason[excluded] = client_fit["exclude_reason"].reindex(tickers).fillna("")[excluded]
    reason[illiquid & (reason == "")] = "insufficient liquidity"
    out["ineligible_reason"] = reason

    return out.sort_values("integrated_score", ascending=False)
