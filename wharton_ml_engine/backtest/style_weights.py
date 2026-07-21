"""Style Weight Recommendation Model (PRD 6.3).

Turns the current regime (and, optionally, how the strategy templates have
performed) into a set of recommended style weights, then tilts them toward the
client's stated preferences.  These weights feed the integration layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from ..client.profile import ClientProfile
from ..config import STYLES
from ..data.source import DataBundle
from .engine import BacktestResult
from .regime import RegimeState, classify_regime
from .templates import STRATEGY_TEMPLATES, normalise_weights


# Regime-conditional style priors (PRD 9: "in regimes like this, quality +
# low-vol outperformed, so we tilted accordingly").
_REGIME_PRIORS: Dict[str, Dict[str, float]] = {
    "bull_low_vol":   {"momentum": 0.25, "growth": 0.22, "quality": 0.2,
                       "garp": 0.15, "value": 0.1, "low_vol": 0.04, "income": 0.04},
    "bull_high_vol":  {"quality": 0.28, "momentum": 0.2, "growth": 0.15,
                       "garp": 0.12, "low_vol": 0.13, "value": 0.08, "income": 0.04},
    "bear_low_vol":   {"quality": 0.3, "low_vol": 0.25, "value": 0.18,
                       "income": 0.15, "garp": 0.07, "growth": 0.03, "momentum": 0.02},
    "bear_high_vol":  {"low_vol": 0.32, "quality": 0.3, "income": 0.18,
                       "value": 0.12, "garp": 0.04, "growth": 0.02, "momentum": 0.02},
    "sideways_low_vol":{"quality": 0.24, "value": 0.22, "garp": 0.18,
                        "low_vol": 0.14, "income": 0.1, "momentum": 0.08, "growth": 0.04},
    "sideways_high_vol":{"quality": 0.28, "low_vol": 0.22, "value": 0.2,
                         "income": 0.14, "garp": 0.08, "momentum": 0.04, "growth": 0.04},
}


@dataclass
class StyleRecommendation:
    weights: Dict[str, float]
    regime: RegimeState
    rationale: str = ""
    best_template: Optional[str] = None
    template_blend: Dict[str, float] = field(default_factory=dict)


def _prior_for(regime: RegimeState) -> Dict[str, float]:
    if regime.label in _REGIME_PRIORS:
        return dict(_REGIME_PRIORS[regime.label])
    # Fall back on trend-only default.
    fallback = {
        "bull": "bull_low_vol", "bear": "bear_high_vol", "sideways": "sideways_low_vol",
    }.get(regime.trend, "sideways_low_vol")
    return dict(_REGIME_PRIORS[fallback])


def recommend_style_weights(
    bundle: DataBundle,
    profile: ClientProfile,
    as_of: pd.Timestamp,
    template_results: Optional[Dict[str, BacktestResult]] = None,
    template_blend_weight: float = 0.35,
) -> StyleRecommendation:
    regime = classify_regime(bundle, as_of)
    prior = _prior_for(regime)

    best_template = None
    blended = dict(prior)
    if template_results:
        # Pick the template with the best Sharpe over the backtest window and
        # blend its style weights into the regime prior.
        ranked = sorted(
            template_results.items(),
            key=lambda kv: (kv[1].metrics.get("sharpe") or float("-inf")),
            reverse=True,
        )
        best_template = ranked[0][0]
        tmpl_w = normalise_weights(STRATEGY_TEMPLATES[best_template])
        blended = {
            s: (1 - template_blend_weight) * prior.get(s, 0.0)
            + template_blend_weight * tmpl_w.get(s, 0.0)
            for s in STYLES
        }

    # Tilt toward client preferences and renormalise.
    tilt = profile.style_tilt_vector()
    tilted = {s: blended.get(s, 0.0) * tilt.get(s, 1.0) for s in STYLES}
    weights = normalise_weights(tilted)

    rationale = (
        f"Regime={regime.label} (trend {regime.trend}, {regime.vol_state}); "
        + (f"best template={best_template}; " if best_template else "")
        + f"client objective={profile.objective}."
    )
    return StyleRecommendation(weights=weights, regime=regime, rationale=rationale,
                               best_template=best_template, template_blend=blended)
