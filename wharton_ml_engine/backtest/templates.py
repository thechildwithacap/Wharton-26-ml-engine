"""Strategy templates and the shared style-score panel (PRD 6.3).

A *template* is a named set of style weights.  The backtest engine ranks the
universe by each template's weighted style score and tracks how the templates
would have performed, feeding the style-weight recommendation.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from ..config import STYLES
from ..data.source import DataBundle
from ..signals.fundamental import fundamental_scores
from ..signals.quant import price_factor_model
from ..utils import weighted_blend


# Canonical strategy templates (PRD 6.3).  Weights need not sum to 1 — they are
# renormalised at use.
STRATEGY_TEMPLATES: Dict[str, Dict[str, float]] = {
    "deep_value":        {"value": 0.6, "quality": 0.2, "income": 0.1, "low_vol": 0.1},
    "quality_compounder":{"quality": 0.55, "growth": 0.2, "value": 0.15, "low_vol": 0.1},
    "garp_blend":        {"garp": 0.4, "growth": 0.25, "quality": 0.2, "value": 0.15},
    "momentum_tilt":     {"momentum": 0.55, "growth": 0.2, "quality": 0.15, "low_vol": 0.1},
    "defensive_low_vol": {"low_vol": 0.45, "quality": 0.3, "income": 0.15, "value": 0.1},
    "hybrid_multifactor":{"value": 0.2, "quality": 0.25, "growth": 0.15,
                          "momentum": 0.2, "low_vol": 0.1, "income": 0.1},
}


def style_score_panel(bundle: DataBundle, as_of: pd.Timestamp) -> pd.DataFrame:
    """Compute just the style scores needed for templates at ``as_of``.

    Lighter than the full signal layer: fundamentals + price factors only.
    Columns are exactly :data:`STYLES`.
    """
    fund = bundle.fundamentals_asof(as_of)
    fnd = fundamental_scores(fund)
    pf = price_factor_model(bundle, as_of)

    out = pd.DataFrame(index=fund.index)
    out["value"] = fnd["value"]
    out["quality"] = fnd["quality"]
    out["growth"] = fnd["growth"]
    out["garp"] = fnd["garp"]
    out["momentum"] = pf["momentum"].reindex(fund.index)
    out["low_vol"] = pf["low_vol"].reindex(fund.index)
    return out[STYLES]


def template_score(panel: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    comps = {s: panel[s] for s in weights if s in panel.columns}
    return weighted_blend(comps, weights)


def normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, v) for v in weights.values())
    if total <= 0:
        return {s: 1.0 / len(STYLES) for s in STYLES}
    return {k: max(0.0, v) / total for k, v in weights.items()}
