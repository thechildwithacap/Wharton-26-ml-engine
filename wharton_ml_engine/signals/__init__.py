"""Signal layer (PRD 6.1): fundamental, quant factor, and hybrid/thematic pods."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from ..client.profile import ClientProfile
from ..data.source import DataBundle
from .fundamental import (
    fundamental_scores,
    growth_garp_model,
    income_model,
    intrinsic_value_model,
    quality_model,
    value_model,
)
from .hybrid import analyst_overlay_model, hybrid_alpha_model, theme_tilt_model
from .macro import MacroRegime, macro_factor_model, macro_regime
from .quant import macro_sensitivity_model, price_factor_model, quant_scores

__all__ = [
    "value_model",
    "intrinsic_value_model",
    "quality_model",
    "growth_garp_model",
    "income_model",
    "fundamental_scores",
    "price_factor_model",
    "macro_sensitivity_model",
    "quant_scores",
    "MacroRegime",
    "macro_regime",
    "macro_factor_model",
    "analyst_overlay_model",
    "hybrid_alpha_model",
    "theme_tilt_model",
    "compute_signals",
]


def compute_signals(
    bundle: DataBundle,
    profile: ClientProfile,
    as_of: Optional[pd.Timestamp] = None,
    analyst_ratings: Optional[Dict[str, Dict[str, float]]] = None,
    alpha_model: Optional[object] = None,
) -> pd.DataFrame:
    """Run every signal-layer model and return one ticker-indexed frame.

    Columns: value, quality, growth, garp, income, momentum, low_vol, size,
    factor, beta, volatility, macro_tilt, analyst, hybrid_alpha, theme_fit,
    plus the ``margin_of_safety`` flag.  When a trained ``alpha_model`` is
    supplied, its 0-100 ``ml_alpha`` learned score is added as a column.
    """
    if as_of is None:
        as_of = bundle.dates()[-1]
    as_of = pd.Timestamp(as_of)

    fund = bundle.fundamentals_asof(as_of)
    fundamentals = fundamental_scores(fund)
    quant = quant_scores(bundle, as_of)
    analyst = analyst_overlay_model(bundle, as_of, analyst_ratings)
    theme = theme_tilt_model(bundle, profile, as_of)
    hybrid = hybrid_alpha_model(fundamentals, analyst, quant)

    out = fundamentals.join(quant, how="outer")
    out["analyst"] = analyst
    out["theme_fit"] = theme
    out["hybrid_alpha"] = hybrid
    if alpha_model is not None:
        out["ml_alpha"] = alpha_model.score(bundle, as_of).reindex(out.index)
    return out
