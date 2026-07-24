"""Replacement classification: "out of favor" vs "thesis broken" (audit Task B).

When a holding's aggregate score falls, the engine should not treat every
decline the same.  This module classifies each name:

* **out_of_favor** — the aggregate score slipped but the underlying component
  scores (value / quality / momentum) are all still in their normal range →
  the thesis is intact, the market is just not paying for it → *hold*.
* **thesis_broken** — a core component score itself broke below a floor (the
  reason you owned it has deteriorated) → *eligible for replacement*.
* **healthy** — score is fine, no action implied.

Scores are 0-100 cross-sectional percentiles, so "broke below a floor" is a
concrete threshold (default 25 = roughly the bottom quartile of the universe).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# The components that define a name's investment thesis.
THESIS_COMPONENTS = ("value", "quality", "momentum")


@dataclass
class HoldingClassification:
    ticker: str
    status: str                     # healthy | out_of_favor | thesis_broken
    weakest_component: str
    weakest_score: float
    reason: str


def classify_holding(
    ticker: str,
    signals: pd.DataFrame,
    integrated_score: float,
    prior_score: Optional[float] = None,
    break_threshold: float = 25.0,
    decline_threshold: float = 5.0,
    components=THESIS_COMPONENTS,
) -> HoldingClassification:
    """Classify one held name from its component scores.

    ``prior_score`` (if given) lets us require the aggregate to have actually
    *declined* before flagging "out of favor"; without it, any non-broken name
    with an intact thesis is simply "healthy".
    """
    comp = {c: float(signals.loc[ticker, c]) for c in components
            if c in signals.columns and ticker in signals.index
            and np.isfinite(signals.loc[ticker, c])}
    if not comp:
        return HoldingClassification(ticker, "healthy", "", float("nan"),
                                     "no component data")
    weakest = min(comp, key=comp.get)
    wscore = comp[weakest]

    if wscore < break_threshold:
        return HoldingClassification(
            ticker, "thesis_broken", weakest, wscore,
            f"{weakest} {wscore:.0f} < floor {break_threshold:.0f} — thesis deteriorated")

    declined = (prior_score is not None
                and (prior_score - integrated_score) >= decline_threshold)
    if declined:
        return HoldingClassification(
            ticker, "out_of_favor", weakest, wscore,
            f"score {prior_score:.0f}->{integrated_score:.0f} but components intact "
            f"(weakest {weakest} {wscore:.0f}) — hold")
    return HoldingClassification(ticker, "healthy", weakest, wscore, "components intact")


def classify_holdings(
    held: List[str],
    signals: pd.DataFrame,
    scored: pd.DataFrame,
    prior_scored: Optional[pd.DataFrame] = None,
    break_threshold: float = 25.0,
) -> Dict[str, HoldingClassification]:
    out: Dict[str, HoldingClassification] = {}
    for t in held:
        if t not in scored.index:
            continue
        cur = float(scored.loc[t, "integrated_score"]) if "integrated_score" in scored else float("nan")
        prior = None
        if prior_scored is not None and t in prior_scored.index:
            prior = float(prior_scored.loc[t, "integrated_score"])
        out[t] = classify_holding(t, signals, cur, prior, break_threshold)
    return out


def replacement_candidates(classifications: Dict[str, HoldingClassification]) -> List[str]:
    """Names eligible for replacement (thesis broken); out-of-favor names are held."""
    return [t for t, c in classifications.items() if c.status == "thesis_broken"]
