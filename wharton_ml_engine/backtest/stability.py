"""Signal Stability Model (PRD 6.3).

Measures day-to-day rank stability of the composite signal to distinguish real
information from noise.  Low stability -> low Signal Confidence -> the engine
trades more cautiously.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import STYLES
from ..data.source import DataBundle
from .templates import style_score_panel


@dataclass
class SignalStability:
    confidence: float          # 0-1
    rank_correlation: float    # Spearman between as_of and lagged ranking
    exposure_drift: float      # mean abs change in style panel
    note: str = ""


def _composite(panel: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.Series:
    if weights is None:
        weights = {s: 1.0 for s in STYLES}
    cols = [s for s in weights if s in panel.columns]
    w = np.array([weights[s] for s in cols], dtype=float)
    w = w / w.sum()
    return (panel[cols].fillna(50.0) * w).sum(axis=1)


def signal_stability(
    bundle: DataBundle,
    as_of: pd.Timestamp,
    weights: Optional[Dict[str, float]] = None,
    lag_days: int = 5,
) -> SignalStability:
    as_of = pd.Timestamp(as_of)
    idx = bundle.prices.index
    prior_candidates = idx[idx <= as_of]
    if len(prior_candidates) <= lag_days:
        return SignalStability(0.5, float("nan"), float("nan"),
                               "insufficient history for stability check")
    lagged = prior_candidates[-(lag_days + 1)]

    panel_now = style_score_panel(bundle, as_of)
    panel_lag = style_score_panel(bundle, lagged)
    common = panel_now.index.intersection(panel_lag.index)
    if len(common) < 5:
        return SignalStability(0.5, float("nan"), float("nan"), "too few names")

    comp_now = _composite(panel_now.loc[common], weights)
    comp_lag = _composite(panel_lag.loc[common], weights)
    rank_corr = comp_now.rank().corr(comp_lag.rank(), method="pearson")  # Spearman
    if not np.isfinite(rank_corr):
        rank_corr = 0.0

    drift = float((panel_now.loc[common] - panel_lag.loc[common]).abs().mean().mean())
    # Confidence: mostly rank stability, lightly penalised by exposure drift.
    confidence = float(np.clip(0.5 * (rank_corr + 1.0) - 0.003 * drift, 0.0, 1.0))
    return SignalStability(confidence=confidence, rank_correlation=float(rank_corr),
                           exposure_drift=drift,
                           note="high stability" if confidence > 0.6 else "watch for noise")
