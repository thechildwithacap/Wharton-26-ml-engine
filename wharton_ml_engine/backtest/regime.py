"""Regime Classification Model (PRD 6.3).

Classifies the market environment from index trend, realised volatility and
breadth into a coarse label (bull / bear / sideways crossed with high / low
vol).  The label drives the style-weight recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from ..data.source import DataBundle


@dataclass
class RegimeState:
    label: str                      # e.g. "bull_low_vol"
    trend: str                      # bull | bear | sideways
    vol_state: str                  # low_vol | high_vol
    features: Dict[str, float] = field(default_factory=dict)


def _breadth(bundle: DataBundle, as_of: pd.Timestamp, window: int = 200) -> float:
    px = bundle.prices_upto(as_of).tail(window)
    if len(px) < 20:
        return 0.5
    ma = px.mean()
    above = (px.iloc[-1] > ma).mean()
    return float(above)


def classify_regime(bundle: DataBundle, as_of: pd.Timestamp,
                    index: str = "SPX") -> RegimeState:
    bench = bundle.benchmarks[index]
    hist = bench.loc[bench.index <= pd.Timestamp(as_of)]
    features: Dict[str, float] = {}

    if len(hist) < 60:
        return RegimeState("sideways_low_vol", "sideways", "low_vol",
                           {"insufficient_history": 1.0})

    ma200 = hist.tail(200).mean()
    last = hist.iloc[-1]
    ret_6m = last / hist.iloc[-min(126, len(hist) - 1)] - 1.0
    rets = hist.pct_change().dropna()
    vol_21 = rets.tail(21).std(ddof=0) * np.sqrt(252)
    vol_long = rets.std(ddof=0) * np.sqrt(252)
    breadth = _breadth(bundle, as_of)

    features.update({
        "price_vs_ma200": float(last / ma200 - 1.0),
        "ret_6m": float(ret_6m),
        "vol_21d_annualised": float(vol_21),
        "vol_ratio": float(vol_21 / vol_long) if vol_long else 1.0,
        "breadth": breadth,
    })

    # Trend.
    if last > ma200 and ret_6m > 0.02 and breadth > 0.55:
        trend = "bull"
    elif last < ma200 and ret_6m < -0.02 and breadth < 0.45:
        trend = "bear"
    else:
        trend = "sideways"

    # Volatility state relative to the security's own long-run vol.
    vol_state = "high_vol" if vol_21 > 1.15 * vol_long else "low_vol"

    return RegimeState(label=f"{trend}_{vol_state}", trend=trend,
                       vol_state=vol_state, features=features)
