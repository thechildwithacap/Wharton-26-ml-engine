"""Build the supervised training panel (features + forward-return labels).

For each monthly rebalance date ``t`` we compute the profile-independent signal
scores using **only** data up to ``t`` and pair them with the **forward**
return realised from ``t`` to ``t + horizon`` (PRD §4.1, §9).  Dates without a
fully-realised forward window are dropped, so there is no look-ahead and no
missing labels.

The result is a long/tidy DataFrame — one row per (date, ticker) — ready for
walk-forward training.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ..data.source import DataBundle
from ..signals.fundamental import fundamental_scores
from ..signals.hybrid import analyst_overlay_model
from ..signals.quant import macro_sensitivity_model, price_factor_model


# Profile-independent features the model learns to weight.
ML_FEATURES: List[str] = [
    "value", "quality", "growth", "garp", "income",
    "momentum", "low_vol", "size", "factor", "macro_tilt", "analyst",
]


def feature_frame(bundle: DataBundle, as_of: pd.Timestamp) -> pd.DataFrame:
    """Assemble the ML feature matrix (0-100 scores) at ``as_of`` — no look-ahead."""
    fund = bundle.fundamentals_asof(as_of)
    fnd = fundamental_scores(fund)
    pf = price_factor_model(bundle, as_of)
    macro = macro_sensitivity_model(bundle, as_of)
    analyst = analyst_overlay_model(bundle, as_of, ratings=None)

    out = pd.DataFrame(index=fund.index)
    for col in ["value", "quality", "growth", "garp", "income"]:
        out[col] = fnd[col]
    for col in ["momentum", "low_vol", "size", "factor"]:
        out[col] = pf[col].reindex(fund.index)
    out["macro_tilt"] = macro["macro_tilt"].reindex(fund.index)
    out["analyst"] = analyst.reindex(fund.index)
    return out[ML_FEATURES]


def _monthly_rebalance_dates(bundle: DataBundle, start, end) -> pd.DatetimeIndex:
    idx = bundle.prices.index
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    window = idx[(idx >= start) & (idx <= end)]
    if len(window) == 0:
        return pd.DatetimeIndex([])
    first = pd.Series(window, index=window).groupby([window.year, window.month]).first()
    return pd.DatetimeIndex(sorted(first.values))


def build_training_panel(
    bundle: DataBundle,
    horizon_days: int = 21,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    min_history: int = 252,
) -> pd.DataFrame:
    """Return a tidy panel: date, ticker, <features>, fwd_return, fwd_excess, outperform.

    ``fwd_excess`` is the forward return minus the cross-sectional mean on that
    date (the model's target — relative selection, not market direction).
    ``outperform`` is 1 if the name beat the cross-sectional median.
    """
    dates = bundle.dates()
    if start is None:
        start = dates[min_history]
    if end is None:
        end = dates[-1]

    rdates = _monthly_rebalance_dates(bundle, start, end)
    prices = bundle.prices
    pos = {d: i for i, d in enumerate(prices.index)}

    rows: List[pd.DataFrame] = []
    for d in rdates:
        i = pos[d]
        j = i + horizon_days
        if j >= len(prices.index):
            continue                                    # forward window not realised
        p0 = prices.iloc[i]
        p1 = prices.iloc[j]
        fwd = (p1 / p0) - 1.0

        feats = feature_frame(bundle, d)
        common = feats.index.intersection(fwd.dropna().index)
        if len(common) < 5:
            continue

        block = feats.loc[common].copy()
        block["fwd_return"] = fwd.loc[common]
        block["fwd_excess"] = block["fwd_return"] - block["fwd_return"].mean()
        block["outperform"] = (block["fwd_return"] >
                               block["fwd_return"].median()).astype(int)
        block["date"] = d
        block["ticker"] = common
        rows.append(block)

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", *ML_FEATURES,
                                     "fwd_return", "fwd_excess", "outperform"])
    panel = pd.concat(rows, ignore_index=True)
    # Impute residual missing features to neutral (50 on the 0-100 scale)
    # rather than dropping rows — a data source that lacks one field (e.g.
    # accruals) should not wipe out the whole training set.  A feature that is
    # entirely constant is harmlessly zeroed by the scaler downstream.
    panel[ML_FEATURES] = panel[ML_FEATURES].fillna(50.0)
    return panel.reset_index(drop=True)
