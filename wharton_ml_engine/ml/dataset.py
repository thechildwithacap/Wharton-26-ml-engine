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
from ..signals.macro import macro_factor_model
from ..signals.quant import macro_sensitivity_model, price_factor_model
from ..utils import clip_score, pct_rank


# Profile-independent features the model learns to weight.
# Income dropped from the alpha feature set (no significant IC, t=+0.19); it
# stays in Client Fit for the income mandate, not as a forecasting feature.
ML_FEATURES: List[str] = [
    "value", "quality", "growth", "garp", "discipline",
    "momentum", "low_vol", "size", "factor", "macro_tilt", "macro_fit", "analyst",
]

# Raw fundamental fields fed to the model directly (as cross-sectional ranks),
# so it can learn from *all* the underlying SEC data — not only the hand-built
# composite style scores.  Prefixed ``f_`` to keep them distinct.
# Raw fundamental features for the extended model.  Dividend yield / payout are
# excluded — like the income style, they are income-mandate inputs (Client Fit),
# not forecasting features.
RAW_FUNDAMENTAL_FIELDS: List[str] = [
    "pe", "pb", "ev_ebit", "fcf_yield", "roe", "roic", "gross_margin",
    "earnings_vol", "debt_equity", "interest_coverage", "revenue_growth",
    "eps_growth", "growth_stability", "accruals", "net_issuance", "asset_growth",
]
RAW_FUNDAMENTAL_FEATURES: List[str] = [f"f_{c}" for c in RAW_FUNDAMENTAL_FIELDS]
EXTENDED_FEATURES: List[str] = ML_FEATURES + RAW_FUNDAMENTAL_FEATURES


def feature_columns(extended=False) -> List[str]:
    """Return a named feature set.

    Backward-compatible: ``extended=True/False`` behaves as before. Also
    accepts the set name directly — ``"base"`` (== ``False``) or ``"extended"``
    (== ``True``) — since callers may prefer a self-describing string. Every
    name returned here must have a :class:`~.features.FeatureSpec` registered
    in :mod:`.features` (enforced by ``test_features_registry.py``).
    """
    if isinstance(extended, str):
        if extended not in ("base", "extended"):
            raise ValueError(f"unknown feature set {extended!r}; use 'base' or 'extended'")
        extended = extended == "extended"
    return EXTENDED_FEATURES if extended else ML_FEATURES


def is_extended(feature_names: List[str]) -> bool:
    return any(str(f).startswith("f_") for f in feature_names)


def feature_frame(bundle: DataBundle, as_of: pd.Timestamp,
                  extended: bool = False) -> pd.DataFrame:
    """Assemble the ML feature matrix (0-100 scores) at ``as_of`` — no look-ahead.

    With ``extended=True`` the raw SEC fundamental fields are appended as
    cross-sectional percentile ranks, giving the model direct access to every
    reported line item rather than only the composite style scores.
    """
    fund = bundle.fundamentals_asof(as_of)
    fnd = fundamental_scores(fund)
    pf = price_factor_model(bundle, as_of)
    macro = macro_sensitivity_model(bundle, as_of)
    macro_f = macro_factor_model(bundle, as_of)
    analyst = analyst_overlay_model(bundle, as_of, ratings=None)

    out = pd.DataFrame(index=fund.index)
    for col in ["value", "quality", "growth", "garp", "discipline"]:
        out[col] = fnd[col]
    for col in ["momentum", "low_vol", "size", "factor"]:
        out[col] = pf[col].reindex(fund.index)
    out["macro_tilt"] = macro["macro_tilt"].reindex(fund.index)
    out["macro_fit"] = macro_f["macro_fit"].reindex(fund.index)
    out["analyst"] = analyst.reindex(fund.index)

    if extended:
        for raw, feat in zip(RAW_FUNDAMENTAL_FIELDS, RAW_FUNDAMENTAL_FEATURES):
            if raw in fund.columns:
                out[feat] = clip_score(pct_rank(fund[raw], ascending=True))
            else:
                out[feat] = np.nan
        return out[EXTENDED_FEATURES]
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
    extended: bool = False,
    point_in_time: bool = False,
    universe_filter: Optional[object] = None,
) -> pd.DataFrame:
    """Return a tidy panel: date, ticker, <features>, fwd_return, fwd_excess, outperform.

    ``fwd_excess`` is the forward return minus the cross-sectional mean on that
    date (the model's target — relative selection, not market direction).
    ``outperform`` is 1 if the name beat the cross-sectional median.
    ``extended`` adds the raw SEC fundamental rank features.

    ``point_in_time=True`` restricts each rebalance date's cross-section (both
    the feature ranks and the forward-return label pool) to
    ``bundle.eligible_asof(d, universe_filter)`` — names that were actually
    listed, not yet delisted, and above the liquidity floor *on that date* —
    instead of every ticker present anywhere in the bundle. Without this, a
    ticker that only joined the universe partway through history still has its
    early-period cross-sectional ranks computed as if it had always been a
    member, which both flatters IC (a future addition tends to be a survivor)
    and is a look-ahead bias in the label pool. Requires the bundle's
    ``SecurityMeta.listing_date``/``delisting_date`` to reflect true point-in-time
    *index* membership (not just IPO/bankruptcy dates) for the effect to be
    meaningful on real S&P-500-style data — a bundle with no listing/delisting
    metadata set behaves identically to ``point_in_time=False``.
    """
    feature_cols = feature_columns(extended)
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

        if point_in_time:
            members = bundle.eligible_asof(d, universe_filter)
            if len(members) < 5:
                continue
            sub = bundle.restrict_universe(members)
            feats = feature_frame(sub, d, extended=extended)
        else:
            feats = feature_frame(bundle, d, extended=extended)
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
        return pd.DataFrame(columns=["date", "ticker", *feature_cols,
                                     "fwd_return", "fwd_excess", "outperform"])
    panel = pd.concat(rows, ignore_index=True)
    # Impute residual missing features to neutral (50 on the 0-100 scale)
    # rather than dropping rows — a data source that lacks one field (e.g.
    # accruals) should not wipe out the whole training set.  A feature that is
    # entirely constant is harmlessly zeroed by the scaler downstream.
    panel[feature_cols] = panel[feature_cols].fillna(50.0)
    return panel.reset_index(drop=True)
