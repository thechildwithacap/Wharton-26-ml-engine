"""Plain-English reasons for a stock's score — no LLM calls, template-filled
from the numbers :mod:`.explain` actually computed.

Every sentence quotes a number that traces back to :class:`~.explain.Explanation`
(a percentile rank) or, when the caller supplies the underlying point-in-time
fundamentals frame, a real raw economic figure for a small set of well-
understood fields (net share issuance, asset growth, P/E, P/B, ROE, dividend
yield). Nothing is fabricated: a feature without a raw-value mapping falls back
to percentile language rather than inventing a number that isn't there.

Style rules (house style, enforced by ``tests/test_narrate.py``):
no em dashes, use contractions, vary sentence length, never use the words
delve/tapestry/robust/pivot/seamless, avoid three-item lists and
"not just X, but also Y" constructions.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .explain import Explanation
from .features import spec as feature_spec

BANNED_WORDS = ("delve", "tapestry", "robust", "pivot", "seamless")
EM_DASH = "—"


# Family -> (positive phrase fn, negative phrase fn). Each takes the top
# feature's percentile rank (0-100) and returns a short lowercase clause
# (no leading capital, no trailing period).
def _pct_phrase(rank: float, positive_word: str, negative_word: str) -> str:
    band = "clearly" if (rank >= 85 or rank <= 15) else "somewhat"
    return f"{band} {positive_word if rank >= 50 else negative_word} (percentile {rank:.0f})"


_FAMILY_PHRASES: Dict[str, tuple] = {
    "value": ("cheap on valuation", "expensive on valuation"),
    "quality": ("strong on profitability", "weak on profitability"),
    "growth": ("growing well", "growing slowly"),
    "momentum": ("in a strong price trend", "in a weak price trend"),
    "risk": ("low-risk on volatility and size", "higher-risk on volatility and size"),
    "macro": ("well-matched to today's market regime", "poorly matched to today's market regime"),
    "issuance": ("returning capital to shareholders", "diluting shareholders"),
    "discipline": ("disciplined with capital", "loose with capital"),
    "analyst_proxy": ("favoured by the fundamentals-based analyst proxy",
                      "disfavoured by the fundamentals-based analyst proxy"),
    "sector": ("well-positioned within its sector", "weak within its sector"),
    "interaction": ("supported by a combined signal", "held back by a combined signal"),
}

# Raw-value formatters for a small set of well-understood fields, keyed by the
# training-panel feature name. Only used when the caller supplies real
# fundamentals (never inferred from a percentile rank).
def _fmt_net_issuance(raw: float) -> Optional[str]:
    if not np.isfinite(raw):
        return None
    if raw < 0:
        return f"share count down {abs(raw) * 100:.0f}% over the trailing year"
    return f"share count up {raw * 100:.0f}% over the trailing year"


def _fmt_asset_growth(raw: float) -> Optional[str]:
    if not np.isfinite(raw):
        return None
    return f"total assets {'up' if raw >= 0 else 'down'} {abs(raw) * 100:.0f}% over the trailing year"


def _fmt_pe(raw: float) -> Optional[str]:
    return f"trades at {raw:.1f}x earnings" if np.isfinite(raw) and raw > 0 else None


def _fmt_pb(raw: float) -> Optional[str]:
    return f"trades at {raw:.1f}x book value" if np.isfinite(raw) and raw > 0 else None


def _fmt_roe(raw: float) -> Optional[str]:
    return f"posts a {raw * 100:.0f}% return on equity" if np.isfinite(raw) else None


def _fmt_dividend_yield(raw: float) -> Optional[str]:
    return f"yields {raw * 100:.1f}%" if np.isfinite(raw) else None


_RAW_FORMATTERS = {
    "net_issuance": _fmt_net_issuance, "f_net_issuance": _fmt_net_issuance,
    "asset_growth": _fmt_asset_growth, "f_asset_growth": _fmt_asset_growth,
    "pe": _fmt_pe, "f_pe": _fmt_pe,
    "pb": _fmt_pb, "f_pb": _fmt_pb,
    "roe": _fmt_roe, "f_roe": _fmt_roe,
}


def _clause_for(feature: str, family: str, raw_pct: float, positive: bool,
                raw_fundamentals: Optional[pd.Series]) -> str:
    pos_word, neg_word = _FAMILY_PHRASES.get(family, (
        f"favourable on {family}", f"unfavourable on {family}"))
    base = _pct_phrase(raw_pct, pos_word, neg_word)
    field = feature[2:] if feature.startswith("f_") else feature
    fmt = _RAW_FORMATTERS.get(field)
    if fmt is not None and raw_fundamentals is not None and field in raw_fundamentals.index:
        detail = fmt(float(raw_fundamentals[field]))
        if detail:
            return f"{detail} ({pos_word if positive else neg_word})"
    return base


def narrate_one(explanation: Explanation, ticker: str, confidence: float,
                n_positive: int = 2, n_negative: int = 2,
                raw_fundamentals: Optional[pd.DataFrame] = None) -> str:
    """Build a 2-4 sentence plain-English reason for one ticker's score."""
    if ticker not in explanation.family_rollup.index:
        return f"No score is available for {ticker} at this date."

    row = explanation.family_rollup.loc[ticker]
    family_cols = [c for c in explanation.family_rollup.columns if c not in ("intercept", "total")]
    fam_contrib = row[family_cols].sort_values(ascending=False)
    positives = fam_contrib[fam_contrib > 0].head(n_positive)
    negatives = fam_contrib[fam_contrib < 0].sort_values().head(n_negative)

    local_t = explanation.local[explanation.local["ticker"] == ticker]
    n_tickers = explanation.meta.get("n_tickers", len(explanation.family_rollup))
    rank = local_t["rank_in_universe"].iloc[0] if len(local_t) else None
    pct = (100.0 * (1.0 - (rank - 1) / max(1, n_tickers - 1))) if rank else 50.0

    raw_row = None
    if raw_fundamentals is not None and ticker in raw_fundamentals.index:
        raw_row = raw_fundamentals.loc[ticker]

    def top_feature_in_family(fam: str):
        sub = local_t[local_t["family"] == fam]
        if sub.empty:
            return None
        return sub.loc[sub["contribution"].abs().idxmax()]

    sentences = []
    if pct >= 80:
        sentences.append(f"{ticker} sits in the model's top {100 - pct:.0f}% (score {pct:.0f}).")
    elif pct <= 20:
        sentences.append(f"{ticker} sits in the model's bottom {pct:.0f}% (score {pct:.0f}).")
    else:
        sentences.append(f"{ticker} lands near the middle of the pack (score {pct:.0f}).")

    if len(positives):
        fam = positives.index[0]
        feat_row = top_feature_in_family(fam)
        if feat_row is not None:
            clause = _clause_for(feat_row["feature"], fam, feat_row["raw_value"],
                                 True, raw_row)
            sentences.append(f"The biggest push comes from {fam.replace('_', ' ')}: it's {clause}.")

    if len(positives) > 1:
        fam = positives.index[1]
        feat_row = top_feature_in_family(fam)
        if feat_row is not None:
            clause = _clause_for(feat_row["feature"], fam, feat_row["raw_value"],
                                 True, raw_row)
            sentences.append(f"It also helps that the stock is {clause}.")

    if len(negatives):
        fam = negatives.index[0]
        feat_row = top_feature_in_family(fam)
        if feat_row is not None:
            clause = _clause_for(feat_row["feature"], fam, feat_row["raw_value"],
                                 False, raw_row)
            sentences.append(f"The drag is {fam.replace('_', ' ')}: it's {clause}.")

    if confidence <= 0.05:
        sentences.append(
            f"Model confidence is {confidence:.2f}, so this view doesn't change the portfolio.")
    elif confidence < 0.5:
        sentences.append(f"Model confidence is only {confidence:.2f}, so this view carries limited weight.")

    text = " ".join(sentences)
    return text


def narrate(explanation: Explanation, confidence: float, tickers=None,
           n_positive: int = 2, n_negative: int = 2,
           raw_fundamentals: Optional[pd.DataFrame] = None) -> pd.Series:
    """Reason text for every (or the given) ticker in ``explanation``."""
    idx = tickers if tickers is not None else list(explanation.family_rollup.index)
    out = {t: narrate_one(explanation, t, confidence, n_positive, n_negative, raw_fundamentals)
          for t in idx}
    return pd.Series(out, name="reason")


def check_style(text: str) -> Dict[str, object]:
    """Return which house-style rules a piece of generated text violates, if any."""
    lower = text.lower()
    violations = {
        "em_dash": EM_DASH in text,
        "banned_words": [w for w in BANNED_WORDS if w in lower],
    }
    violations["ok"] = not violations["em_dash"] and not violations["banned_words"]
    return violations
