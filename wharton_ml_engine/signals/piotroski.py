"""Piotroski F-Score — the 9-point fundamental-strength / value-trap filter.

Piotroski (2000) showed that among *cheap* stocks, a simple 9-point check on
profitability, leverage and operating efficiency separates the genuinely
undervalued from the value traps.  Buying high-F-score value names and avoiding
low-F-score ones produced a large spread.  That makes it the natural companion
to the intrinsic-value / margin-of-safety model: value says *how cheap*, the
F-score says *whether the cheapness is deserved*.

The nine binary tests (1 point each, 9 = strongest):

Profitability
  1. ROA > 0
  2. Operating cash flow > 0
  3. ROA improved year-over-year
  4. Operating cash flow > net income   (earnings backed by cash, low accruals)
Leverage / liquidity / source of funds
  5. Long-term leverage decreased
  6. Current ratio improved
  7. No new shares issued
Operating efficiency
  8. Gross margin improved
  9. Asset turnover improved

Data note (honest): the engine's fundamentals panel carries snapshots, not a
full statement history, so tests are computed from the fields available and
each one that cannot be evaluated is reported in ``f_missing`` rather than
silently scored as a pass.  ``f_score`` counts only the tests that passed, and
``f_score_pct`` normalises by the tests actually evaluated so names with partial
data stay comparable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..utils import clip_score, pct_rank

# The nine tests, in Piotroski's order.
F_TESTS = [
    "roa_positive", "cfo_positive", "roa_improved", "accruals_ok",
    "leverage_down", "liquidity_up", "no_dilution",
    "margin_up", "turnover_up",
]


def _prev(panel: pd.DataFrame, tickers, col: str, as_of: pd.Timestamp,
          lookback_days: int = 365) -> pd.Series:
    """Value of ``col`` roughly one year before ``as_of`` (point-in-time)."""
    if panel is None or panel.empty or col not in panel.columns:
        return pd.Series(np.nan, index=tickers)
    dates = panel.index.get_level_values(0)
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=lookback_days)
    visible = panel.loc[dates <= cutoff]
    if visible.empty:
        return pd.Series(np.nan, index=tickers)
    latest = (visible.reset_index().sort_values("date")
              .groupby("ticker").tail(1).set_index("ticker"))
    return latest[col].reindex(tickers)


def piotroski_f_score(fund: pd.DataFrame,
                      prior: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Compute the F-score from a current fundamentals frame (+ optional prior).

    ``fund`` is ticker-indexed (as from ``DataBundle.fundamentals_asof``).
    ``prior`` is the same frame ~1 year earlier; when omitted, the four
    year-over-year tests are marked missing rather than assumed.
    """
    idx = fund.index
    g = lambda c: (pd.to_numeric(fund[c], errors="coerce")
                   if c in fund.columns else pd.Series(np.nan, index=idx))
    gp = lambda c: (pd.to_numeric(prior[c], errors="coerce").reindex(idx)
                    if (prior is not None and c in prior.columns)
                    else pd.Series(np.nan, index=idx))

    roe, roic = g("roe"), g("roic")
    # ROA proxy: ROIC is return on total capital — the closest available stand-in
    # for return on assets in this schema.
    roa = roic.where(roic.notna(), roe)
    roa_prev = gp("roic").where(gp("roic").notna(), gp("roe"))

    fcf, accr = g("fcf_yield"), g("accruals")
    debt, debt_prev = g("debt_equity"), gp("debt_equity")
    cover, cover_prev = g("interest_coverage"), gp("interest_coverage")
    issue = g("net_issuance")
    gm, gm_prev = g("gross_margin"), gp("gross_margin")
    rev_growth, asset_growth = g("revenue_growth"), g("asset_growth")

    tests = {}
    #  1. ROA > 0
    tests["roa_positive"] = roa > 0
    #  2. Operating cash flow positive (FCF yield as the available cash proxy)
    tests["cfo_positive"] = fcf > 0
    #  3. ROA improved YoY
    tests["roa_improved"] = roa > roa_prev
    #  4. Earnings backed by cash: low/negative accruals
    tests["accruals_ok"] = accr < 0 if accr.notna().any() else pd.Series(np.nan, index=idx)
    #  5. Leverage decreased
    tests["leverage_down"] = debt < debt_prev
    #  6. Liquidity improved (interest coverage as the solvency proxy for the
    #     current ratio, which this schema does not carry)
    tests["liquidity_up"] = cover > cover_prev
    #  7. No new shares issued (buyback or flat)
    tests["no_dilution"] = issue <= 0
    #  8. Gross margin improved
    tests["margin_up"] = gm > gm_prev
    #  9. Asset turnover improved: revenue growing faster than assets
    tests["turnover_up"] = rev_growth > asset_growth

    out = pd.DataFrame(index=idx)
    evaluable = pd.Series(0, index=idx)
    passed = pd.Series(0, index=idx)
    for name in F_TESTS:
        t = tests[name]
        val = t.fillna(False).astype(bool)
        # A NaN input makes the comparison return False; mark those rows as
        # not-evaluable so missing data is never scored as a failed test.
        inputs_known = _inputs_known(name, locals_map={
            "roa": roa, "roa_prev": roa_prev, "fcf": fcf, "accr": accr,
            "debt": debt, "debt_prev": debt_prev, "cover": cover,
            "cover_prev": cover_prev, "issue": issue, "gm": gm,
            "gm_prev": gm_prev, "rev_growth": rev_growth,
            "asset_growth": asset_growth,
        }, index=idx)
        out[name] = val.where(inputs_known)
        evaluable += inputs_known.astype(int)
        passed += (val & inputs_known).astype(int)

    out["f_score"] = passed
    out["f_evaluated"] = evaluable
    out["f_missing"] = len(F_TESTS) - evaluable
    # Normalised so partial-data names remain comparable (0-9 scale).
    out["f_score_pct"] = np.where(evaluable > 0, passed / evaluable, np.nan)
    out["f_score_adj"] = out["f_score_pct"] * len(F_TESTS)
    # Cross-sectional 0-100 score for blending with the other style scores.
    out["piotroski"] = clip_score(pct_rank(out["f_score_pct"], ascending=True))
    # Piotroski's own labels.
    out["f_strong"] = out["f_score_adj"] >= 7.0
    out["f_weak"] = out["f_score_adj"] <= 3.0
    return out


_TEST_INPUTS = {
    "roa_positive": ["roa"],
    "cfo_positive": ["fcf"],
    "roa_improved": ["roa", "roa_prev"],
    "accruals_ok": ["accr"],
    "leverage_down": ["debt", "debt_prev"],
    "liquidity_up": ["cover", "cover_prev"],
    "no_dilution": ["issue"],
    "margin_up": ["gm", "gm_prev"],
    "turnover_up": ["rev_growth", "asset_growth"],
}


def _inputs_known(test: str, locals_map: dict, index) -> pd.Series:
    known = pd.Series(True, index=index)
    for key in _TEST_INPUTS[test]:
        s = locals_map.get(key)
        if s is None:
            return pd.Series(False, index=index)
        known &= s.notna()
    return known
