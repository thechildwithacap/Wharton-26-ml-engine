"""Client Fit scoring and hard mandate rules (PRD 5.2).

Produces, per stock:

* ``client_fit`` — a 0-100 alignment score (higher = better fit).
* ``hard_excluded`` / ``exclude_reason`` — securities the mandate forbids.

The hard rules are the non-negotiable exclusions (avoided sectors, excluded
themes, ESG screen).  Soft alignment (income need, risk posture, sector
preference, style preference) shapes the score but does not exclude.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..data.source import DataBundle
from ..utils import clip_score, pct_rank
from .profile import ClientProfile


def _trailing_beta(bundle: DataBundle, as_of: pd.Timestamp, window: int = 252) -> pd.Series:
    prices = bundle.prices_upto(as_of).tail(window + 1)
    if len(prices) < 30 or "SPX" not in bundle.benchmarks.columns:
        return pd.Series(1.0, index=bundle.tickers)
    mkt = bundle.benchmarks["SPX"].reindex(prices.index).pct_change().dropna()
    rets = prices.pct_change().reindex(mkt.index)
    var = mkt.var(ddof=0)
    if not np.isfinite(var) or var == 0:
        return pd.Series(1.0, index=bundle.tickers)
    betas = {}
    for t in bundle.tickers:
        cov = rets[t].cov(mkt)
        betas[t] = cov / var if np.isfinite(cov) else 1.0
    return pd.Series(betas)


def compute_client_fit(
    bundle: DataBundle,
    profile: ClientProfile,
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    if as_of is None:
        as_of = bundle.dates()[-1]
    as_of = pd.Timestamp(as_of)

    fund = bundle.fundamentals_asof(as_of)
    tickers = list(fund.index)
    beta = _trailing_beta(bundle, as_of).reindex(tickers)

    out = pd.DataFrame(index=tickers)
    out["sector"] = [bundle.meta[t].sector for t in tickers]

    # ---- hard exclusions -------------------------------------------------
    excluded = pd.Series(False, index=tickers)
    reason = pd.Series("", index=tickers)

    avoid = set(profile.sector_avoidances)
    for t in tickers:
        m = bundle.meta[t]
        if m.sector in avoid:
            excluded[t] = True
            reason[t] = f"sector excluded ({m.sector})"
        elif profile.excluded_themes and set(m.themes) & set(profile.excluded_themes):
            excluded[t] = True
            reason[t] = "excluded theme"
    out["hard_excluded"] = excluded
    out["exclude_reason"] = reason

    # ---- soft alignment score -------------------------------------------
    score = pd.Series(50.0, index=tickers)

    # Sector preference.
    prefs = set(profile.sector_preferences)
    score += pd.Series([12.0 if bundle.meta[t].sector in prefs else 0.0 for t in tickers],
                       index=tickers)

    # Income alignment.
    income_emphasis = 1.0 if profile.objective == "income" else (
        0.5 if profile.min_dividend_yield > 0 else 0.2)
    yld_rank = pct_rank(fund["dividend_yield"], ascending=True).reindex(tickers)
    score += income_emphasis * (yld_rank.fillna(50.0) - 50.0) * 0.20

    # Risk-posture alignment.
    if profile.risk_tolerance <= 2:
        safe = pct_rank(fund["earnings_vol"], ascending=False).reindex(tickers)     # low vol good
        low_beta = pct_rank(beta, ascending=False).reindex(tickers)
        score += 0.15 * (safe.fillna(50.0) - 50.0) + 0.10 * (low_beta.fillna(50.0) - 50.0)
    elif profile.risk_tolerance >= 4:
        grw = pct_rank(fund["revenue_growth"], ascending=True).reindex(tickers)
        score += 0.15 * (grw.fillna(50.0) - 50.0)

    # ESG proxy: without ESG data, screen the carbon-heavy sectors down.
    if profile.esg_required:
        heavy = {"Energy", "Materials", "Utilities"}
        score += pd.Series([-8.0 if bundle.meta[t].sector in heavy else 3.0
                            for t in tickers], index=tickers)

    out["client_fit"] = clip_score(score)
    # Excluded names carry a zero fit so ranking never surfaces them.
    out.loc[excluded, "client_fit"] = 0.0
    return out
