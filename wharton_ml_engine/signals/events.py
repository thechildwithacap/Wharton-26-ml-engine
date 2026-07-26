"""Earnings-event signals: post-earnings drift and the overreaction setup.

Value investing profits from *market overreaction* to news.  The article's
worked example is Fitbit: revenue up >50% year-over-year and guidance raised,
but EPS dipped on heavy R&D spending, so the market dumped the stock ~19% —
punishing a company whose fundamentals were still improving.  A value investor
buys that dislocation.

This module reconstructs that setup from data already in the bundle:

* the **event date** is when a new fundamentals snapshot became public (the SEC
  filing / report date already carried in the point-in-time panel);
* the **price reaction** is the return over a short window around that date;
* the **fundamental trend** is whether revenue/EPS/profitability were improving.

``overreaction`` scores highest when the market punished a name whose
fundamentals were *improving* — a disagreement between price and business
performance.  ``earnings_drift`` is the plain post-event return, which captures
the opposite, momentum-style effect (post-earnings-announcement drift), so the
two are deliberately separate signals rather than one blended number.

Everything is strictly point-in-time: only snapshots dated on or before
``as_of`` and prices up to ``as_of`` are read.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..data.source import DataBundle
from ..utils import clip_score, pct_rank, score_higher_is_better


def _last_event_dates(bundle: DataBundle, as_of: pd.Timestamp) -> pd.Series:
    """Most recent fundamentals-publication date per ticker, on or before ``as_of``."""
    dates = bundle.fundamentals.index.get_level_values(0)
    visible = bundle.fundamentals.loc[dates <= as_of]
    if visible.empty:
        return pd.Series(dtype="datetime64[ns]")
    r = visible.reset_index()
    return r.groupby("ticker")["date"].max()


def earnings_event_model(bundle: DataBundle, as_of: pd.Timestamp,
                         pre_days: int = 3, post_days: int = 3,
                         stale_days: int = 200) -> pd.DataFrame:
    """Per-stock earnings-event signals as of ``as_of``.

    Columns
    -------
    days_since_event   trading days since the last fundamentals publication
    event_reaction     return around the event window, minus the market's
    earnings_drift     return from the event window to ``as_of``, minus market
    fundamental_trend  0-100: is the business improving (revenue/EPS/ROE)?
    overreaction       0-100: punished by the market *despite* improving
                       fundamentals (higher = bigger apparent dislocation)
    """
    as_of = pd.Timestamp(as_of)
    tickers = bundle.tickers
    out = pd.DataFrame(index=tickers)
    px = bundle.prices_upto(as_of)
    if len(px) < 30:
        for c in ("days_since_event", "event_reaction", "earnings_drift"):
            out[c] = np.nan
        out["fundamental_trend"] = 50.0
        out["overreaction"] = 50.0
        return out

    idx = px.index
    events = _last_event_dates(bundle, as_of)

    # Market return series to neutralise the event window (isolate the
    # stock-specific reaction rather than a market-wide move).
    if "SPX" in bundle.benchmarks.columns:
        mkt = bundle.benchmarks["SPX"].reindex(idx).ffill()
    else:
        mkt = px.mean(axis=1)

    days_since, reaction, drift = {}, {}, {}
    for t in tickers:
        ev = events.get(t)
        if ev is None or pd.isna(ev):
            days_since[t] = reaction[t] = drift[t] = np.nan
            continue
        # locate the event on the trading calendar
        loc = idx.searchsorted(pd.Timestamp(ev))
        if loc >= len(idx):
            days_since[t] = reaction[t] = drift[t] = np.nan
            continue
        days_since[t] = float(len(idx) - 1 - loc)
        i0, i1 = max(0, loc - pre_days), min(len(idx) - 1, loc + post_days)
        s = px[t]
        p0, p1, pN = s.iloc[i0], s.iloc[i1], s.iloc[-1]
        m0, m1, mN = mkt.iloc[i0], mkt.iloc[i1], mkt.iloc[-1]
        if np.isfinite(p0) and np.isfinite(p1) and p0 > 0 and m0 > 0:
            reaction[t] = float((p1 / p0 - 1.0) - (m1 / m0 - 1.0))
        else:
            reaction[t] = np.nan
        if np.isfinite(p1) and np.isfinite(pN) and p1 > 0 and m1 > 0:
            drift[t] = float((pN / p1 - 1.0) - (mN / m1 - 1.0))
        else:
            drift[t] = np.nan

    out["days_since_event"] = pd.Series(days_since)
    out["event_reaction"] = pd.Series(reaction)
    out["earnings_drift"] = pd.Series(drift)

    # --- is the business actually improving? ---
    fund = bundle.fundamentals_asof(as_of)
    prior = bundle.fundamentals_asof(as_of - pd.Timedelta(days=365))
    def _col(f, c):
        return (pd.to_numeric(f[c], errors="coerce").reindex(tickers)
                if (f is not None and not f.empty and c in f.columns)
                else pd.Series(np.nan, index=tickers))
    rev_g, eps_g = _col(fund, "revenue_growth"), _col(fund, "eps_growth")
    roe, roe_prev = _col(fund, "roe"), _col(prior, "roe")
    roe_delta = roe - roe_prev

    trend_raw = pd.concat([
        score_higher_is_better(rev_g), score_higher_is_better(eps_g),
        score_higher_is_better(roe_delta),
    ], axis=1).mean(axis=1, skipna=True)
    out["fundamental_trend"] = clip_score(trend_raw)

    # --- overreaction: market punished an improving business ---
    # Only meaningful while the event is recent; stale events decay to neutral.
    punished = clip_score(pct_rank(-out["event_reaction"], ascending=True))
    combo = 0.5 * punished + 0.5 * out["fundamental_trend"]
    recent = out["days_since_event"] <= stale_days
    out["overreaction"] = clip_score(combo).where(recent, 50.0).fillna(50.0)
    return out
