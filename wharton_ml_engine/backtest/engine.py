"""Strategy Template Backtest Model (PRD 6.3).

A light, transparent backtester: at each monthly rebalance it ranks the
universe by a template's weighted style score, holds the top-N equal weighted,
and stitches the daily returns together.  It intentionally trades monthly and
equal-weights to stay fast and legible — this is a *research* backtest for
choosing style tilts, not an execution simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.source import DataBundle
from ..utils import annualise_return, annualise_vol, max_drawdown, sharpe_ratio
from .templates import STRATEGY_TEMPLATES, style_score_panel, template_score


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series
    metrics: Dict[str, float] = field(default_factory=dict)
    holdings_by_date: Dict[pd.Timestamp, List[str]] = field(default_factory=dict)


def _rebalance_dates(bundle: DataBundle, start, end) -> pd.DatetimeIndex:
    idx = bundle.prices.index
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    window = idx[(idx >= start) & (idx <= end)]
    if len(window) == 0:
        return pd.DatetimeIndex([])
    months = pd.Series(window, index=window).groupby(
        [window.year, window.month]
    ).first()
    return pd.DatetimeIndex(sorted(months.values))


def _metrics(returns: pd.Series, periods: int = 252) -> Dict[str, float]:
    if returns.dropna().empty:
        return {k: float("nan") for k in
                ["ann_return", "ann_vol", "sharpe", "max_drawdown", "worst_year"]}
    yearly = (1 + returns).groupby(returns.index.year).prod() - 1.0
    return {
        "ann_return": annualise_return(returns, periods),
        "ann_vol": annualise_vol(returns, periods),
        "sharpe": sharpe_ratio(returns, periods=periods),
        "max_drawdown": max_drawdown(returns),
        "worst_year": float(yearly.min()) if len(yearly) else float("nan"),
    }


def backtest_template(
    bundle: DataBundle,
    weights: Dict[str, float],
    start,
    end,
    name: str = "template",
    top_n: int = 25,
) -> BacktestResult:
    rdates = _rebalance_dates(bundle, start, end)
    daily_prices = bundle.prices
    all_returns: List[pd.Series] = []
    holdings: Dict[pd.Timestamp, List[str]] = {}

    for i, d in enumerate(rdates):
        panel = style_score_panel(bundle, d)
        score = template_score(panel, weights).dropna()
        if score.empty:
            continue
        picks = list(score.sort_values(ascending=False).head(top_n).index)
        holdings[d] = picks

        nxt = rdates[i + 1] if i + 1 < len(rdates) else pd.Timestamp(end)
        mask = (daily_prices.index > d) & (daily_prices.index <= nxt)
        seg = daily_prices.loc[mask, picks]
        if len(seg) < 2:
            continue
        # Equal-weight daily-rebalanced portfolio return.
        port_ret = seg.pct_change().dropna(how="all").mean(axis=1)
        all_returns.append(port_ret)

    if all_returns:
        returns = pd.concat(all_returns).sort_index()
        returns = returns[~returns.index.duplicated(keep="first")]
    else:
        returns = pd.Series(dtype=float)

    return BacktestResult(name=name, returns=returns,
                          metrics=_metrics(returns), holdings_by_date=holdings)


def backtest_all_templates(
    bundle: DataBundle,
    start,
    end,
    top_n: int = 25,
    templates: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, BacktestResult]:
    """Backtest every template, computing each rebalance panel only once.

    The style-score panel is the expensive part, so it is cached per rebalance
    date and reused across all templates (a large speed-up over running each
    template independently).
    """
    templates = templates or STRATEGY_TEMPLATES
    rdates = _rebalance_dates(bundle, start, end)
    daily_prices = bundle.prices

    # Precompute the panel and the forward return segment once per date.
    panels: Dict[pd.Timestamp, pd.DataFrame] = {}
    segments: Dict[pd.Timestamp, pd.DataFrame] = {}
    for i, d in enumerate(rdates):
        panels[d] = style_score_panel(bundle, d)
        nxt = rdates[i + 1] if i + 1 < len(rdates) else pd.Timestamp(end)
        mask = (daily_prices.index > d) & (daily_prices.index <= nxt)
        segments[d] = daily_prices.loc[mask]

    results: Dict[str, BacktestResult] = {}
    for name, w in templates.items():
        all_returns: List[pd.Series] = []
        holdings: Dict[pd.Timestamp, List[str]] = {}
        for d in rdates:
            score = template_score(panels[d], w).dropna()
            if score.empty:
                continue
            picks = list(score.sort_values(ascending=False).head(top_n).index)
            holdings[d] = picks
            seg = segments[d][picks]
            if len(seg) < 2:
                continue
            all_returns.append(seg.pct_change().dropna(how="all").mean(axis=1))
        if all_returns:
            returns = pd.concat(all_returns).sort_index()
            returns = returns[~returns.index.duplicated(keep="first")]
        else:
            returns = pd.Series(dtype=float)
        results[name] = BacktestResult(name=name, returns=returns,
                                       metrics=_metrics(returns),
                                       holdings_by_date=holdings)
    return results


def benchmark_returns(bundle: DataBundle, start, end, index: str = "SPX") -> pd.Series:
    b = bundle.benchmarks[index]
    b = b[(b.index >= pd.Timestamp(start)) & (b.index <= pd.Timestamp(end))]
    return b.pct_change().dropna()


def backtest_template_pit(
    bundle: DataBundle,
    weights: Dict[str, float],
    start,
    end,
    name: str = "template",
    top_n: int = 25,
    universe: Optional[List[str]] = None,
) -> BacktestResult:
    """Point-in-time backtest that respects delisting (survivorship-safe).

    Unlike :func:`backtest_template`, at each rebalance the pick list is drawn
    only from names **eligible on that date** (``bundle.eligible_asof`` — listed,
    not yet delisted, above the floors).  A held name that delists mid-period is
    carried in the return stream up to and including its final (delisting) mark,
    then drops out — so a failure's loss is realised, never silently truncated.

    ``universe`` optionally restricts the candidate set (e.g. survivor-only, to
    *measure* the bias against the full point-in-time set).
    """
    rdates = _rebalance_dates(bundle, start, end)
    daily_prices = bundle.prices
    restrict = set(universe) if universe is not None else None

    all_returns: List[pd.Series] = []
    holdings: Dict[pd.Timestamp, List[str]] = {}
    for i, d in enumerate(rdates):
        eligible = bundle.eligible_asof(d)
        if restrict is not None:
            eligible = [t for t in eligible if t in restrict]
        if not eligible:
            continue
        panel = style_score_panel(bundle, d).reindex(eligible)
        score = template_score(panel, weights).dropna()
        if score.empty:
            continue
        picks = list(score.sort_values(ascending=False).head(top_n).index)
        holdings[d] = picks

        nxt = rdates[i + 1] if i + 1 < len(rdates) else pd.Timestamp(end)
        mask = (daily_prices.index > d) & (daily_prices.index <= nxt)
        seg = daily_prices.loc[mask, picks]
        if len(seg) < 2:
            continue
        # Per-name daily returns; a name that delists mid-segment simply stops
        # contributing after its last priced day (its delisting mark is the final
        # step in that series), and the equal-weight mean is over live names.
        all_returns.append(seg.pct_change().dropna(how="all").mean(axis=1))

    if all_returns:
        returns = pd.concat(all_returns).sort_index()
        returns = returns[~returns.index.duplicated(keep="first")]
    else:
        returns = pd.Series(dtype=float)
    return BacktestResult(name=name, returns=returns,
                          metrics=_metrics(returns), holdings_by_date=holdings)


@dataclass
class MeasuredSurvivorshipBias:
    """The bias measured from a real two-way backtest (not modeled)."""

    full_cagr: float            # delisting-inclusive (bias-free) annual return
    survivor_cagr: float        # survivor-only (biased) annual return
    bias_annual: float          # survivor_cagr - full_cagr (how much it overstates)
    n_full: int                 # names in the delisting-inclusive universe
    n_survivors: int            # names still listed at ``end``
    n_delisted: int
    full_metrics: Dict[str, float]
    survivor_metrics: Dict[str, float]

    def as_dict(self) -> Dict[str, object]:
        return {
            "full_cagr": round(self.full_cagr, 4),
            "survivor_cagr": round(self.survivor_cagr, 4),
            "measured_bias_annual": round(self.bias_annual, 4),
            "n_full": self.n_full, "n_survivors": self.n_survivors,
            "n_delisted": self.n_delisted,
            "full_sharpe": round(self.full_metrics.get("sharpe", float("nan")), 3),
            "survivor_sharpe": round(self.survivor_metrics.get("sharpe", float("nan")), 3),
            "method": "two-way PIT backtest on the same bundle (real delistings)",
        }


def measure_survivorship_bias(
    bundle: DataBundle,
    weights: Dict[str, float],
    start,
    end,
    top_n: int = 25,
) -> MeasuredSurvivorshipBias:
    """Measure survivorship bias by backtesting the *same* strategy twice.

    1. **Full** — the point-in-time universe including names that later delisted
       (survivorship-free): losses on failures are realised.
    2. **Survivor-only** — restricted to names still listed at ``end`` (the
       biased view a free feed forces on you, since it 404s delisted tickers).

    The gap ``survivor_cagr - full_cagr`` is the annual return the survivor view
    **overstates** — measured from real prices, not assumed.  Requires a bundle
    carrying delisting metadata (CRSP, or the synthetic delisting demo).
    """
    end_ts = pd.Timestamp(end)
    survivors = [t for t, m in bundle.meta.items()
                 if m.delisting_date is None or pd.Timestamp(m.delisting_date) > end_ts]
    all_names = list(bundle.meta.keys())
    n_delisted = len(all_names) - len(survivors)

    full = backtest_template_pit(bundle, weights, start, end,
                                 name="full_universe", top_n=top_n)
    surv = backtest_template_pit(bundle, weights, start, end,
                                 name="survivors_only", top_n=top_n,
                                 universe=survivors)
    full_cagr = full.metrics.get("ann_return", float("nan"))
    surv_cagr = surv.metrics.get("ann_return", float("nan"))
    return MeasuredSurvivorshipBias(
        full_cagr=full_cagr, survivor_cagr=surv_cagr,
        bias_annual=surv_cagr - full_cagr,
        n_full=len(all_names), n_survivors=len(survivors), n_delisted=n_delisted,
        full_metrics=full.metrics, survivor_metrics=surv.metrics,
    )


def templates_summary(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    rows = {name: res.metrics for name, res in results.items()}
    return pd.DataFrame(rows).T
