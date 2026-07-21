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


def templates_summary(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    rows = {name: res.metrics for name, res in results.items()}
    return pd.DataFrame(rows).T
