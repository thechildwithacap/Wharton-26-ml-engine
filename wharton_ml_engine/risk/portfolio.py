"""Portfolio-level risk aggregation, concentration and crowding (PRD 6.2).

Given a set of weights (ticker -> weight), these functions measure the
book-level risk the construction and trade-decision layers must respect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import STYLES, PortfolioConstraints
from ..data.source import DataBundle


@dataclass
class PortfolioRiskSummary:
    volatility: float
    beta: float
    top_weight: float
    top5_weight: float
    herfindahl: float
    effective_n: float
    sector_exposure: Dict[str, float] = field(default_factory=dict)
    style_exposure: Dict[str, float] = field(default_factory=dict)
    n_holdings: int = 0

    def as_dict(self) -> Dict[str, float]:
        d = {
            "volatility": self.volatility,
            "beta": self.beta,
            "top_weight": self.top_weight,
            "top5_weight": self.top5_weight,
            "herfindahl": self.herfindahl,
            "effective_n": self.effective_n,
            "n_holdings": self.n_holdings,
        }
        d.update({f"sector::{k}": v for k, v in self.sector_exposure.items()})
        return d


def _weights_series(weights: Dict[str, float]) -> pd.Series:
    s = pd.Series(weights, dtype=float)
    return s[s > 0]


def concentration_metrics(weights: Dict[str, float],
                          sectors: Dict[str, str]) -> Dict[str, float]:
    w = _weights_series(weights)
    if w.empty:
        return {"top_weight": 0.0, "top5_weight": 0.0, "herfindahl": 0.0,
                "effective_n": 0.0, "n_holdings": 0}
    ordered = w.sort_values(ascending=False)
    hhi = float((w ** 2).sum())
    return {
        "top_weight": float(ordered.iloc[0]),
        "top5_weight": float(ordered.head(5).sum()),
        "herfindahl": hhi,
        "effective_n": float(1.0 / hhi) if hhi > 0 else 0.0,
        "n_holdings": int(len(w)),
    }


def sector_exposure(weights: Dict[str, float], sectors: Dict[str, str]) -> Dict[str, float]:
    w = _weights_series(weights)
    exp: Dict[str, float] = {}
    for t, wt in w.items():
        sec = sectors.get(t, "Unknown")
        exp[sec] = exp.get(sec, 0.0) + float(wt)
    return exp


def style_exposure(weights: Dict[str, float], signals: pd.DataFrame) -> Dict[str, float]:
    """Weight-average of each style score across the book (0-100)."""
    w = _weights_series(weights)
    w = w / w.sum()
    exp = {}
    for s in STYLES:
        if s in signals.columns:
            aligned = signals[s].reindex(w.index).fillna(50.0)
            exp[s] = float((w * aligned).sum())
    return exp


def aggregate_portfolio_risk(
    weights: Dict[str, float],
    bundle: DataBundle,
    signals: pd.DataFrame,
    as_of: pd.Timestamp,
    window: int = 252,
) -> PortfolioRiskSummary:
    w = _weights_series(weights)
    if w.empty:
        return PortfolioRiskSummary(0, 0, 0, 0, 0, 0, {}, {}, 0)
    w = w / w.sum()

    px = bundle.prices_upto(as_of).tail(window + 1)
    rets = px[list(w.index)].pct_change().dropna(how="all")
    port_ret = (rets * w).sum(axis=1)
    vol = float(port_ret.std(ddof=0) * np.sqrt(252)) if len(port_ret) > 2 else float("nan")

    beta = float("nan")
    if "SPX" in bundle.benchmarks.columns and len(port_ret) > 20:
        mkt = bundle.benchmarks["SPX"].reindex(px.index).pct_change().reindex(port_ret.index)
        var = mkt.var(ddof=0)
        beta = float(port_ret.cov(mkt) / var) if var else float("nan")

    conc = concentration_metrics(weights, bundle.sectors)
    return PortfolioRiskSummary(
        volatility=vol,
        beta=beta,
        top_weight=conc["top_weight"],
        top5_weight=conc["top5_weight"],
        herfindahl=conc["herfindahl"],
        effective_n=conc["effective_n"],
        sector_exposure=sector_exposure(weights, bundle.sectors),
        style_exposure=style_exposure(weights, signals),
        n_holdings=conc["n_holdings"],
    )


def concentration_alerts(summary: PortfolioRiskSummary,
                         constraints: PortfolioConstraints) -> List[str]:
    alerts: List[str] = []
    if summary.n_holdings < constraints.min_holdings:
        alerts.append(f"only {summary.n_holdings} holdings (< {constraints.min_holdings})")
    if summary.n_holdings > constraints.max_holdings:
        alerts.append(f"{summary.n_holdings} holdings (> {constraints.max_holdings})")
    if summary.top_weight > constraints.max_weight_per_stock + 1e-9:
        alerts.append(f"top position {summary.top_weight:.1%} exceeds cap "
                      f"{constraints.max_weight_per_stock:.1%}")
    for sec, exp in summary.sector_exposure.items():
        if exp > constraints.max_weight_per_sector + 1e-9:
            alerts.append(f"sector {sec} at {exp:.1%} exceeds cap "
                          f"{constraints.max_weight_per_sector:.1%}")
    return alerts


def crowding_model(weights: Dict[str, float], signals: pd.DataFrame,
                   summary: PortfolioRiskSummary) -> Dict[str, object]:
    """Detect extreme factor tilts vulnerable to reversal (PRD 6.2)."""
    flags: List[str] = []
    de_risk: List[str] = []
    for s in STYLES:
        exp = summary.style_exposure.get(s)
        if exp is None:
            continue
        # A book averaging > 70 on a single style is crowded into it.
        if exp > 70.0:
            flags.append(f"crowded in {s} (avg {exp:.0f})")
            de_risk.append(s)
    momentum_exp = summary.style_exposure.get("momentum", 50.0)
    crowding_score = float(np.clip(100 - max(0.0, momentum_exp - 50) * 2, 0, 100))
    return {"crowding_flags": flags, "de_risk_styles": de_risk,
            "crowding_score": crowding_score}
