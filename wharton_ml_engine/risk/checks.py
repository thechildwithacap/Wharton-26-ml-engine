"""Operational / mandate risk pods (PRD 6.2).

Turnover & transaction cost, model robustness, data quality, and the
compliance / mandate gate.  These act as independent checks that can veto,
scale back, or freeze trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..client.profile import ClientProfile
from ..config import STYLES, CostModel, PortfolioConstraints
from ..data.source import DataBundle
from ..utils import clip_score


# ------------------------------------------------------------------ turnover
@dataclass
class TurnoverResult:
    turnover: float          # one-way, fraction of book
    cost_estimate: float     # fraction of book consumed by costs
    turnover_score: float    # 0-100, higher = less turnover

    def as_dict(self) -> Dict[str, float]:
        return {"turnover": self.turnover, "cost_estimate": self.cost_estimate,
                "turnover_score": self.turnover_score}


def turnover_and_cost(current: Dict[str, float], target: Dict[str, float],
                      costs: CostModel) -> TurnoverResult:
    tickers = set(current) | set(target)
    total_abs = sum(abs(target.get(t, 0.0) - current.get(t, 0.0)) for t in tickers)
    one_way = total_abs / 2.0                        # buys == sells for a fully-invested book
    cost = total_abs * costs.cost_fraction()         # both legs pay cost
    # 0 turnover -> 100; 50% one-way turnover -> ~0.
    score = clip_score(pd.Series([100.0 * (1.0 - one_way / 0.5)])).iloc[0]
    return TurnoverResult(turnover=one_way, cost_estimate=cost, turnover_score=float(score))


# ---------------------------------------------------------------- robustness
def model_robustness_scores(signals: pd.DataFrame) -> Dict[str, float]:
    """Cheap cross-sectional robustness proxy per style.

    Rewards good coverage (few missing values) and healthy dispersion (a signal
    that is flat across the universe carries little information and is
    downweighted).  The time-series stability check lives in the backtest layer.
    """
    scores: Dict[str, float] = {}
    n = len(signals)
    for s in STYLES:
        if s not in signals.columns or n == 0:
            scores[s] = 50.0
            continue
        col = signals[s]
        coverage = float(col.notna().mean())
        dispersion = float(col.std(ddof=0))
        disp_score = min(dispersion / 20.0, 1.0)     # ~20-pt std considered healthy
        scores[s] = float(clip_score(pd.Series([100.0 * (0.5 * coverage + 0.5 * disp_score)])).iloc[0])
    return scores


# -------------------------------------------------------------- data quality
@dataclass
class DataQualityReport:
    status: str                       # OK | Warning | Critical
    price_missing_frac: float
    stale_tickers: List[str] = field(default_factory=list)
    outlier_tickers: List[str] = field(default_factory=list)
    fundamentals_coverage: float = 1.0
    notes: List[str] = field(default_factory=list)

    @property
    def freezes_trading(self) -> bool:
        return self.status == "Critical"


def data_quality_report(bundle: DataBundle, as_of: pd.Timestamp,
                        window: int = 60) -> DataQualityReport:
    px = bundle.prices_upto(as_of).tail(window)
    notes: List[str] = []

    missing = float(px.isna().mean().mean()) if px.size else 1.0

    # Stale: a series that has not moved for the whole window.
    stale = [t for t in px.columns if px[t].dropna().nunique() <= 1 and len(px[t].dropna()) > 1]

    # Outliers: any single-day return beyond +/-50% (likely bad tick / unadjusted).
    rets = px.pct_change()
    outliers = [t for t in px.columns if (rets[t].abs() > 0.5).any()]

    fund = bundle.fundamentals_asof(as_of)
    fcov = float(fund.notna().mean().mean()) if len(fund) else 0.0

    status = "OK"
    if missing > 0.02 or stale or outliers or fcov < 0.9:
        status = "Warning"
    if missing > 0.10 or fcov < 0.6 or len(stale) > 0.2 * max(len(px.columns), 1):
        status = "Critical"
    if stale:
        notes.append(f"{len(stale)} stale price series")
    if outliers:
        notes.append(f"{len(outliers)} tickers with extreme daily moves")
    if missing > 0.02:
        notes.append(f"{missing:.1%} missing prices in window")

    return DataQualityReport(status=status, price_missing_frac=missing,
                             stale_tickers=stale, outlier_tickers=outliers,
                             fundamentals_coverage=fcov, notes=notes)


# ----------------------------------------------------------------- mandate
@dataclass
class MandateCheck:
    passed: bool
    violations: List[str] = field(default_factory=list)


def mandate_check(weights: Dict[str, float], bundle: DataBundle,
                  profile: ClientProfile, constraints: PortfolioConstraints,
                  tol: float = 1e-6) -> MandateCheck:
    violations: List[str] = []
    w = {t: v for t, v in weights.items() if abs(v) > tol}

    # No shorting / leverage.
    if not constraints.allow_shorting and any(v < -tol for v in w.values()):
        violations.append("short position present but shorting is disallowed")
    total = sum(w.values())
    if not constraints.allow_leverage and total > 1.0 + 1e-4:
        violations.append(f"gross exposure {total:.2f} > 1.0 without leverage")

    # Holdings count.
    n = len(w)
    if n < constraints.min_holdings:
        violations.append(f"{n} holdings < min {constraints.min_holdings}")
    if n > constraints.max_holdings:
        violations.append(f"{n} holdings > max {constraints.max_holdings}")

    # Position caps.
    for t, v in w.items():
        if v > constraints.max_weight_per_stock + 1e-6:
            violations.append(f"{t} weight {v:.1%} > cap {constraints.max_weight_per_stock:.1%}")

    # Sector caps.
    sec_exp: Dict[str, float] = {}
    for t, v in w.items():
        sec = bundle.meta[t].sector
        sec_exp[sec] = sec_exp.get(sec, 0.0) + v
    for sec, exp in sec_exp.items():
        if exp > constraints.max_weight_per_sector + 1e-6:
            violations.append(f"sector {sec} {exp:.1%} > cap {constraints.max_weight_per_sector:.1%}")

    # Client exclusions must not appear.
    avoid = set(profile.sector_avoidances)
    for t in w:
        m = bundle.meta[t]
        if m.sector in avoid:
            violations.append(f"{t} in excluded sector {m.sector}")
        if profile.excluded_themes and set(m.themes) & set(profile.excluded_themes):
            violations.append(f"{t} carries an excluded theme")

    return MandateCheck(passed=not violations, violations=violations)
