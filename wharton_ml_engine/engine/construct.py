"""Portfolio Construction (PRD 7.2).

Selects ~20-30 eligible names and assigns weights that respect every hard
constraint: position caps, sector caps, per-name liquidity caps and the
min/max holdings count.  Uses transparent score-proportional weighting followed
by iterative "water-filling" to enforce the caps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import PortfolioConstraints
from ..data.source import DataBundle


@dataclass
class ConstructionResult:
    weights: Dict[str, float]
    log: List[str] = field(default_factory=list)
    selected: List[str] = field(default_factory=list)


def _select_candidates(scored: pd.DataFrame, sectors: Dict[str, str],
                       constraints: PortfolioConstraints) -> List[str]:
    """Pick up to ``max_holdings`` names, diversifying by sector as we go."""
    ranked = scored[scored["integrated_score"].notna()].sort_values(
        "integrated_score", ascending=False)
    # Soft per-sector count cap keeps any one sector from dominating selection.
    max_per_sector = max(2, math.ceil(constraints.max_weight_per_sector *
                                      constraints.max_holdings) + 1)
    sector_count: Dict[str, int] = {}
    picks: List[str] = []
    for t in ranked.index:
        sec = sectors.get(t, "Unknown")
        if sector_count.get(sec, 0) >= max_per_sector:
            continue
        picks.append(t)
        sector_count[sec] = sector_count.get(sec, 0) + 1
        if len(picks) >= constraints.max_holdings:
            break
    return picks


def _cap_vector(names: List[str], liquidity: Optional[pd.DataFrame],
                constraints: PortfolioConstraints) -> pd.Series:
    caps = pd.Series(constraints.max_weight_per_stock, index=names)
    if liquidity is not None and "max_weight_by_liquidity" in liquidity.columns:
        liq_cap = liquidity["max_weight_by_liquidity"].reindex(names)
        caps = pd.concat([caps, liq_cap], axis=1).min(axis=1)
    return caps.clip(lower=constraints.min_weight_per_stock)


def _apply_position_caps(w: pd.Series, caps: pd.Series, iters: int = 50) -> pd.Series:
    """Water-fill so no weight exceeds its cap while the book stays fully invested."""
    w = w.clip(lower=0.0)
    for _ in range(iters):
        w = w / w.sum()
        over = w > caps + 1e-12
        if not over.any():
            break
        excess = (w[over] - caps[over]).sum()
        w[over] = caps[over]
        under = ~over
        if not under.any() or w[under].sum() == 0:
            break
        w[under] += excess * (w[under] / w[under].sum())
    return w / w.sum()


def _apply_sector_caps(w: pd.Series, sectors: Dict[str, str], caps: pd.Series,
                       constraints: PortfolioConstraints, iters: int = 50) -> pd.Series:
    for _ in range(iters):
        w = w / w.sum()
        sec_w = w.groupby(w.index.map(lambda t: sectors.get(t, "Unknown"))).sum()
        over_secs = sec_w[sec_w > constraints.max_weight_per_sector + 1e-9]
        if over_secs.empty:
            break
        for sec, total in over_secs.items():
            members = [t for t in w.index if sectors.get(t) == sec]
            scale = constraints.max_weight_per_sector / total
            freed = w[members].sum() * (1 - scale)
            w[members] *= scale
            # Redistribute to under-cap names with headroom.
            others = [t for t in w.index if sectors.get(t) != sec]
            if others and w[others].sum() > 0:
                headroom = (caps[others] - w[others]).clip(lower=0)
                if headroom.sum() > 0:
                    w[others] += freed * (headroom / headroom.sum())
        w = w / w.sum()
    # Re-apply position caps in case redistribution pushed a name over.
    return _apply_position_caps(w, caps)


def construct_portfolio(
    scored: pd.DataFrame,
    bundle: DataBundle,
    constraints: PortfolioConstraints,
    liquidity: Optional[pd.DataFrame] = None,
) -> ConstructionResult:
    log: List[str] = []
    sectors = bundle.sectors

    picks = _select_candidates(scored, sectors, constraints)
    if len(picks) < constraints.min_holdings:
        log.append(f"only {len(picks)} eligible names for min {constraints.min_holdings}; "
                   "portfolio will be smaller than target")
    if not picks:
        return ConstructionResult(weights={}, log=["no eligible candidates"], selected=[])

    scores = scored.loc[picks, "integrated_score"]
    # Score-proportional base weights above a floor so the tilt is meaningful
    # but not winner-take-all.
    base = (scores - scores.min() + 5.0)
    w = base / base.sum()

    caps = _cap_vector(picks, liquidity, constraints)
    w = _apply_position_caps(w, caps)
    w = _apply_sector_caps(w, sectors, caps, constraints)

    # Drop sub-minimum positions and renormalise; keep within the holdings band.
    keep = w[w >= constraints.min_weight_per_stock]
    if len(keep) > constraints.max_holdings:
        keep = keep.sort_values(ascending=False).head(constraints.max_holdings)
    if len(keep) >= constraints.min_holdings:
        w = keep
    w = w / w.sum()
    # Final cap pass after trimming.
    w = _apply_position_caps(w, caps.reindex(w.index).fillna(constraints.max_weight_per_stock))
    w = _apply_sector_caps(w, sectors,
                           caps.reindex(w.index).fillna(constraints.max_weight_per_stock),
                           constraints)

    log.append(f"selected {len(w)} holdings; top weight {w.max():.1%}; "
               f"largest sector "
               f"{w.groupby(w.index.map(lambda t: sectors.get(t))).sum().max():.1%}")
    return ConstructionResult(weights=w.to_dict(), log=log, selected=list(w.index))
