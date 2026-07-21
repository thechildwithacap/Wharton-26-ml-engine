"""Trading Decision Logic (PRD 7.2).

Compares the current book with a candidate book and approves a rebalance only
when *all* gates pass:

* expected integrated-score improvement (net of transaction costs) clears the
  hurdle;
* portfolio risk does not materially worsen;
* signal confidence is high enough;
* the mandate check passes and data quality is not Critical.

Otherwise it holds.  Always returns an explicit, auditable trade list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import EngineConfig
from ..risk.checks import DataQualityReport, MandateCheck, TurnoverResult
from ..risk.portfolio import PortfolioRiskSummary


@dataclass
class TradeDecision:
    approved: bool
    action: str                      # "rebalance" | "hold"
    net_score_improvement: float
    trades: pd.DataFrame
    reasons: List[str] = field(default_factory=list)
    checks: Dict[str, bool] = field(default_factory=dict)


def _book_score(weights: Dict[str, float], scores: pd.Series) -> float:
    if not weights:
        return 0.0
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    aligned = scores.reindex(w.index).fillna(scores.median())
    return float((w * aligned).sum())


def build_trade_list(current: Dict[str, float], candidate: Dict[str, float],
                     scored: pd.DataFrame) -> pd.DataFrame:
    tickers = sorted(set(current) | set(candidate))
    rows = []
    for t in tickers:
        cw, nw = current.get(t, 0.0), candidate.get(t, 0.0)
        delta = nw - cw
        if abs(delta) < 1e-6:
            action = "hold"
        elif cw == 0:
            action = "add"
        elif nw == 0:
            action = "exit"
        elif delta > 0:
            action = "increase"
        else:
            action = "trim"
        rows.append({
            "ticker": t,
            "current_weight": cw,
            "target_weight": nw,
            "delta": delta,
            "action": action,
            "integrated_score": float(scored["integrated_score"].get(t, np.nan))
            if "integrated_score" in scored.columns else np.nan,
        })
    columns = ["current_weight", "target_weight", "delta", "action", "integrated_score"]
    if not rows:
        return pd.DataFrame(columns=columns, index=pd.Index([], name="ticker"))
    df = pd.DataFrame(rows).set_index("ticker")
    return df.sort_values("delta", key=lambda s: s.abs(), ascending=False)


def decide_trades(
    current_weights: Dict[str, float],
    candidate_weights: Dict[str, float],
    scored: pd.DataFrame,
    config: EngineConfig,
    turnover: TurnoverResult,
    mandate: MandateCheck,
    data_quality: DataQualityReport,
    signal_confidence: float,
    current_risk: Optional[PortfolioRiskSummary] = None,
    candidate_risk: Optional[PortfolioRiskSummary] = None,
) -> TradeDecision:
    scores = scored["integrated_score"] if "integrated_score" in scored.columns else pd.Series(dtype=float)
    reasons: List[str] = []
    checks: Dict[str, bool] = {}

    # --- expected improvement, net of costs (cost as score-equivalent points).
    raw_improvement = _book_score(candidate_weights, scores) - _book_score(current_weights, scores)
    cost_points = turnover.cost_estimate * 100.0
    net_improvement = raw_improvement - cost_points

    is_initial = not current_weights
    if is_initial:
        checks["improvement"] = True
        reasons.append("initial construction (no existing book)")
    else:
        checks["improvement"] = net_improvement >= config.min_score_improvement_to_trade
        reasons.append(
            f"net score improvement {net_improvement:+.2f} "
            f"(raw {raw_improvement:+.2f} - cost {cost_points:.2f}); "
            f"hurdle {config.min_score_improvement_to_trade:.2f}"
        )

    # --- mandate & data quality are hard gates.
    checks["mandate"] = mandate.passed
    if not mandate.passed:
        reasons.append("mandate FAIL: " + "; ".join(mandate.violations[:4]))

    checks["data_quality"] = not data_quality.freezes_trading
    if data_quality.freezes_trading:
        reasons.append("data quality Critical: new trades frozen")

    # --- signal confidence.
    checks["signal_confidence"] = signal_confidence >= config.min_signal_confidence
    reasons.append(f"signal confidence {signal_confidence:.2f} "
                   f"(floor {config.min_signal_confidence:.2f})")

    # --- risk not materially worse than current.
    risk_ok = True
    if current_risk and candidate_risk and not is_initial:
        vol_worse = (candidate_risk.volatility - current_risk.volatility) > 0.02
        beta_worse = (candidate_risk.beta - current_risk.beta) > 0.15
        if vol_worse or beta_worse:
            risk_ok = False
            reasons.append(
                f"risk worsens (vol {current_risk.volatility:.2f}->"
                f"{candidate_risk.volatility:.2f}, beta {current_risk.beta:.2f}->"
                f"{candidate_risk.beta:.2f})")
    checks["risk"] = risk_ok

    # --- turnover cap: exceeding it blocks a full rebalance (initial build is
    #     exempt — there is no existing book to preserve).
    if is_initial:
        checks["turnover"] = True
    else:
        checks["turnover"] = turnover.turnover <= config.constraints.max_turnover_per_rebalance
        if not checks["turnover"]:
            reasons.append(f"turnover {turnover.turnover:.1%} exceeds cap "
                           f"{config.constraints.max_turnover_per_rebalance:.1%}")

    approved = all(checks.values())
    trades = build_trade_list(current_weights, candidate_weights, scored)
    if not approved:
        # On hold, the actionable trade list is empty (keep current book).
        held = build_trade_list(current_weights, current_weights, scored)
        return TradeDecision(False, "hold", net_improvement, held, reasons, checks)
    return TradeDecision(True, "rebalance", net_improvement, trades, reasons, checks)
