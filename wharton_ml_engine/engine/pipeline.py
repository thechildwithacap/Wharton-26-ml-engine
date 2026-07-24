"""Main engine orchestration (PRD section 7).

``run_engine`` wires every layer together for a single decision date and
returns an :class:`EngineReport` that carries all intermediate outputs, so the
recommendation is fully auditable end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from ..backtest.engine import BacktestResult, backtest_all_templates, templates_summary
from ..backtest.regime import RegimeState
from ..backtest.stability import SignalStability, signal_stability
from ..backtest.style_weights import StyleRecommendation, recommend_style_weights
from ..client.fit import compute_client_fit
from ..client.profile import ClientProfile
from ..config import EngineConfig
from ..data.source import DataBundle
from ..risk.checks import (
    DataQualityReport,
    MandateCheck,
    TurnoverResult,
    data_quality_report,
    mandate_check,
    model_robustness_scores,
    turnover_and_cost,
)
from ..risk.portfolio import (
    PortfolioRiskSummary,
    aggregate_portfolio_risk,
    concentration_alerts,
    crowding_model,
)
from ..risk.stock import liquidity_model, stock_risk_model
from ..signals import compute_signals
from .construct import ConstructionResult, construct_portfolio
from .integrate import integrate_scores
from .trade import TradeDecision, decide_trades


@dataclass
class EngineReport:
    as_of: pd.Timestamp
    profile: ClientProfile
    config: EngineConfig

    regime: RegimeState
    style_recommendation: StyleRecommendation
    signal_confidence: float
    signal_stability: SignalStability

    signals: pd.DataFrame
    client_fit: pd.DataFrame
    stock_risk: pd.DataFrame
    liquidity: pd.DataFrame
    robustness: Dict[str, float]
    scored: pd.DataFrame

    construction: ConstructionResult
    candidate_weights: Dict[str, float]
    candidate_risk: PortfolioRiskSummary
    crowding: Dict[str, object]
    concentration_alerts: List[str]

    turnover: TurnoverResult
    mandate: MandateCheck
    data_quality: DataQualityReport
    decision: TradeDecision

    templates: Optional[Dict[str, BacktestResult]] = None
    templates_summary: Optional[pd.DataFrame] = field(default=None)
    ml_metrics: Optional[Dict[str, float]] = None

    # ---- convenience views --------------------------------------------------
    def ranked_table(self, top: int = 30) -> pd.DataFrame:
        cols = [c for c in ["integrated_score", "style_score", "client_fit",
                            "risk_score", "liquidity_score"] if c in self.scored.columns]
        return self.scored[cols].head(top)

    def portfolio_table(self) -> pd.DataFrame:
        w = pd.Series(self.candidate_weights, name="weight").sort_values(ascending=False)
        df = w.to_frame()
        df["sector"] = [self._sector(t) for t in df.index]
        if "integrated_score" in self.scored.columns:
            df["integrated_score"] = self.scored["integrated_score"].reindex(df.index)
        return df

    _sector_map: Dict[str, str] = field(default_factory=dict)

    def _sector(self, t: str) -> str:
        return self._sector_map.get(t, "")


def run_engine(
    bundle: DataBundle,
    profile: ClientProfile,
    config: Optional[EngineConfig] = None,
    as_of: Optional[pd.Timestamp] = None,
    current_weights: Optional[Dict[str, float]] = None,
    analyst_ratings: Optional[Dict[str, Dict[str, float]]] = None,
    alpha_model: Optional[object] = None,
    run_backtest: bool = True,
    backtest_lookback_days: int = 1000,
    portfolio_value: float = 1_000_000.0,
) -> EngineReport:
    config = config or EngineConfig()
    constraints = profile.apply_to_constraints(config.constraints)
    if as_of is None:
        as_of = bundle.dates()[-1]
    as_of = pd.Timestamp(as_of)
    current_weights = current_weights or {}

    # --- environment: data quality, regime, backtests, style weights ---------
    dq = data_quality_report(bundle, as_of)

    templates: Optional[Dict[str, BacktestResult]] = None
    tsummary: Optional[pd.DataFrame] = None
    if run_backtest:
        dates = bundle.dates()
        start_idx = max(0, len(dates[dates <= as_of]) - backtest_lookback_days)
        bt_start = dates[start_idx]
        templates = backtest_all_templates(bundle, bt_start, as_of,
                                           top_n=min(25, len(bundle.tickers)))
        tsummary = templates_summary(templates)

    style_rec = recommend_style_weights(bundle, profile, as_of, templates)
    stability = signal_stability(bundle, as_of, style_rec.weights)

    # Regime-aware sizing: flatten conviction for contrarian regimes, concentrate
    # for quality-compounder trends (Task C).
    if config.regime_aware_sizing:
        from .construct import regime_sizing_posture
        posture = regime_sizing_posture(
            style_rec.regime.trend, style_rec.regime.vol_state,
            constraints.concentration, constraints.max_weight_per_stock)
        constraints.concentration = posture["concentration"]
        constraints.max_weight_per_stock = posture["max_weight"]

    # --- signal, client, risk layers -----------------------------------------
    signals = compute_signals(bundle, profile, as_of, analyst_ratings, alpha_model)
    if config.sector_neutral:
        from ..config import STYLES
        from ..utils import sector_neutralize
        cols = [c for c in STYLES if c in signals.columns]
        signals = sector_neutralize(signals, bundle.sectors, cols)
    fit = compute_client_fit(bundle, profile, as_of)
    srisk = stock_risk_model(bundle, as_of)
    liq = liquidity_model(bundle, as_of, constraints, portfolio_value)
    robustness = model_robustness_scores(signals)
    market_cap = bundle.fundamentals_asof(as_of).get("market_cap")  # for benchmark tilt

    # --- integration & construction (with a crowding-aware second pass) -------
    scored = integrate_scores(signals, fit, srisk, liq, style_rec.weights,
                              profile, robustness)
    if config.universe_filter is not None:
        eligible = set(bundle.eligible_asof(as_of, config.universe_filter))
        scored.loc[[t for t in scored.index if t not in eligible], "integrated_score"] = float("nan")
    construction = construct_portfolio(scored, bundle, constraints, liq, market_cap)
    candidate = construction.weights

    crowd = crowding_model(
        candidate, signals,
        aggregate_portfolio_risk(candidate, bundle, signals, as_of),
    )
    if crowd["de_risk_styles"]:
        scored = integrate_scores(signals, fit, srisk, liq, style_rec.weights,
                                  profile, robustness,
                                  de_risk_styles=crowd["de_risk_styles"])
        construction = construct_portfolio(scored, bundle, constraints, liq, market_cap)
        candidate = construction.weights

    # --- candidate risk & checks ---------------------------------------------
    cand_risk = aggregate_portfolio_risk(candidate, bundle, signals, as_of)
    curr_risk = (aggregate_portfolio_risk(current_weights, bundle, signals, as_of)
                 if current_weights else None)
    alerts = concentration_alerts(cand_risk, constraints)
    turnover = turnover_and_cost(current_weights, candidate, config.costs)
    mandate = mandate_check(candidate, bundle, profile, constraints)

    decision = decide_trades(
        current_weights, candidate, scored, config, turnover, mandate, dq,
        stability.confidence, curr_risk, cand_risk,
    )

    report = EngineReport(
        as_of=as_of, profile=profile, config=config, regime=style_rec.regime,
        style_recommendation=style_rec, signal_confidence=stability.confidence,
        signal_stability=stability, signals=signals, client_fit=fit,
        stock_risk=srisk, liquidity=liq, robustness=robustness, scored=scored,
        construction=construction, candidate_weights=candidate,
        candidate_risk=cand_risk, crowding=crowd, concentration_alerts=alerts,
        turnover=turnover, mandate=mandate, data_quality=dq, decision=decision,
        templates=templates, templates_summary=tsummary,
        ml_metrics=(dict(alpha_model.metrics) if alpha_model is not None else None),
    )
    report._sector_map = bundle.sectors
    return report
