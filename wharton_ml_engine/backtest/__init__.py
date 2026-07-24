"""Backtesting & Regime layer (PRD 6.3)."""

from .engine import (
    BacktestResult,
    MeasuredSurvivorshipBias,
    backtest_all_templates,
    backtest_template,
    backtest_template_pit,
    benchmark_returns,
    measure_survivorship_bias,
    templates_summary,
)
from .regime import RegimeState, classify_regime
from .stability import SignalStability, signal_stability
from .style_weights import StyleRecommendation, recommend_style_weights
from .templates import (
    STRATEGY_TEMPLATES,
    normalise_weights,
    style_score_panel,
    template_score,
)

__all__ = [
    "STRATEGY_TEMPLATES",
    "style_score_panel",
    "template_score",
    "normalise_weights",
    "BacktestResult",
    "backtest_template",
    "backtest_template_pit",
    "backtest_all_templates",
    "measure_survivorship_bias",
    "MeasuredSurvivorshipBias",
    "benchmark_returns",
    "templates_summary",
    "RegimeState",
    "classify_regime",
    "StyleRecommendation",
    "recommend_style_weights",
    "SignalStability",
    "signal_stability",
]
