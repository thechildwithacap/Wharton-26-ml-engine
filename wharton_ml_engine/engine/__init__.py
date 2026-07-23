"""Integration layer — the main portfolio engine (PRD section 7)."""

from .backtest_portfolio import (
    ComparisonResult,
    StrategyResult,
    backtest_ml_vs_rules,
    format_comparison,
)
from .construct import ConstructionResult, construct_portfolio
from .integrate import integrate_scores
from .paper import (
    PaperPortfolio,
    PaperStepResult,
    paper_trade_step,
    performance_summary,
)
from .pipeline import EngineReport, run_engine
from .trade import TradeDecision, build_trade_list, decide_trades

__all__ = [
    "integrate_scores",
    "ConstructionResult",
    "construct_portfolio",
    "TradeDecision",
    "build_trade_list",
    "decide_trades",
    "EngineReport",
    "run_engine",
    "backtest_ml_vs_rules",
    "ComparisonResult",
    "StrategyResult",
    "format_comparison",
    "PaperPortfolio",
    "PaperStepResult",
    "paper_trade_step",
    "performance_summary",
]
