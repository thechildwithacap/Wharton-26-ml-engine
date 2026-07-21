"""Integration layer — the main portfolio engine (PRD section 7)."""

from .construct import ConstructionResult, construct_portfolio
from .integrate import integrate_scores
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
]
