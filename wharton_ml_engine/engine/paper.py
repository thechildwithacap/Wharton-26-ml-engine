"""Paper-trading portfolio — a simulated book the team maintains over time.

The Wharton competition *is* a paper-trading simulation (WInS): you hold a
virtual portfolio, it's marked to market as prices move, and you rebalance on
your own cadence.  This module mirrors that workflow — it is **decision support,
not an auto-executing bot**: it tracks a simulated book, applies the engine's
recommendations when you choose to, and records performance vs a benchmark so
the process is auditable for the IPS / Final Report.

State is plain JSON, so a team can run one "trading day" at a time (as new data
arrives) and keep a persistent, reproducible track record.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..client.profile import ClientProfile
from ..config import EngineConfig
from ..data.source import DataBundle
from .pipeline import run_engine


@dataclass
class PaperPortfolio:
    """A simulated portfolio: cash + fractional-share positions + a NAV history."""

    cash: float
    initial_capital: float
    positions: Dict[str, float] = field(default_factory=dict)   # ticker -> shares
    history: List[dict] = field(default_factory=list)           # marks over time
    benchmark_index: str = "SPX"
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ---- valuation ----------------------------------------------------------
    def holdings_value(self, prices: pd.Series) -> float:
        return float(sum(sh * float(prices.get(t, np.nan)) for t, sh in self.positions.items()
                         if np.isfinite(prices.get(t, np.nan))))

    def nav(self, prices: pd.Series) -> float:
        return self.cash + self.holdings_value(prices)

    def weights(self, prices: pd.Series) -> Dict[str, float]:
        nav = self.nav(prices)
        if nav <= 0:
            return {}
        return {t: sh * float(prices.get(t, 0.0)) / nav
                for t, sh in self.positions.items() if sh > 0}

    # ---- trading ------------------------------------------------------------
    def rebalance_to(self, target: Dict[str, float], prices: pd.Series,
                     cost_fraction: float) -> dict:
        """Move the book to ``target`` weights; charge proportional costs.

        Fractional shares are allowed (this is a simulation).  Returns a small
        trade summary (turnover, cost, number of trades).
        """
        nav = self.nav(prices)
        if nav <= 0:
            return {"turnover": 0.0, "cost": 0.0, "n_trades": 0}
        tradable = {t: w for t, w in target.items()
                    if t in prices.index and np.isfinite(prices[t]) and prices[t] > 0}
        tot = sum(tradable.values()) or 1.0
        traded_notional = 0.0
        n_trades = 0
        new_positions: Dict[str, float] = {}
        # Target dollars per name (weights renormalised over tradable names).
        for t, w in tradable.items():
            tgt_dollars = nav * (w / tot)
            tgt_shares = tgt_dollars / float(prices[t])
            cur_shares = self.positions.get(t, 0.0)
            if abs(tgt_shares - cur_shares) * float(prices[t]) > 1e-6:
                n_trades += 1
                traded_notional += abs(tgt_shares - cur_shares) * float(prices[t])
            new_positions[t] = tgt_shares
        # Names dropped entirely (target 0) are sold.
        for t, sh in self.positions.items():
            if t not in new_positions and sh != 0:
                traded_notional += abs(sh) * float(prices.get(t, 0.0))
                n_trades += 1
        cost = traded_notional * cost_fraction
        self.positions = {t: sh for t, sh in new_positions.items() if abs(sh) > 1e-12}
        self.cash = nav - self.holdings_value(prices) - cost
        return {"turnover": round(traded_notional / nav, 4), "cost": round(cost, 2),
                "n_trades": n_trades}

    def mark(self, date, prices: pd.Series, benchmark: Optional[float] = None,
             note: str = "") -> dict:
        rec = {"date": str(pd.Timestamp(date).date()), "nav": round(self.nav(prices), 2),
               "cash": round(self.cash, 2), "n_positions": len(self.positions),
               "benchmark": (round(float(benchmark), 4) if benchmark is not None else None),
               "note": note}
        self.history.append(rec)
        return rec

    # ---- persistence --------------------------------------------------------
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump({
                "cash": self.cash, "initial_capital": self.initial_capital,
                "positions": self.positions, "history": self.history,
                "benchmark_index": self.benchmark_index, "created": self.created,
            }, fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "PaperPortfolio":
        with open(path) as fh:
            d = json.load(fh)
        return cls(cash=d["cash"], initial_capital=d["initial_capital"],
                   positions={k: float(v) for k, v in d.get("positions", {}).items()},
                   history=d.get("history", []),
                   benchmark_index=d.get("benchmark_index", "SPX"),
                   created=d.get("created", ""))

    @classmethod
    def new(cls, capital: float = 100_000.0, benchmark_index: str = "SPX") -> "PaperPortfolio":
        return cls(cash=capital, initial_capital=capital, benchmark_index=benchmark_index)


@dataclass
class PaperStepResult:
    date: pd.Timestamp
    nav: float
    action: str                 # "rebalance" | "hold"
    trade: dict
    decision_reasons: List[str]
    report: object              # EngineReport

    def summary(self) -> str:
        return (f"[{self.date.date()}] NAV ${self.nav:,.0f}  ·  {self.action.upper()}"
                f"  ·  {self.trade.get('n_trades', 0)} trades"
                f"  ·  turnover {self.trade.get('turnover', 0):.0%}")


def paper_trade_step(
    bundle: DataBundle,
    profile: ClientProfile,
    portfolio: PaperPortfolio,
    as_of=None,
    alpha_model: Optional[object] = None,
    config: Optional[EngineConfig] = None,
    force_rebalance: bool = False,
) -> PaperStepResult:
    """Run one paper-trading step: mark, ask the engine, act on its decision.

    Marks the book to market at ``as_of``, runs the engine with the current book
    as the incumbent, and — if the engine approves a rebalance (or
    ``force_rebalance``) — moves the book to the recommended weights.  Records a
    mark to the history either way.
    """
    config = config or EngineConfig()
    as_of = bundle.dates()[-1] if as_of is None else pd.Timestamp(as_of)
    prices = bundle.prices.loc[bundle.prices.index <= as_of].iloc[-1]

    current = portfolio.weights(prices)
    report = run_engine(bundle, profile, config=config, as_of=as_of,
                        current_weights=current, alpha_model=alpha_model,
                        run_backtest=False)

    do_trade = force_rebalance or report.decision.approved
    trade = {"turnover": 0.0, "cost": 0.0, "n_trades": 0}
    if do_trade and report.candidate_weights:
        trade = portfolio.rebalance_to(report.candidate_weights, prices,
                                       config.costs.cost_fraction())
        action = "rebalance"
    else:
        action = "hold"

    bench_level = None
    if portfolio.benchmark_index in bundle.benchmarks.columns:
        bser = bundle.benchmarks[portfolio.benchmark_index]
        bser = bser[bser.index <= as_of]
        if len(bser):
            bench_level = float(bser.iloc[-1])
    portfolio.mark(as_of, prices, benchmark=bench_level, note=action)

    return PaperStepResult(date=as_of, nav=portfolio.nav(prices), action=action,
                           trade=trade, decision_reasons=report.decision.reasons,
                           report=report)


def performance_summary(portfolio: PaperPortfolio) -> dict:
    """Track-record stats from the recorded NAV history (vs the benchmark)."""
    if not portfolio.history:
        return {"status": "no history yet"}
    hist = pd.DataFrame(portfolio.history)
    nav = hist["nav"].astype(float)
    total_return = float(nav.iloc[-1] / portfolio.initial_capital - 1.0)
    out = {
        "start": hist["date"].iloc[0], "asof": hist["date"].iloc[-1],
        "marks": len(hist),
        "nav": float(nav.iloc[-1]),
        "total_return": round(total_return, 4),
    }
    if hist["benchmark"].notna().any():
        b = hist["benchmark"].astype(float)
        first = b[b.notna()].iloc[0]
        bench_ret = float(b.iloc[-1] / first - 1.0)
        out["benchmark_return"] = round(bench_ret, 4)
        out["excess_return"] = round(total_return - bench_ret, 4)
    if len(nav) > 2:
        rets = nav.pct_change().dropna()
        curve = nav / nav.cummax() - 1.0
        out["max_drawdown"] = round(float(curve.min()), 4)
        vol = float(rets.std(ddof=0))
        out["return_vol_per_mark"] = round(vol, 4)
    return out
