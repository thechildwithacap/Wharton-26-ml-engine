"""Competition-rules adapter — one place to encode a contest's rulebook.

Most investment competitions (the Wharton Global High School comp included)
dictate the same handful of levers: a starting balance, an allowed universe,
position and sector caps, whether shorting / leverage / options are permitted, a
rebalance cadence, a benchmark, and — the one that decides strategy — **how you
are scored** (total return vs. risk-adjusted vs. vs-benchmark).

Rather than thread those through the code by hand every time the rules change,
capture them once in :class:`CompetitionRules`.  ``.to_engine_config()`` maps the
rules onto the engine's existing knobs (``EngineConfig`` /
``PortfolioConstraints`` / ``UniverseFilter``), and — importantly — ``.readiness()``
reports, per lever, whether it is **enforced** by config today, only **partial**,
or genuinely **needs code**.  So when the real rules arrive, adaptation is: fill
in the fields, read the readiness report, and build only the few items it flags.
No guessing which parts already work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import CostModel, EngineConfig, PortfolioConstraints, UniverseFilter

# How the contest ranks entries.  This is the single most strategy-defining rule:
# a total-return contest rewards concentration; a Sharpe contest rewards the
# diversified, risk-managed book the engine already builds.
SCORING_METRICS = ("total_return", "sharpe", "vs_benchmark", "drawdown_adjusted")

# Rebalance cadence -> a sensible per-rebalance turnover cap (fraction of book).
_CADENCE_TURNOVER = {
    "daily": 0.50, "weekly": 0.40, "biweekly": 0.35,
    "monthly": 0.30, "quarterly": 0.20, "buy_and_hold": 0.05,
}


@dataclass
class CompetitionRules:
    """A contest rulebook, expressed as tunable fields.

    Leave anything unknown at its default; ``readiness()`` will show what still
    needs to be pinned down.  Nothing here is competition-specific until you set
    it — this is a container, not an assumption.
    """

    name: str = "Generic long-only equity competition"

    # --- capital & horizon ---------------------------------------------------
    starting_capital: float = 100_000.0
    start_date: Optional[str] = None            # ISO; contest window start
    end_date: Optional[str] = None              # ISO; contest window end

    # --- universe ------------------------------------------------------------
    allowed_index: Optional[str] = None         # e.g. "SPX", "NDX"; None = all data
    allowed_security_types: Tuple[str, ...] = ("common",)
    min_price: float = 5.0
    min_adv_usd: float = 1_000_000.0
    allowed_asset_classes: Tuple[str, ...] = ("equity",)   # equity|etf|bond|crypto

    # --- position / diversification limits -----------------------------------
    min_holdings: int = 15
    max_holdings: int = 30
    max_position_pct: float = 0.10              # hard cap per name
    max_sector_pct: float = 0.30

    # --- cash --------------------------------------------------------------- -
    min_invested_pct: float = 1.0               # must be >=X invested (1.0 = fully)
    max_cash_pct: float = 0.0                   # cash allowed up to this share

    # --- instruments ---------------------------------------------------------
    allow_shorting: bool = False
    allow_leverage: bool = False
    allow_options: bool = False

    # --- trading -------------------------------------------------------------
    rebalance_frequency: str = "monthly"        # see _CADENCE_TURNOVER
    max_trades: Optional[int] = None            # cap on number of trades (if any)
    min_holding_days: int = 0                   # anti-day-trading rule

    # --- objective & benchmark ----------------------------------------------
    benchmark: str = "SPX"
    scoring_metric: str = "total_return"        # SCORING_METRICS
    esg_required: bool = False

    # --- mapping to the engine ----------------------------------------------
    def to_constraints(self) -> PortfolioConstraints:
        c = PortfolioConstraints(
            min_holdings=self.min_holdings,
            max_holdings=self.max_holdings,
            max_weight_per_stock=self.max_position_pct,
            target_max_weight_per_stock=min(self.max_position_pct * 0.6,
                                            self.max_position_pct),
            max_weight_per_sector=self.max_sector_pct,
            soft_max_weight_per_sector=min(self.max_sector_pct * 0.8, self.max_sector_pct),
            max_turnover_per_rebalance=_CADENCE_TURNOVER.get(self.rebalance_frequency, 0.30),
            allow_leverage=self.allow_leverage,
            allow_shorting=self.allow_shorting,
            allow_options=self.allow_options,
        )
        # A total-return contest wants conviction; a Sharpe/preservation contest
        # wants spread.  Nudge the concentration exponent accordingly (still
        # inside the position caps, so it never violates the rules).
        if self.scoring_metric == "total_return":
            c.concentration = 1.6
        elif self.scoring_metric == "drawdown_adjusted":
            c.concentration = 0.8
        # vs_benchmark contests benefit from a partial cap-weight tilt.
        if self.scoring_metric == "vs_benchmark":
            c.benchmark_tilt = 0.35
        c.validate()
        return c

    def to_universe_filter(self) -> UniverseFilter:
        return UniverseFilter(
            security_types=tuple(self.allowed_security_types),
            min_price=self.min_price,
            min_adv_usd=self.min_adv_usd,
        )

    def horizon_days(self, trading_days_per_year: int = 252) -> int:
        """Contest length in trading days when a window is given, else 126."""
        if self.start_date and self.end_date:
            import pandas as pd
            days = (pd.Timestamp(self.end_date) - pd.Timestamp(self.start_date)).days
            return max(21, int(round(days * trading_days_per_year / 365.0)))
        return 126

    def to_engine_config(self, base: Optional[EngineConfig] = None) -> EngineConfig:
        cfg = base or EngineConfig()
        cfg.constraints = self.to_constraints()
        cfg.universe_filter = self.to_universe_filter()
        cfg.ml_horizon_days = self.horizon_days(cfg.trading_days_per_year)
        # Risk-scored contests reward staying diversified & regime-aware.
        cfg.regime_aware_sizing = self.scoring_metric in ("sharpe", "drawdown_adjusted")
        cfg.sector_neutral = self.scoring_metric == "vs_benchmark"
        return cfg

    # --- honesty layer: what actually works today ---------------------------
    def readiness(self) -> Dict[str, List[Dict[str, str]]]:
        """Classify every lever as enforced / partial / needs_code (honest map).

        This is the adaptation checklist: when the real rules arrive, only the
        ``needs_code`` items require new work.
        """
        enforced, partial, needs_code = [], [], []

        def add(bucket, lever, how):
            bucket.append({"lever": lever, "how": how})

        # --- fully handled by config today ---
        add(enforced, "starting_capital", "run_engine(portfolio_value=...)")
        add(enforced, "min/max holdings", "PortfolioConstraints.min/max_holdings")
        add(enforced, "max position %", "PortfolioConstraints.max_weight_per_stock")
        add(enforced, "max sector %", "PortfolioConstraints.max_weight_per_sector")
        add(enforced, "universe price/liquidity floor", "UniverseFilter.min_price/min_adv_usd")
        add(enforced, "security types (no ADR/SPAC/preferred)", "UniverseFilter.security_types")
        add(enforced, "long-only", "allow_shorting=False (construction is long-only)")
        add(enforced, "index restriction", "data.indices.available_universe(allowed_index)")
        add(enforced, "ESG / sector exclusions", "ClientProfile.esg_required / sector_avoidances")
        add(enforced, "benchmark comparison", "backtest benchmark_index / risk beta")
        add(enforced, "contest horizon -> ML label", "horizon_days() -> EngineConfig.ml_horizon_days")

        # --- works but approximate / not exact ---
        add(partial, "rebalance cadence", "mapped to a turnover cap, not an exact schedule/trade counter")
        add(partial, "scoring metric tilt", "nudges concentration/benchmark_tilt; not a true objective optimiser")
        add(partial, "leverage", "flag exists; construction does not yet lever the book")

        # --- genuinely needs new code when a rule requires it ---
        if self.min_invested_pct < 1.0 or self.max_cash_pct > 0.0:
            add(needs_code, "cash / de-gross to cash",
                "construction is always ~fully invested; add a cash sleeve + regime de-gross")
        if self.allow_shorting:
            add(needs_code, "short book",
                "long-only today; add short selection + gross/net exposure limits")
        if self.max_trades is not None:
            add(needs_code, "max-trades cap", "add a hard trade-count gate in decide_trades")
        if self.min_holding_days > 0:
            add(needs_code, "minimum holding period", "add a per-name hold-age gate before trimming")
        if any(a not in ("equity",) for a in self.allowed_asset_classes):
            add(needs_code, "non-equity assets",
                "engine is equities-only; add ETF/bond/crypto instrument handling")
        if self.scoring_metric in ("total_return", "sharpe", "drawdown_adjusted"):
            add(needs_code, "objective-aligned construction",
                f"engine maximises the integrated factor score, not '{self.scoring_metric}' "
                "directly; wire the contest metric into sizing to truly optimise for it")

        return {"enforced": enforced, "partial": partial, "needs_code": needs_code}

    def summary(self) -> str:
        r = self.readiness()
        lines = [f"Competition: {self.name}",
                 f"  scoring: {self.scoring_metric} | benchmark: {self.benchmark} | "
                 f"universe: {self.allowed_index or 'all data'} | "
                 f"holdings {self.min_holdings}-{self.max_holdings} | "
                 f"max pos {self.max_position_pct:.0%} | horizon {self.horizon_days()}d",
                 f"  ENFORCED by config: {len(r['enforced'])} levers",
                 f"  PARTIAL: {len(r['partial'])}",
                 f"  NEEDS CODE: {len(r['needs_code'])}"]
        for item in r["needs_code"]:
            lines.append(f"    - {item['lever']}: {item['how']}")
        return "\n".join(lines)
