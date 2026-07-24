"""Global configuration, constraints, and tunable defaults for the engine.

Every number the engine relies on should be reproducible.  Rather than
scattering magic constants across modules, the competition/mandate rules and
model tuning knobs live here so they are auditable in one place (see PRD
sections 8 and 11 on transparency).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Portfolio construction / mandate constraints (PRD section 8)
# ---------------------------------------------------------------------------


@dataclass
class PortfolioConstraints:
    """Hard and soft limits used during construction and mandate checks.

    Defaults follow the PRD.  ``ClientProfile`` may tighten (never loosen past
    competition rules) these per the fictional client's IPS.
    """

    min_holdings: int = 20
    max_holdings: int = 30

    max_weight_per_stock: float = 0.08          # hard cap, PRD "max 7-8%"
    target_max_weight_per_stock: float = 0.05   # soft target, "typically 2-5%"
    min_weight_per_stock: float = 0.02

    max_weight_per_sector: float = 0.25         # tunable per client
    soft_max_weight_per_sector: float = 0.20

    # Factor diversification: no single style may dominate the book.
    max_style_weight: float = 0.40

    # --- construction tilt knobs (risk/return dial) --------------------------
    # benchmark_tilt in [0, 1] interpolates the book between pure engine alpha
    # (0 = select & weight purely on the integrated score) and the cap-weighted
    # index (1 = select & weight purely by market cap).  Raise it when the goal
    # is to track / beat a cap-weighted benchmark in a mega-cap-led regime;
    # keep it at 0 for maximum factor diversification.
    benchmark_tilt: float = 0.0
    # concentration is the exponent on score-proportional base weights: 1.0 =
    # linear (default), >1 concentrates into the top names (up to the position
    # cap), <1 flattens toward equal weight.
    concentration: float = 1.0

    # Turnover discipline (fraction of book traded per rebalance).
    max_turnover_per_rebalance: float = 0.35

    # Liquidity: a position may not exceed this share of a name's average
    # daily traded value (keeps us from assuming untradeable fills).
    max_participation_of_adv: float = 0.10

    # Instruments / behaviours that may be banned by the competition.
    allow_leverage: bool = False
    allow_shorting: bool = False
    allow_options: bool = False

    def validate(self) -> None:
        assert 0 < self.min_holdings <= self.max_holdings
        assert 0 < self.min_weight_per_stock <= self.target_max_weight_per_stock
        assert self.target_max_weight_per_stock <= self.max_weight_per_stock <= 1
        assert 0 < self.soft_max_weight_per_sector <= self.max_weight_per_sector <= 1
        assert 0.0 <= self.benchmark_tilt <= 1.0
        assert 0.25 <= self.concentration <= 4.0


# ---------------------------------------------------------------------------
# Transaction cost model (PRD 6.2 turnover & cost)
# ---------------------------------------------------------------------------


@dataclass
class UniverseFilter:
    """Point-in-time universe eligibility rules (PRD §4, survivorship fix).

    Applied *as of each decision date* so that delisted / acquired names are
    included while they were tradable and dropped afterward — never filtered by
    what is listed *today*.  Security-type and liquidity floors are explicit
    (not silent defaults) so the universe definition is auditable.
    """

    # Security types to include; excludes ADRs, preferreds, SPACs, CEFs, etc.
    security_types: tuple = ("common",)
    min_price: float = 5.0            # drop sub-$5 names (untradeable/penny)
    min_adv_usd: float = 1_000_000.0  # drop names trading < $1M/day on average

    def describe(self) -> str:
        return (f"types={list(self.security_types)}, min_price=${self.min_price:.0f}, "
                f"min_adv=${self.min_adv_usd/1e6:.1f}M")


@dataclass
class CostModel:
    """Very simple, transparent proportional cost model in basis points."""

    commission_bps: float = 1.0      # per-trade commission proxy
    spread_bps: float = 5.0          # half-spread proxy
    impact_bps: float = 3.0          # market-impact proxy at target participation

    def cost_fraction(self) -> float:
        """One-way cost as a fraction of traded notional."""
        return (self.commission_bps + self.spread_bps + self.impact_bps) / 10_000.0


# ---------------------------------------------------------------------------
# Engine-wide settings
# ---------------------------------------------------------------------------


# Canonical style names used consistently across signal, backtest and engine
# layers.  Keep this list authoritative — other modules import it.
STYLES: List[str] = [
    "value",
    "quality",
    "growth",
    "garp",
    "momentum",
    "low_vol",
    "income",
]

# Sectors used by the synthetic universe and sector-cap logic.  A real run
# would take these from the approved-universe metadata.
SECTORS: List[str] = [
    "Technology",
    "Health Care",
    "Financials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Industrials",
    "Energy",
    "Materials",
    "Utilities",
    "Communication Services",
    "Real Estate",
]

# Investment themes for the theme/sector-tilt model (PRD 6.1.3).
THEMES: List[str] = [
    "AI",
    "Renewables",
    "Healthcare Innovation",
    "Cybersecurity",
    "Reshoring",
    "Dividend Aristocrat",
]


@dataclass
class EngineConfig:
    """Top-level configuration bundle passed through the pipeline."""

    seed: int = 42
    trading_days_per_year: int = 252
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)
    costs: CostModel = field(default_factory=CostModel)

    # Minimum expected net-of-cost score improvement (0-100 scale) required to
    # approve a rebalance (PRD 7.2 trading decision logic).
    min_score_improvement_to_trade: float = 1.5

    # Signal-confidence floor below which the engine trades more cautiously.
    min_signal_confidence: float = 0.35

    # Strip sector-average exposure from the style scores before integration, so
    # the book's tilt is a within-sector factor bet rather than a sector bet.
    sector_neutral: bool = False

    # Point-in-time universe eligibility (None = score every name in the bundle).
    universe_filter: "Optional[UniverseFilter]" = None

    # Adjust the conviction/sizing curve by regime (flatter for contrarian
    # setups, more concentrated for quality-compounder trends).
    regime_aware_sizing: bool = False

    def __post_init__(self) -> None:
        self.constraints.validate()
