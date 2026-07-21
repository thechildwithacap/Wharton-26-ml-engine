"""The fictional-client profile and the hard rules it implies (PRD section 5).

A :class:`ClientProfile` is the machine-readable form of the Wharton case's
client.  It carries the client's objectives, risk tolerance and constraints,
and it can (a) *tighten* the portfolio constraints to match the mandate and
(b) express the client's style tilts that feed the integrated score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..config import PortfolioConstraints, STYLES


RISK_LABELS = {
    1: "very conservative",
    2: "conservative",
    3: "balanced",
    4: "aggressive",
    5: "very aggressive",
}

# Risk tolerance -> soft target annual volatility band and max portfolio beta.
_RISK_VOL_BANDS = {
    1: (0.06, 0.10, 0.75),
    2: (0.08, 0.13, 0.90),
    3: (0.10, 0.16, 1.05),
    4: (0.13, 0.20, 1.20),
    5: (0.16, 0.28, 1.40),
}

_TURNOVER_CAP = {"low": 0.15, "medium": 0.30, "high": 0.50}


@dataclass
class ClientProfile:
    name: str = "Fictional Client"
    age: int = 45
    profession: str = "Professional"
    geography: str = "United States"

    # growth | income | capital_preservation | balanced
    objective: str = "balanced"
    risk_tolerance: int = 3           # 1..5
    horizon: str = "long"             # short | medium | long

    sector_preferences: List[str] = field(default_factory=list)
    sector_avoidances: List[str] = field(default_factory=list)

    # Style tilts in [0, 1]; higher = client favours this style.
    style_preferences: Dict[str, float] = field(default_factory=dict)

    esg_required: bool = False
    excluded_themes: List[str] = field(default_factory=list)

    turnover_tolerance: str = "medium"  # low | medium | high
    min_dividend_yield: float = 0.0     # portfolio-level income need

    def __post_init__(self) -> None:
        assert self.objective in {"growth", "income", "capital_preservation", "balanced"}
        assert self.risk_tolerance in RISK_LABELS
        assert self.horizon in {"short", "medium", "long"}
        assert self.turnover_tolerance in _TURNOVER_CAP
        # Normalise style-preference keys and default the rest to neutral (0.5).
        self.style_preferences = {
            s: float(self.style_preferences.get(s, 0.5)) for s in STYLES
        }

    # ------------------------------------------------------------------ views
    @property
    def risk_label(self) -> str:
        return RISK_LABELS[self.risk_tolerance]

    @property
    def target_vol_band(self):
        lo, hi, _ = _RISK_VOL_BANDS[self.risk_tolerance]
        return (lo, hi)

    @property
    def max_beta(self) -> float:
        return _RISK_VOL_BANDS[self.risk_tolerance][2]

    def apply_to_constraints(self, base: PortfolioConstraints) -> PortfolioConstraints:
        """Return a copy of ``base`` tightened to honour this client's mandate."""
        import copy

        c = copy.deepcopy(base)
        # Turnover discipline follows the client's stated tolerance.
        c.max_turnover_per_rebalance = min(
            c.max_turnover_per_rebalance, _TURNOVER_CAP[self.turnover_tolerance]
        )
        # Conservative clients get tighter position/sector caps.
        if self.risk_tolerance <= 2:
            c.max_weight_per_stock = min(c.max_weight_per_stock, 0.06)
            c.target_max_weight_per_stock = min(c.target_max_weight_per_stock, 0.04)
            c.max_weight_per_sector = min(c.max_weight_per_sector, 0.20)
        if self.objective == "capital_preservation":
            c.max_weight_per_stock = min(c.max_weight_per_stock, 0.05)
        return c

    def style_tilt_vector(self) -> Dict[str, float]:
        """Objective + explicit preferences -> per-style multiplier around 1.0.

        Used by the integration layer to nudge the regime style weights toward
        what the client actually wants.
        """
        tilt = {s: 1.0 for s in STYLES}
        obj_bias = {
            "growth": {"growth": 1.3, "garp": 1.15, "momentum": 1.1, "income": 0.7},
            "income": {"income": 1.4, "quality": 1.15, "low_vol": 1.1, "growth": 0.7},
            "capital_preservation": {"low_vol": 1.4, "quality": 1.25, "value": 1.1,
                                     "momentum": 0.7, "growth": 0.75},
            "balanced": {},
        }[self.objective]
        for s in STYLES:
            pref = self.style_preferences.get(s, 0.5)
            tilt[s] *= obj_bias.get(s, 1.0)
            tilt[s] *= 0.6 + 0.8 * pref   # 0.5 pref -> 1.0, 1.0 pref -> 1.4
        return tilt
