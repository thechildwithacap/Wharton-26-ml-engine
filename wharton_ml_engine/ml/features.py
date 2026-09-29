"""Feature registry: what every ML feature *means*, not just its name.

The model itself only sees standardised numbers; the registry is the layer
that lets a human (or an explanation generator, see :mod:`.narrate`) say why a
feature is in the model, which direction it should point, where it comes from,
and how stale its inputs are allowed to be. This is what makes the model
auditable instead of a coefficient list.

Only features that actually exist in this codebase's :mod:`.dataset` module are
registered here — 12 composite style scores + 16 raw SEC-fundamental ranks = 28.
A prior draft of this registry (see ``docs/PRD_v3_ML.md``) assumed 16 additional
price/macro features (``rev_1m``, ``beta_mkt``, FRED-based macro betas, …) from
a ``tools/ml_v2.py`` module that does not exist in this repository. The
``family`` taxonomy and the ``FeatureSpec`` schema below are deliberately built
to accommodate that future ``price``/``macro``/``interaction`` work without a
breaking change — see :data:`FAMILIES` — but nothing is registered under those
families until the feature actually exists and is wired into
:func:`wharton_ml_engine.ml.dataset.feature_frame`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

# The full family taxonomy, including families reserved for future feature
# sets (price/macro/interaction) so the schema doesn't need to change when
# W4/§17 style work adds them — only new FeatureSpec entries would be needed.
FAMILIES: Tuple[str, ...] = (
    "value", "quality", "growth", "momentum", "reversal", "risk", "macro",
    "issuance", "sector", "analyst_proxy", "interaction", "discipline",
)

Sign = Literal[-1, 0, 1]


@dataclass(frozen=True)
class FeatureSpec:
    name: str              # column name as it appears in the training panel
    family: str             # one of FAMILIES
    description: str        # one plain sentence a judge understands
    expected_sign: Sign     # +1 = higher is better (expected positive IC),
                            # -1 = higher is worse (expected negative IC),
                            #  0 = no directional prior (report but don't grade)
    rationale: str           # the economic reason, one or two sentences
    source: str              # "sec" | "prices" | "macro" | "derived"
    lag_days: int             # how stale the underlying input is allowed to be
    min_history_days: int = 0
    is_proxy: bool = False    # True when the feature stands in for data we
                              # don't actually have (e.g. the analyst overlay)

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r} for {self.name!r}")
        if self.expected_sign not in (-1, 0, 1):
            raise ValueError(f"expected_sign must be -1/0/1, got {self.expected_sign!r}")


def _composite(name, family, description, sign, rationale, is_proxy=False) -> FeatureSpec:
    """Composite style scores are derived from fundamentals as-of the decision
    date (0-45 day filing lag baked into the SEC point-in-time gate) or from
    the price history up to the decision date (0-day lag)."""
    lag = 45 if family in ("value", "quality", "growth", "discipline") else 0
    return FeatureSpec(name=name, family=family, description=description,
                       expected_sign=sign, rationale=rationale, source="derived",
                       lag_days=lag, min_history_days=252 if family in
                       ("momentum", "risk") else 0, is_proxy=is_proxy)


# --- the 12 composite style scores (ML_FEATURES) --------------------------
_COMPOSITES: List[FeatureSpec] = [
    _composite("value", "value",
              "Blended valuation: P/E, P/B, EV/EBIT, FCF yield, balance-sheet "
              "soundness — higher means cheaper relative to fundamentals.",
              1, "Classic value premium: cheap stocks with sound balance "
              "sheets have historically outperformed expensive ones over "
              "multi-quarter horizons (Graham; Fama-French HML)."),
    _composite("quality", "quality",
              "Profitability and earnings stability: ROE, ROIC, gross margin, "
              "earnings volatility.",
              1, "High, stable profitability tends to persist and is "
              "rewarded by the market over time (quality factor)."),
    _composite("growth", "growth",
              "Revenue and EPS growth, weighted by how stable that growth "
              "has been.",
              1, "Sustainable growth (not just a single good quarter) is "
              "associated with continued outperformance."),
    _composite("garp", "growth",
              "Growth-at-a-reasonable-price: a PEG-style blend of the growth "
              "score with how expensive that growth is.",
              1, "Combines the growth and value priors: punishes paying an "
              "unreasonable multiple for growth."),
    _composite("discipline", "discipline",
              "Capital discipline: net share issuance (buybacks vs. "
              "dilution), asset growth, and accrual quality.",
              1, "Buybacks and conservative asset growth are independent, "
              "well-documented anomalies (issuance/investment effects) — "
              "management's use of capital, not the P&L.", is_proxy=False),
    _composite("momentum", "momentum",
              "12-month price return, skipping the most recent month.",
              1, "Cross-sectional momentum: recent relative winners tend to "
              "keep winning over the following months (Jegadeesh-Titman)."),
    _composite("low_vol", "risk",
              "Trailing realised volatility, inverted (higher score = lower "
              "volatility).",
              1, "The low-volatility anomaly: historically, low-vol stocks "
              "have delivered better risk-adjusted returns than the CAPM "
              "predicts."),
    _composite("size", "risk",
              "Market capitalisation, inverted (higher score = smaller cap).",
              1, "The classical size premium (Banz/Fama-French SMB). NOTE: "
              "this project measured this factor collapsing from a false "
              "t=5.14 to a real t=1.51 once a split-adjustment data bug was "
              "fixed — treat any size result with extra scrutiny."),
    _composite("factor", "momentum",
              "Composite of momentum, low-vol, size and short-term reversal.",
              0, "A blended price-factor score; no single directional prior "
              "beyond its components, which can offset each other."),
    _composite("macro_tilt", "macro",
              "Beta to the broad market, inverted (higher = more defensive).",
              0, "Static defensiveness only helps in a risk-off regime; in "
              "the risk-on window this project has data for, it tested "
              "significantly negative (defensive names lagged)."),
    _composite("macro_fit", "macro",
              "Regime-conditioned exposure match: rewards high-beta/growth "
              "exposure when the tape is risk-on, the reverse when risk-off.",
              0, "Sign is designed to flip with the regime, so no static "
              "directional prior applies; it has not yet been tested through "
              "a real risk-off stretch."),
    _composite("analyst", "analyst_proxy",
              "Analyst-overlay proxy, in practice a blend of quality/accruals "
              "signals — NOT real sell-side estimate or revision data.",
              0, "Registered as a proxy so it is never mistaken for "
              "independent information: it correlates ~0.94 with `quality` "
              "in this dataset.", is_proxy=True),
]

# --- the 16 raw SEC-fundamental ranks (RAW_FUNDAMENTAL_FEATURES, f_* cols) --
_RAW_SEC_SPECS: Dict[str, Tuple[str, str, Sign, str]] = {
    # field: (family, description, expected_sign, rationale)
    "pe": ("value", "Price / earnings, cross-sectional percentile rank "
          "(lower P/E ranks higher).", 1, "Cheaper earnings multiple."),
    "pb": ("value", "Price / book, percentile rank (lower ranks higher).",
          1, "Cheaper relative to net assets (the Graham/Fama-French value axis)."),
    "ev_ebit": ("value", "Enterprise value / EBIT, percentile rank (lower "
               "ranks higher).", 1, "Capital-structure-neutral cheapness."),
    "fcf_yield": ("value", "Free cash flow / market cap, percentile rank.",
                 1, "Cash-based valuation, harder to manipulate than earnings."),
    "roe": ("quality", "Return on equity, percentile rank.",
           1, "Profitability relative to shareholder capital."),
    "roic": ("quality", "Return on invested capital, percentile rank.",
            1, "Profitability relative to all capital employed, capital-"
            "structure neutral."),
    "gross_margin": ("quality", "Gross margin, percentile rank.",
                    1, "Pricing power / cost efficiency."),
    "earnings_vol": ("risk", "Volatility of yearly earnings growth, percentile "
                    "rank (lower volatility ranks higher).",
                    1, "Earnings stability is rewarded; volatile earnings "
                    "carry a discount."),
    "debt_equity": ("risk", "Debt / equity, percentile rank (lower leverage "
                   "ranks higher).", 1, "Lower leverage reduces distress risk."),
    "interest_coverage": ("risk", "EBIT / interest expense, percentile rank.",
                          1, "Higher coverage means more cushion against a "
                          "downturn."),
    "revenue_growth": ("growth", "Year-over-year revenue growth, percentile "
                       "rank.", 1, "Top-line growth."),
    "eps_growth": ("growth", "Year-over-year EPS growth, percentile rank.",
                  1, "Bottom-line growth."),
    "growth_stability": ("growth", "Consistency of growth (0-1), percentile "
                         "rank.", 1, "Steady growth is valued over erratic growth."),
    "accruals": ("discipline", "(Net income - operating cash flow) / assets, "
                "percentile rank (lower accruals rank higher).",
                1, "Sloan accruals anomaly: earnings backed by cash "
                "outperform accrual-heavy earnings."),
    "net_issuance": ("issuance", "Year-over-year change in shares outstanding, "
                     "percentile rank (buybacks rank higher).",
                     1, "Net issuance / buyback anomaly. NOTE: this is the "
                     "single feature in this project's audit history with "
                     "the most consistent standalone evidence — flag any "
                     "future result that contradicts its sign for review."),
    "asset_growth": ("discipline", "Year-over-year change in total assets, "
                     "percentile rank (slower growth ranks higher).",
                     1, "Investment/asset-growth anomaly: aggressive expansion "
                     "tends to precede weaker forward returns."),
}
_RAW_SPECS: List[FeatureSpec] = [
    FeatureSpec(name=f"f_{field}", family=family, description=desc,
               expected_sign=sign, rationale=rationale, source="sec",
               lag_days=45, min_history_days=365)
    for field, (family, desc, sign, rationale) in _RAW_SEC_SPECS.items()
]

REGISTRY: Dict[str, FeatureSpec] = {s.name: s for s in _COMPOSITES + _RAW_SPECS}


def spec(name: str) -> FeatureSpec:
    """Look up a feature's spec by its training-panel column name."""
    try:
        return REGISTRY[name]
    except KeyError as e:
        raise KeyError(
            f"{name!r} has no FeatureSpec — register it in ml/features.py "
            "before using it in a training panel or explanation.") from e


def specs_for(feature_names: List[str]) -> List[FeatureSpec]:
    return [spec(n) for n in feature_names]


def family_of(feature_names: List[str]) -> Dict[str, str]:
    return {n: spec(n).family for n in feature_names}


def proxy_features(feature_names: List[str]) -> List[str]:
    """Features that stand in for data we don't actually have — flag these
    wherever a report claims a feature is independent information."""
    return [n for n in feature_names if spec(n).is_proxy]
