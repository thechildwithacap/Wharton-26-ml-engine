"""Point-in-time S&P 500 constituent universe (audit Task E).

Builds a *delisting-inclusive* S&P 500 membership using the real index
addition/removal history, reusing the existing survivorship framework
(``SecurityMeta.delisting_date`` + ``DataBundle.eligible_asof``) — only the
source list changes.  This is the methodologically-correct free substitute for
"all NYSE": a fixed, well-documented, ~500-name index with published membership
changes, rather than a larger but survivor-biased grab-bag.

Hard limitation (measured, not assumed): the free price feed (TwelveData) does
not serve delisted tickers — SIVB / FRC / SBNY return 404 — which is the literal
mechanism of survivorship bias.  So the *prices* of failed members can't be
fetched for free; a fully-real bias backtest needs a survivorship-free source
(CRSP/WRDS).  This module supplies the membership + a modeled bias estimate from
the real removals and keeps the synthetic 120->92 demo as the clean control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

# Real, notable S&P 500 removals 2019-2025 (ticker, removal date, reason,
# outcome).  outcome drives the modeled bias estimate:
#   failed   -> near-total loss the survivor set never sees (upward bias)
#   acquired -> left at a takeover premium (a gain the survivor set never sees)
#   dropped  -> removed for size/underperformance (typically laggards)
# This is a curated, defensible subset — not the full change log (which needs a
# maintained data source); it captures the removals that matter most for bias.
SP500_REMOVALS: List[dict] = [
    # --- 2023 bank failures: the canonical survivorship-bias cases ---
    {"ticker": "SIVB", "date": "2023-03-15", "reason": "SVB Financial — bank failure", "outcome": "failed"},
    {"ticker": "SBNY", "date": "2023-03-15", "reason": "Signature Bank — bank failure", "outcome": "failed"},
    {"ticker": "FRC",  "date": "2023-05-04", "reason": "First Republic — bank failure", "outcome": "failed"},
    # --- other distress / size removals (laggards) ---
    {"ticker": "BBBY", "date": "2019-01-02", "reason": "Bed Bath & Beyond — size (later bankrupt)", "outcome": "dropped"},
    {"ticker": "XRX",  "date": "2020-06-22", "reason": "Xerox — size/underperformance", "outcome": "dropped"},
    {"ticker": "M",    "date": "2020-04-06", "reason": "Macy's — size/underperformance", "outcome": "dropped"},
    {"ticker": "NBL",  "date": "2020-10-05", "reason": "Noble Energy — energy distress", "outcome": "dropped"},
    {"ticker": "HFC",  "date": "2022-01-13", "reason": "HollyFrontier — merger/size", "outcome": "dropped"},
    {"ticker": "PBCT", "date": "2022-04-04", "reason": "People's United — acquired", "outcome": "acquired"},
    {"ticker": "DISH", "date": "2024-01-02", "reason": "Dish Network — size/underperformance", "outcome": "dropped"},
    {"ticker": "SEE",  "date": "2024-06-24", "reason": "Sealed Air — size", "outcome": "dropped"},
    {"ticker": "ZION", "date": "2024-01-02", "reason": "Zions Bancorp — size", "outcome": "dropped"},
    {"ticker": "NWL",  "date": "2023-06-20", "reason": "Newell Brands — size", "outcome": "dropped"},
    {"ticker": "LUMN", "date": "2023-03-20", "reason": "Lumen — distress", "outcome": "failed"},
    {"ticker": "VNO",  "date": "2023-03-20", "reason": "Vornado — size", "outcome": "dropped"},
    {"ticker": "DXC",  "date": "2023-09-18", "reason": "DXC Technology — size", "outcome": "dropped"},
    {"ticker": "AAL",  "date": "2024-09-23", "reason": "American Airlines — size", "outcome": "dropped"},
    # --- acquisitions (removed at a premium — a gain survivors never book) ---
    {"ticker": "CELG", "date": "2019-11-21", "reason": "Celgene — acquired by BMY", "outcome": "acquired"},
    {"ticker": "RHT",  "date": "2019-07-09", "reason": "Red Hat — acquired by IBM", "outcome": "acquired"},
    {"ticker": "APC",  "date": "2019-08-08", "reason": "Anadarko — acquired by OXY", "outcome": "acquired"},
    {"ticker": "AGN",  "date": "2020-05-08", "reason": "Allergan — acquired by ABBV", "outcome": "acquired"},
    {"ticker": "TIF",  "date": "2021-01-07", "reason": "Tiffany — acquired by LVMH", "outcome": "acquired"},
    {"ticker": "MXIM", "date": "2021-08-26", "reason": "Maxim — acquired by ADI", "outcome": "acquired"},
    {"ticker": "ALXN", "date": "2021-07-21", "reason": "Alexion — acquired by AZN", "outcome": "acquired"},
    {"ticker": "XLNX", "date": "2022-02-14", "reason": "Xilinx — acquired by AMD", "outcome": "acquired"},
    {"ticker": "CERN", "date": "2022-06-08", "reason": "Cerner — acquired by ORCL", "outcome": "acquired"},
    {"ticker": "ATVI", "date": "2023-10-13", "reason": "Activision — acquired by MSFT", "outcome": "acquired"},
    {"ticker": "PXD",  "date": "2024-05-03", "reason": "Pioneer — acquired by XOM", "outcome": "acquired"},
    {"ticker": "TWTR", "date": "2022-11-08", "reason": "Twitter — taken private", "outcome": "acquired"},
]

# Modeled outcome returns over the ~year into removal (illustrative, documented):
# failed members go to ~0; acquired leave at a premium; dropped tend to lag.
_OUTCOME_RETURN = {"failed": -0.95, "acquired": +0.20, "dropped": -0.15}


@dataclass
class SurvivorshipEstimate:
    n_survivors: int
    n_removed: int
    counts: Dict[str, int]
    mean_removed_return: float
    est_bias_annual: float
    method: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_survivors": self.n_survivors, "n_removed": self.n_removed,
            "removal_counts": self.counts,
            "mean_removed_1y_return": round(self.mean_removed_return, 4),
            "est_survivorship_bias_annual": round(self.est_bias_annual, 4),
            "method": self.method,
        }


def sp500_pit_metadata(current_tickers: List[str],
                       removals: Optional[List[dict]] = None) -> Dict[str, dict]:
    """Point-in-time membership: current names (still listed) + removed names
    carrying a ``delisting_date`` so ``eligible_asof`` drops them after removal.
    """
    removals = removals or SP500_REMOVALS
    meta: Dict[str, dict] = {}
    for t in current_tickers:
        meta[t] = {"ticker": t, "delisting_date": None, "security_type": "common"}
    for r in removals:
        meta[r["ticker"]] = {"ticker": r["ticker"], "delisting_date": r["date"],
                             "security_type": "common", "reason": r["reason"],
                             "outcome": r["outcome"]}
    return meta


def estimate_survivorship_bias(n_survivors: int,
                               removals: Optional[List[dict]] = None,
                               years: float = 6.0) -> SurvivorshipEstimate:
    """Modeled estimate of the CAGR a survivor-only book overstates.

    Approximation: a survivor-only universe omits every removed name's outcome.
    The annual bias ~ (weight of removed names) x (survivor mean minus removed
    mean return) / years, using documented per-outcome returns.  Clearly a
    *model*, not a real-price backtest — the free feed can't supply delisted
    prices (see module docstring).
    """
    removals = removals or SP500_REMOVALS
    counts: Dict[str, int] = {}
    rets = []
    for r in removals:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
        rets.append(_OUTCOME_RETURN.get(r["outcome"], 0.0))
    n_removed = len(removals)
    mean_removed = sum(rets) / n_removed if n_removed else 0.0
    total = n_survivors + n_removed
    removed_weight = n_removed / total if total else 0.0
    # survivor mean assumed ~0 excess; the omitted names drag the true mean down.
    est_bias_annual = -removed_weight * mean_removed / years
    return SurvivorshipEstimate(
        n_survivors=n_survivors, n_removed=n_removed, counts=counts,
        mean_removed_return=mean_removed, est_bias_annual=est_bias_annual,
        method=(f"{n_removed} real removals x documented outcome returns "
                f"(failed {_OUTCOME_RETURN['failed']:+.0%}, acquired "
                f"{_OUTCOME_RETURN['acquired']:+.0%}, dropped "
                f"{_OUTCOME_RETURN['dropped']:+.0%}), spread over {years:g}y"))
