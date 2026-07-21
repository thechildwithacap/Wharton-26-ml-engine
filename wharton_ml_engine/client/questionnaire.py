"""Structured 20+ question client questionnaire (PRD 5.1).

The questionnaire is the auditable bridge between the Wharton case narrative
and the machine-readable :class:`ClientProfile`.  Each question has a stable
``key``; :func:`profile_from_answers` maps a dict of answers to a profile.  A
worked example (:func:`sample_answers`) stands in for a filled-out case so the
demo runs, but a team edits the answers each competition year.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import SECTORS, STYLES, THEMES
from .profile import ClientProfile


@dataclass
class Question:
    key: str
    prompt: str
    kind: str                       # choice | multichoice | scale | number | bool | text
    options: Optional[List[Any]] = None
    help: str = ""


QUESTIONNAIRE: List[Question] = [
    # --- Demographics & background ---------------------------------------
    Question("name", "Client name / label", "text"),
    Question("age", "Client age", "number"),
    Question("profession", "Profession / income source", "text"),
    Question("geography", "Country / tax domicile", "text"),
    Question("dependents", "Number of financial dependents", "number"),
    Question("net_worth_band", "Investable net-worth band", "choice",
             ["low", "medium", "high"]),
    # --- Objectives -------------------------------------------------------
    Question("objective", "Primary objective", "choice",
             ["growth", "income", "capital_preservation", "balanced"]),
    Question("secondary_objective", "Secondary objective", "choice",
             ["growth", "income", "capital_preservation", "balanced", "none"]),
    Question("return_expectation", "Target annual return", "number",
             help="Used only as a sanity check against risk tolerance."),
    Question("income_need", "Needs regular income from portfolio?", "bool"),
    Question("min_dividend_yield", "Minimum acceptable portfolio dividend yield",
             "number"),
    # --- Risk -------------------------------------------------------------
    Question("risk_tolerance", "Risk tolerance (1 very conservative .. 5 very aggressive)",
             "scale", [1, 2, 3, 4, 5]),
    Question("max_loss_tolerance", "Maximum tolerable peak-to-trough drawdown",
             "number"),
    Question("reaction_to_20pct_drop", "Reaction to a 20% drawdown", "choice",
             ["sell everything", "sell some", "hold", "buy more"]),
    Question("liquidity_need", "Might funds be needed at short notice?", "bool"),
    # --- Horizon ----------------------------------------------------------
    Question("horizon", "Investment time horizon", "choice",
             ["short", "medium", "long"]),
    Question("horizon_years", "Horizon in years", "number"),
    # --- Preferences & constraints ---------------------------------------
    Question("sector_preferences", "Preferred sectors", "multichoice", SECTORS),
    Question("sector_avoidances", "Sectors to avoid / exclude", "multichoice", SECTORS),
    Question("excluded_themes", "Themes to exclude", "multichoice", THEMES),
    Question("esg_required", "ESG screening required?", "bool"),
    Question("style_preferences", "Preferred styles (0-1 each)", "multichoice", STYLES),
    Question("turnover_tolerance", "Turnover tolerance", "choice",
             ["low", "medium", "high"]),
    Question("home_country_only", "Restrict to domestic (US) equities?", "bool"),
]


def _obj_to_risk_hint(objective: str) -> int:
    return {"capital_preservation": 2, "income": 3, "balanced": 3, "growth": 4}.get(
        objective, 3
    )


def profile_from_answers(answers: Dict[str, Any]) -> ClientProfile:
    """Build a :class:`ClientProfile` from a (possibly partial) answer dict."""
    objective = answers.get("objective", "balanced")

    style_prefs = answers.get("style_preferences") or {}
    # Accept either {style: weight} or a list of favoured styles.
    if isinstance(style_prefs, (list, tuple, set)):
        style_prefs = {s: (0.8 if s in style_prefs else 0.5) for s in STYLES}

    return ClientProfile(
        name=answers.get("name", "Fictional Client"),
        age=int(answers.get("age", 45)),
        profession=answers.get("profession", "Professional"),
        geography=answers.get("geography", "United States"),
        objective=objective,
        risk_tolerance=int(answers.get("risk_tolerance", _obj_to_risk_hint(objective))),
        horizon=answers.get("horizon", "long"),
        sector_preferences=list(answers.get("sector_preferences", [])),
        sector_avoidances=list(answers.get("sector_avoidances", [])),
        style_preferences=style_prefs,
        esg_required=bool(answers.get("esg_required", False)),
        excluded_themes=list(answers.get("excluded_themes", [])),
        turnover_tolerance=answers.get("turnover_tolerance", "medium"),
        min_dividend_yield=float(answers.get("min_dividend_yield", 0.0)),
    )


def sample_answers() -> Dict[str, Any]:
    """A worked example: a 52-year-old approaching-retirement balanced client."""
    return {
        "name": "R. Whitman (Case Client)",
        "age": 52,
        "profession": "Small-business owner",
        "geography": "United States",
        "dependents": 2,
        "net_worth_band": "medium",
        "objective": "balanced",
        "secondary_objective": "income",
        "return_expectation": 0.08,
        "income_need": True,
        "min_dividend_yield": 0.012,
        "risk_tolerance": 3,
        "max_loss_tolerance": 0.20,
        "reaction_to_20pct_drop": "hold",
        "liquidity_need": False,
        "horizon": "long",
        "horizon_years": 12,
        "sector_preferences": ["Health Care", "Technology", "Industrials"],
        "sector_avoidances": ["Energy"],
        "excluded_themes": [],
        "esg_required": True,
        "style_preferences": {
            "value": 0.6, "quality": 0.8, "growth": 0.5, "garp": 0.6,
            "momentum": 0.4, "low_vol": 0.7, "income": 0.6,
        },
        "turnover_tolerance": "medium",
        "home_country_only": True,
    }


def sample_profile() -> ClientProfile:
    return profile_from_answers(sample_answers())
