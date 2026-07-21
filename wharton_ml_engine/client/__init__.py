"""Client & constraints module (PRD section 5)."""

from .fit import compute_client_fit
from .profile import ClientProfile, RISK_LABELS
from .questionnaire import (
    QUESTIONNAIRE,
    Question,
    profile_from_answers,
    sample_answers,
    sample_profile,
)

__all__ = [
    "ClientProfile",
    "RISK_LABELS",
    "QUESTIONNAIRE",
    "Question",
    "profile_from_answers",
    "sample_answers",
    "sample_profile",
    "compute_client_fit",
]
