"""``MLView``: one object bundling the ML layer's opinion, confidence and
plain-English reasons for a decision date — the single interface any
downstream consumer (the engine's integration layer, the API/UI, a future
firm-hierarchy layer) should read from, instead of each building its own path
into the model's internals.

This is a deliberately small slice of the original PRD's W6 (integration):
the ``firm``/``desk``/``goals`` package hierarchy that PRD draft assumed does
not exist in this codebase (see ``docs/PRD_v3_ML.md`` for the reconciliation).
What *does* exist and *is* wired here: the engine's ``integrate_scores`` blend
(via the evidence gate on ``AlphaModel.score``) and, through this module, the
API/UI layer (``wharton_ml_engine/api``), which can render ``MLView.reasons``
and ``MLView.card`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from ..data.source import DataBundle
from .card import ModelCard, build_model_card
from .explain import Explanation, explain
from .narrate import narrate
from .predict import AlphaModel
from .validation import count_trials


@dataclass
class MLView:
    as_of: pd.Timestamp
    model_id: str
    confidence: float
    score: pd.Series             # 0-100 GATED percentile, index = ticker (what the portfolio actually sees)
    raw_score: pd.Series         # 0-100 ungated percentile (what the model actually predicted)
    explanation: Explanation
    reasons: pd.Series           # ticker -> plain-English reason
    card: ModelCard

    def to_dict(self) -> Dict[str, object]:
        """JSON-safe summary for an API response — full per-ticker detail lives
        in ``explanation``/``reasons`` for callers that want it."""
        return {
            "as_of": str(self.as_of.date()), "model_id": self.model_id,
            "confidence": round(self.confidence, 4),
            "gated_off": self.confidence <= 0.05,
            "scores": {t: (None if pd.isna(v) else round(float(v), 2))
                      for t, v in self.score.items()},
            "reasons": dict(self.reasons),
            "card": self.card.to_dict(),
        }


def build_ml_view(alpha_model: AlphaModel, bundle: DataBundle,
                  as_of: Optional[pd.Timestamp] = None,
                  model_id: str = "model",
                  registry_family: Optional[str] = None,
                  raw_fundamentals: Optional[pd.DataFrame] = None) -> MLView:
    """Build the full :class:`MLView` for ``as_of`` from an already-loaded
    :class:`AlphaModel` — the entry point for a caller that trains or loads its
    model once and reuses it across requests (e.g. the API service), so it
    never round-trips through disk just to get a view.

    ``registry_family``, if given, is used to look up how many variants this
    model's family has logged in the experiment registry (see
    ``ml/validation.py``), which the evidence gate's Sidak correction uses —
    without it, confidence is computed with ``n_trials=1`` (no multiple-testing
    penalty), which is only appropriate for a model that wasn't compared
    against alternatives.

    Computed over ``bundle``'s **full** universe (never pre-restricted to a
    subset of tickers) so that percentile ranks and "top X%" language stay
    honest — a caller that only wants a handful of tickers' reasons (e.g. the
    API trimming its response to the recommended holdings) should select from
    the returned ``MLView.reasons``/``.score`` afterward, not ask this function
    to compute over a smaller universe.
    """
    if as_of is None:
        as_of = bundle.dates()[-1]
    as_of = pd.Timestamp(as_of)

    n_trials = count_trials(registry_family) if registry_family else 1
    confidence = alpha_model.confidence(n_trials=n_trials)
    gated = alpha_model.score(bundle, as_of, gate=True, n_trials=n_trials)
    raw = alpha_model.score(bundle, as_of, gate=False)
    exp = explain(alpha_model, bundle, as_of, model_id=model_id)
    fund = raw_fundamentals
    if fund is None:
        try:
            fund = bundle.fundamentals_asof(as_of)
        except Exception:
            fund = None
    reasons = narrate(exp, confidence, raw_fundamentals=fund)
    card = build_model_card(alpha_model, model_id=model_id, n_trials=n_trials)

    return MLView(as_of=as_of, model_id=model_id, confidence=confidence,
                 score=gated, raw_score=raw, explanation=exp, reasons=reasons,
                 card=card)


def load_ml_view(model_path: str, bundle: DataBundle,
                 as_of: Optional[pd.Timestamp] = None,
                 model_id: Optional[str] = None,
                 registry_family: Optional[str] = None,
                 raw_fundamentals: Optional[pd.DataFrame] = None) -> MLView:
    """Load a saved model from ``model_path`` and build its :class:`MLView` —
    thin wrapper over :func:`build_ml_view` for a caller that only has a path
    (e.g. a one-off CLI run) rather than an already-loaded model in memory."""
    am = AlphaModel.load(model_path)
    return build_ml_view(am, bundle, as_of=as_of, model_id=model_id or model_path,
                         registry_family=registry_family,
                         raw_fundamentals=raw_fundamentals)
