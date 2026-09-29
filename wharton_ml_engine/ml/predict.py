"""Live scoring with a trained alpha model.

Wraps a :class:`TrainedModel` and turns its raw prediction into a 0-100
``ml_alpha`` score (cross-sectional percentile rank), consistent with every
other signal in the engine so it slots straight into the integration layer.

**Evidence gate.** ``score()`` shrinks its output toward the neutral midpoint
(50) by ``1 - confidence()`` before returning it. A model with strong,
Newey-West-significant out-of-sample evidence keeps its full say in the
integration blend (``integrate_scores`` gives ``ml_alpha`` a fixed 22% blend
weight whenever it's present — see ``engine/integrate.py``); a model with weak
or insignificant evidence gets shrunk toward "no information", so its fixed
blend weight stops moving the portfolio's rankings even though the weight
itself doesn't change. This closes a real gap: previously *any* trained model,
however weak, got the same fixed 22% say. Pass ``gate=False`` to get the raw,
ungated score — used by :mod:`.explain`, which needs to decompose the model's
actual prediction, not a confidence-shrunk proxy for it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..data.source import DataBundle
from ..utils import clip_score, pct_rank
from .dataset import feature_frame, is_extended
from .train import TrainedModel


class AlphaModel:
    def __init__(self, trained: TrainedModel) -> None:
        self.trained = trained

    @classmethod
    def load(cls, path: str) -> "AlphaModel":
        return cls(TrainedModel.load(path))

    @property
    def metrics(self):
        return self.trained.metrics

    def coefficients(self) -> pd.Series:
        return self.trained.coefficients()

    def confidence(self, n_trials: int = 1) -> float:
        """Evidence-based confidence in [0, 1] from this model's out-of-sample
        walk-forward metrics, Sidak-corrected for how many variants were tried.

        Returns 0.0 (no confidence) when the model's evidence is missing,
        non-finite, or the mean IC isn't even positive — a negative-IC model
        gets zero say regardless of its t-stat's magnitude. Otherwise scales
        with the multiple-testing-adjusted t-stat: roughly 0 near t_adj=1,
        approaching 1.0 by t_adj=3 (the Harvey-Liu-Zhu reference bar).

        ``n_trials`` should be the number of variants evaluated in this
        model's experiment-registry family (see
        ``wharton_ml_engine.ml.validation.count_trials``); the default of 1
        applies no multiple-testing penalty, which is only appropriate for a
        model that genuinely wasn't compared against alternatives.
        """
        m = self.trained.metrics or {}
        mean_ic = m.get("mean_ic")
        nw_t = m.get("nw_t")
        if mean_ic is None or nw_t is None:
            return 0.0
        if not (np.isfinite(mean_ic) and np.isfinite(nw_t)):
            return 0.0
        if mean_ic <= 0:
            return 0.0
        from .validation import adjusted_significance
        adj = adjusted_significance(nw_t, n_trials=n_trials)
        t_adj = adj["t_adj"]
        if t_adj is None or not np.isfinite(t_adj) or t_adj <= 0:
            return 0.0
        return float(np.clip((t_adj - 1.0) / 2.0, 0.0, 1.0))

    def score(self, bundle: DataBundle, as_of: Optional[pd.Timestamp] = None,
             gate: bool = True, n_trials: int = 1) -> pd.Series:
        """Return the 0-100 ``ml_alpha`` score per ticker at ``as_of``.

        With ``gate=True`` (default) the score is shrunk toward 50 by
        ``1 - confidence(n_trials)`` — see the module docstring. Pass
        ``gate=False`` for the raw, ungated cross-sectional score (e.g. for
        explanation/decomposition, where shrinking would misrepresent what the
        model actually predicted).
        """
        if as_of is None:
            as_of = bundle.dates()[-1]
        ext = is_extended(self.trained.feature_names)
        feats = feature_frame(bundle, pd.Timestamp(as_of), extended=ext)
        raw = self.trained.raw_predict(feats)
        raw_score = clip_score(pct_rank(raw, ascending=True))
        if gate:
            conf = self.confidence(n_trials=n_trials)
            raw_score = 50.0 + conf * (raw_score - 50.0)
        return raw_score.rename("ml_alpha")

    def raw_score(self, bundle: DataBundle, as_of: Optional[pd.Timestamp] = None) -> pd.Series:
        if as_of is None:
            as_of = bundle.dates()[-1]
        ext = is_extended(self.trained.feature_names)
        return self.trained.raw_predict(
            feature_frame(bundle, pd.Timestamp(as_of), extended=ext))
