"""Live scoring with a trained alpha model.

Wraps a :class:`TrainedModel` and turns its raw prediction into a 0-100
``ml_alpha`` score (cross-sectional percentile rank), consistent with every
other signal in the engine so it slots straight into the integration layer.
"""

from __future__ import annotations

from typing import Optional

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

    def score(self, bundle: DataBundle, as_of: Optional[pd.Timestamp] = None) -> pd.Series:
        """Return the 0-100 ``ml_alpha`` score per ticker at ``as_of``."""
        if as_of is None:
            as_of = bundle.dates()[-1]
        ext = is_extended(self.trained.feature_names)
        feats = feature_frame(bundle, pd.Timestamp(as_of), extended=ext)
        raw = self.trained.raw_predict(feats)
        return clip_score(pct_rank(raw, ascending=True)).rename("ml_alpha")

    def raw_score(self, bundle: DataBundle, as_of: Optional[pd.Timestamp] = None) -> pd.Series:
        if as_of is None:
            as_of = bundle.dates()[-1]
        ext = is_extended(self.trained.feature_names)
        return self.trained.raw_predict(
            feature_frame(bundle, pd.Timestamp(as_of), extended=ext))
