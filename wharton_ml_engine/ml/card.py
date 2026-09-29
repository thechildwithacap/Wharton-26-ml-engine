"""Model cards: one JSON + one Markdown file per trained model, generated from
what the model actually is and actually measured — not written by hand.

A model card is the single artifact a judge, teammate, or future maintainer
should be able to read to understand: what data trained this model, what it's
allowed to do, how confident it is and why, and what its known limitations are.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .features import proxy_features, spec as feature_spec
from .predict import AlphaModel

DEFAULT_INTENDED_USE = (
    "Cross-sectional stock-selection tilt within the ML-eligible sleeve of the "
    "engine's recommended portfolio, blended alongside the rule-based factor "
    "score under evidence-gated weighting (see AlphaModel.confidence()). Not "
    "intended to size any liability-matching, hedge, or cash allocation, and "
    "never used as the sole basis for a trade."
)


@dataclass
class ModelCard:
    model_id: str
    task: str
    target_col: str
    horizon_days: int
    model_type: str
    hyperparameters: Dict[str, object]
    feature_names: List[str]
    feature_families: Dict[str, str]
    proxy_features: List[str]
    n_trials_in_family: int
    research_metrics: Dict[str, object]
    confidence: float
    trained_at: Optional[str]
    data_window: Optional[List[str]]
    n_rows: Optional[int]
    n_dates: Optional[int]
    holdout: Dict[str, object]
    known_limitations: List[str]
    intended_use: str
    top_families: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def to_markdown(self) -> str:
        m = self.research_metrics
        lines = [
            f"# Model card: {self.model_id}",
            "",
            f"**Task:** {self.task} · **Target:** {self.target_col} · "
            f"**Horizon:** {self.horizon_days} trading days",
            f"**Model type:** {self.model_type} · **Hyperparameters:** "
            f"`{json.dumps(self.hyperparameters, default=str)}`",
            f"**Trained:** {self.trained_at or 'unknown'} · "
            f"**Data window:** {self.data_window or 'unknown'} "
            f"({self.n_rows or '?'} rows, {self.n_dates or '?'} dates)",
            "",
            "## Confidence",
            f"**{self.confidence:.2f}** (0 = gated off, 1 = full weight), from "
            f"{self.n_trials_in_family} logged trial(s) in this model's family.",
        ]
        if self.confidence <= 0.05:
            lines.append(
                "This model is currently GATED OFF: its score contributes no "
                "differentiating information to the portfolio.")
        lines += [
            "",
            "## Research-window metrics",
            "| metric | value |", "|---|---|",
        ]
        for k in ("mean_ic", "nw_t", "nw_lags", "n_periods", "n_effective_periods",
                 "hit_rate", "oos_r2", "oos_accuracy"):
            if k in m and m[k] is not None:
                v = m[k]
                lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
        lines += ["", "## Holdout check"]
        if self.holdout.get("checked"):
            lines.append(f"Checked once on {self.holdout.get('n_holdout_dates', '?')} "
                        f"locked-holdout dates: nw_t={self.holdout.get('nw_t', 'n/a')}.")
        else:
            lines.append("Not yet checked against the locked holdout.")
        lines += ["", "## Features", "| feature | family | proxy? |", "|---|---|---|"]
        for f in self.feature_names:
            lines.append(f"| {f} | {self.feature_families.get(f, '?')} | "
                        f"{'yes' if f in self.proxy_features else ''} |")
        if self.top_families:
            lines += ["", "## Top families by permutation importance",
                      "| family | mean IC drop |", "|---|---|"]
            for row in self.top_families[:5]:
                lines.append(f"| {row.get('family')} | {row.get('mean_ic_drop', float('nan')):.4f} |")
        lines += ["", "## Known limitations"]
        for lim in self.known_limitations:
            lines.append(f"- {lim}")
        lines += ["", "## Intended use", self.intended_use]
        return "\n".join(lines)


def _default_limitations(feature_names: List[str], confidence: float,
                         mean_ic: Optional[float]) -> List[str]:
    lims = []
    proxies = proxy_features(feature_names)
    if proxies:
        lims.append(
            f"{', '.join(proxies)} stand in for data this engine doesn't actually "
            f"have (see the feature registry's is_proxy flag) — treat their "
            f"contribution as weaker evidence than a genuinely independent feature.")
    if confidence <= 0.05:
        lims.append("Evidence gate is at zero: this model's score is shrunk to a "
                    "constant and contributes no differentiating information to "
                    "the portfolio, regardless of what its raw prediction says.")
    if mean_ic is not None and mean_ic <= 0:
        lims.append("Mean out-of-sample IC is not positive in the research window.")
    lims.append("All research-window metrics are from one historical window; "
               "see docs/PRD_v3_ML.md for the multi-regime validation gap.")
    return lims


def build_model_card(
    alpha_model: AlphaModel,
    model_id: str,
    n_trials: int = 1,
    holdout_result: Optional[object] = None,
    top_families: Optional[pd.DataFrame] = None,
    intended_use: str = DEFAULT_INTENDED_USE,
    extra_limitations: Optional[List[str]] = None,
) -> ModelCard:
    trained = alpha_model.trained
    conf = alpha_model.confidence(n_trials=n_trials)
    fam_map = {f: feature_spec(f).family for f in trained.feature_names}
    holdout = {"checked": False}
    if holdout_result is not None:
        holdout = {"checked": True,
                  "n_holdout_dates": getattr(holdout_result, "n_holdout_dates", None),
                  "nw_t": (getattr(holdout_result, "metrics", {}) or {}).get("nw_t"),
                  "already_checked": getattr(holdout_result, "already_checked", None)}
    limitations = _default_limitations(trained.feature_names, conf,
                                       trained.metrics.get("mean_ic"))
    if extra_limitations:
        limitations = limitations + list(extra_limitations)

    top_fams = []
    if top_families is not None and not top_families.empty:
        top_fams = top_families.sort_values("mean_ic_drop", ascending=False).to_dict("records")

    return ModelCard(
        model_id=model_id, task=trained.task, target_col=trained.target_col,
        horizon_days=trained.horizon_days,
        model_type=trained.model.to_dict().get("type", "unknown"),
        hyperparameters=dict(trained.meta.get("hyperparams", {})),
        feature_names=list(trained.feature_names), feature_families=fam_map,
        proxy_features=proxy_features(trained.feature_names),
        n_trials_in_family=n_trials, research_metrics=dict(trained.metrics),
        confidence=conf, trained_at=trained.meta.get("trained_at"),
        data_window=trained.meta.get("date_range"),
        n_rows=trained.meta.get("n_rows"), n_dates=trained.meta.get("n_dates"),
        holdout=holdout, known_limitations=limitations, intended_use=intended_use,
        top_families=top_fams,
    )


def write_model_card(card: ModelCard, path: str) -> Dict[str, str]:
    """Write ``<path>.card.json`` and ``<path>.card.md``. ``path`` may or may
    not include an extension — either way the outputs land alongside it."""
    base = path[:-5] if path.endswith(".json") else path
    json_path, md_path = f"{base}.card.json", f"{base}.card.md"
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w") as fh:
        json.dump(card.to_dict(), fh, indent=2, default=str)
    with open(md_path, "w") as fh:
        fh.write(card.to_markdown())
    return {"json": json_path, "md": md_path}
