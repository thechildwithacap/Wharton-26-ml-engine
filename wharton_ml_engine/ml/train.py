"""Walk-forward training and evaluation of the alpha model.

Evaluation is **walk-forward** (expanding window) with a purge gap so a model is
only ever tested on dates strictly *after* the data it trained on — the honest
way to estimate a stock-selection signal.  The scaler is fit on the training
fold only.  After evaluation, a final model is fit on all available history for
live scoring, and it carries the out-of-sample metrics with it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.source import DataBundle
from .dataset import ML_FEATURES, build_training_panel, feature_columns
from .metrics import accuracy, ic_summary, per_date_ic, r2_score
from .models import (
    LogisticRegressor,
    RidgeRegressor,
    StandardScaler,
    model_from_dict,
)


@dataclass
class TrainedModel:
    model: object
    scaler: StandardScaler
    feature_names: List[str]
    task: str                       # "regression" | "classification"
    target_col: str
    horizon_days: int
    metrics: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    # ---- scoring ------------------------------------------------------------
    def raw_predict(self, features: pd.DataFrame) -> pd.Series:
        X = self.scaler.transform(features[self.feature_names].to_numpy())
        if self.task == "classification":
            vals = self.model.predict_proba(X)
        else:
            vals = self.model.predict(X)
        return pd.Series(vals, index=features.index)

    def coefficients(self) -> pd.Series:
        if getattr(self.model, "coef_", None) is None:
            return pd.Series(dtype=float)
        return pd.Series(self.model.coef_, index=self.feature_names).sort_values(
            key=lambda s: s.abs(), ascending=False)

    # ---- persistence (JSON, no pickle) --------------------------------------
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "model": self.model.to_dict(),
            "scaler": self.scaler.to_dict(),
            "feature_names": self.feature_names,
            "task": self.task,
            "target_col": self.target_col,
            "horizon_days": self.horizon_days,
            "metrics": self.metrics,
            "meta": self.meta,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "TrainedModel":
        with open(path) as fh:
            p = json.load(fh)
        return cls(
            model=model_from_dict(p["model"]),
            scaler=StandardScaler.from_dict(p["scaler"]),
            feature_names=list(p["feature_names"]),
            task=p["task"],
            target_col=p["target_col"],
            horizon_days=int(p["horizon_days"]),
            metrics=p.get("metrics", {}),
            meta=p.get("meta", {}),
        )


def _make_model(task: str, **hp):
    if task == "classification":
        return LogisticRegressor(l2=hp.get("l2", 1.0), lr=hp.get("lr", 0.2),
                                 epochs=hp.get("epochs", 500))
    return RidgeRegressor(alpha=hp.get("alpha", 10.0))


def walk_forward_evaluate(
    panel: pd.DataFrame,
    task: str,
    feature_names: Optional[List[str]] = None,
    n_splits: int = 5,
    min_train_periods: int = 12,
    purge_periods: int = 1,
    horizon_days: Optional[int] = None,
    research_only: bool = False,
    holdout_start: Optional[pd.Timestamp] = None,
    holdout_fraction: Optional[float] = None,
    return_models: bool = False,
    **hp,
) -> Dict[str, object]:
    """Expanding-window out-of-sample evaluation.  Returns OOS preds + metrics.

    ``horizon_days`` is the forward-return label horizon the panel was built
    with (see :func:`wharton_ml_engine.ml.dataset.build_training_panel`) — used
    only to pick a Newey-West lag count for ``nw_t`` in the returned summary; it
    never changes which rows are evaluated.

    ``research_only=True`` drops any locked-holdout dates from ``panel``
    *before* the walk-forward split, so a research/tuning run can never see
    them. Default is ``False`` (unchanged historical behaviour, evaluates every
    date in ``panel``) so existing callers are unaffected; pass
    ``research_only=True`` explicitly wherever holdout discipline matters (see
    :mod:`wharton_ml_engine.ml.validation`). Only
    :func:`wharton_ml_engine.ml.validation.final_check` is meant to evaluate the
    holdout itself, exactly once.

    ``return_models=True`` additionally returns ``"folds"``: a list of
    ``{"scaler", "model", "test_dates", "feature_names"}`` for each walk-forward
    fold, so callers (e.g. :mod:`.explain`'s permutation importance) can re-score
    a fold's already-fitted model on perturbed test data without ever fitting on
    that fold's own test dates — genuine out-of-sample permutation importance,
    not an in-sample one dressed up as OOS.
    """
    feature_names = feature_names or ML_FEATURES
    target_col = "outperform" if task == "classification" else "fwd_excess"

    if research_only and "date" in panel.columns and not panel.empty:
        from .validation import DEFAULT_HOLDOUT_FRACTION, research_only_panel
        panel = research_only_panel(
            panel, holdout_start=holdout_start,
            holdout_fraction=(holdout_fraction if holdout_fraction is not None
                              else DEFAULT_HOLDOUT_FRACTION))

    dates = np.array(sorted(panel["date"].unique()))
    P = len(dates)
    if P <= min_train_periods + 1:
        return {"oos": pd.DataFrame(), "ic": pd.Series(dtype=float),
                "summary": ic_summary(pd.Series(dtype=float))}

    test_start_idx = min_train_periods
    remaining = P - test_start_idx
    block = max(1, int(np.ceil(remaining / n_splits)))

    oos_rows: List[pd.DataFrame] = []
    folds: List[Dict[str, object]] = []
    for b in range(test_start_idx, P, block):
        test_dates = dates[b:b + block]
        cutoff = dates[b - purge_periods] if b - purge_periods > 0 else dates[0]
        train = panel[panel["date"] < cutoff]
        test = panel[panel["date"].isin(test_dates)]
        if len(train) < 30 or test.empty:
            continue

        scaler = StandardScaler().fit(train[feature_names].to_numpy())
        Xtr = scaler.transform(train[feature_names].to_numpy())
        model = _make_model(task, **hp)
        model.fit(Xtr, train[target_col].to_numpy())

        Xte = scaler.transform(test[feature_names].to_numpy())
        pred = (model.predict_proba(Xte) if task == "classification"
                else model.predict(Xte))
        rec = test[["date", "ticker", "fwd_return", "fwd_excess", "outperform"]].copy()
        rec["pred"] = pred
        oos_rows.append(rec)
        if return_models:
            folds.append({"scaler": scaler, "model": model,
                         "test_dates": list(test_dates),
                         "feature_names": list(feature_names)})

    if not oos_rows:
        result = {"oos": pd.DataFrame(), "ic": pd.Series(dtype=float),
                 "summary": ic_summary(pd.Series(dtype=float))}
        if return_models:
            result["folds"] = folds
        return result

    oos = pd.concat(oos_rows, ignore_index=True)
    ic = per_date_ic(oos, "pred", "fwd_return", method="rank")
    summary = ic_summary(ic)
    # Newey-West-corrected significance alongside the naive one — metrics.
    # ic_summary's ic_t_stat understates uncertainty for overlapping labels
    # (see ml/validation.py). Best-effort: never blocks core evaluation.
    try:
        from .validation import ic_summary_nw
        summary.update(ic_summary_nw(ic, horizon_days=horizon_days))
    except Exception:
        pass
    if task == "classification":
        summary["oos_accuracy"] = accuracy((oos["pred"] >= 0.5).astype(int).to_numpy(),
                                           oos["outperform"].to_numpy())
    else:
        summary["oos_r2"] = r2_score(oos["pred"].to_numpy(), oos["fwd_excess"].to_numpy())
    result = {"oos": oos, "ic": ic, "summary": summary}
    if return_models:
        result["folds"] = folds
    return result


def train_alpha_model(
    bundle: DataBundle,
    task: str = "regression",
    horizon_days: int = 21,
    feature_names: Optional[List[str]] = None,
    n_splits: int = 5,
    min_train_periods: int = 12,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    panel: Optional[pd.DataFrame] = None,
    extended: bool = False,
    **hp,
) -> TrainedModel:
    """Build the panel, evaluate walk-forward, then fit the final live model.

    ``extended=True`` trains on the full SEC fundamental feature set (raw ranked
    line items) in addition to the composite style scores.
    """
    if feature_names is None:
        feature_names = feature_columns(extended)
    target_col = "outperform" if task == "classification" else "fwd_excess"
    if panel is None:
        panel = build_training_panel(bundle, horizon_days=horizon_days,
                                     start=start, end=end, extended=extended)
    if panel.empty:
        raise ValueError("training panel is empty — need more history")

    evaluation = walk_forward_evaluate(
        panel, task=task, feature_names=feature_names, n_splits=n_splits,
        min_train_periods=min_train_periods, horizon_days=horizon_days, **hp,
    )

    # Final fit on all data for live scoring.
    scaler = StandardScaler().fit(panel[feature_names].to_numpy())
    X = scaler.transform(panel[feature_names].to_numpy())
    model = _make_model(task, **hp)
    model.fit(X, panel[target_col].to_numpy())

    metrics = {k: v for k, v in evaluation["summary"].items()}
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(panel)),
        "n_dates": int(panel["date"].nunique()),
        "date_range": [str(panel["date"].min()), str(panel["date"].max())],
        "hyperparams": hp,
        "n_splits": n_splits,
        "extended": bool(any(str(f).startswith("f_") for f in feature_names)),
        "n_features": len(feature_names),
    }
    return TrainedModel(model=model, scaler=scaler, feature_names=feature_names,
                        task=task, target_col=target_col, horizon_days=horizon_days,
                        metrics=metrics, meta=meta)
