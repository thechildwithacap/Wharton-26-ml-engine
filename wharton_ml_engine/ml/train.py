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
from .dataset import ML_FEATURES, build_training_panel
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
    **hp,
) -> Dict[str, object]:
    """Expanding-window out-of-sample evaluation.  Returns OOS preds + metrics."""
    feature_names = feature_names or ML_FEATURES
    target_col = "outperform" if task == "classification" else "fwd_excess"

    dates = np.array(sorted(panel["date"].unique()))
    P = len(dates)
    if P <= min_train_periods + 1:
        return {"oos": pd.DataFrame(), "ic": pd.Series(dtype=float),
                "summary": ic_summary(pd.Series(dtype=float))}

    test_start_idx = min_train_periods
    remaining = P - test_start_idx
    block = max(1, int(np.ceil(remaining / n_splits)))

    oos_rows: List[pd.DataFrame] = []
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

    if not oos_rows:
        return {"oos": pd.DataFrame(), "ic": pd.Series(dtype=float),
                "summary": ic_summary(pd.Series(dtype=float))}

    oos = pd.concat(oos_rows, ignore_index=True)
    ic = per_date_ic(oos, "pred", "fwd_return", method="rank")
    summary = ic_summary(ic)
    if task == "classification":
        summary["oos_accuracy"] = accuracy((oos["pred"] >= 0.5).astype(int).to_numpy(),
                                           oos["outperform"].to_numpy())
    else:
        summary["oos_r2"] = r2_score(oos["pred"].to_numpy(), oos["fwd_excess"].to_numpy())
    return {"oos": oos, "ic": ic, "summary": summary}


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
    **hp,
) -> TrainedModel:
    """Build the panel, evaluate walk-forward, then fit the final live model."""
    feature_names = feature_names or ML_FEATURES
    target_col = "outperform" if task == "classification" else "fwd_excess"
    if panel is None:
        panel = build_training_panel(bundle, horizon_days=horizon_days,
                                     start=start, end=end)
    if panel.empty:
        raise ValueError("training panel is empty — need more history")

    evaluation = walk_forward_evaluate(
        panel, task=task, feature_names=feature_names, n_splits=n_splits,
        min_train_periods=min_train_periods, **hp,
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
    }
    return TrainedModel(model=model, scaler=scaler, feature_names=feature_names,
                        task=task, target_col=target_col, horizon_days=horizon_days,
                        metrics=metrics, meta=meta)
