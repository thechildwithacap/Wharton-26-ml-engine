"""Evaluation metrics for cross-sectional alpha models.

The workhorse metric for stock-selection models is the **Information
Coefficient (IC)** — the correlation between predicted scores and realised
forward returns, measured *cross-sectionally per date* and then averaged.  Rank
IC (Spearman) is preferred because we care about ordering names, not hitting an
exact return.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def pearson_ic(pred: pd.Series, actual: pd.Series) -> float:
    df = pd.concat([pred, actual], axis=1).dropna()
    if len(df) < 3 or df.iloc[:, 0].std(ddof=0) == 0 or df.iloc[:, 1].std(ddof=0) == 0:
        return float("nan")
    return float(df.iloc[:, 0].corr(df.iloc[:, 1]))


def rank_ic(pred: pd.Series, actual: pd.Series) -> float:
    df = pd.concat([pred, actual], axis=1).dropna()
    if len(df) < 3:
        return float("nan")
    return float(df.iloc[:, 0].rank().corr(df.iloc[:, 1].rank()))


def r2_score(pred: np.ndarray, actual: np.ndarray) -> float:
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def accuracy(pred_label: np.ndarray, actual_label: np.ndarray) -> float:
    pred_label, actual_label = np.asarray(pred_label), np.asarray(actual_label)
    if len(actual_label) == 0:
        return float("nan")
    return float((pred_label == actual_label).mean())


def per_date_ic(frame: pd.DataFrame, pred_col: str, actual_col: str,
                date_col: str = "date", method: str = "rank") -> pd.Series:
    """IC computed within each date, returned as a date-indexed series."""
    fn = rank_ic if method == "rank" else pearson_ic
    out = {}
    for d, grp in frame.groupby(date_col):
        out[d] = fn(grp[pred_col], grp[actual_col])
    return pd.Series(out).sort_index()


def ic_summary(ic_series: pd.Series) -> Dict[str, float]:
    """Mean IC, its standard error, the IC information ratio and hit rate.

    The IC information ratio (mean / std) annualised is the standard measure of
    a signal's consistency; ``ic_t_stat`` tests whether mean IC differs from 0.
    """
    ic = ic_series.dropna()
    n = len(ic)
    if n == 0:
        return {"mean_ic": float("nan"), "ic_std": float("nan"),
                "ic_ir": float("nan"), "ic_t_stat": float("nan"),
                "hit_rate": float("nan"), "n_periods": 0}
    mean, std = float(ic.mean()), float(ic.std(ddof=1)) if n > 1 else float("nan")
    ir = mean / std if std and np.isfinite(std) and std > 0 else float("nan")
    return {
        "mean_ic": mean,
        "ic_std": std,
        "ic_ir": ir,
        "ic_t_stat": ir * np.sqrt(n) if np.isfinite(ir) else float("nan"),
        "hit_rate": float((ic > 0).mean()),
        "n_periods": n,
    }
