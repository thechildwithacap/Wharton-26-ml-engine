"""Validation protocol: the bar any ML claim in this project must clear.

Three problems this module exists to solve:

1. **Overlapping labels inflate naive t-stats.** A 126-day forward-return label
   observed monthly overlaps its neighbours by ~5 months, so consecutive IC
   observations are autocorrelated and the naive ``std/sqrt(n)`` standard error
   (used by :func:`wharton_ml_engine.ml.metrics.ic_summary`) understates the
   true uncertainty. :func:`newey_west_t` computes a Bartlett-kernel
   heteroskedasticity-and-autocorrelation-consistent (HAC) standard error
   instead — stdlib/numpy only, no scipy dependency (matches the project's
   dependency-light design; scipy isn't installed here).
2. **Trying many variants and reporting only the best one flatters the winner.**
   :func:`log_experiment` / :func:`count_trials` / :func:`adjusted_significance`
   keep a durable record of every variant evaluated (not just the ones that
   worked) and apply a Sidak correction for how many were tried in the same
   family, so a reported significance is never quietly the best of N unlogged
   attempts.
3. **A model can look good on data it was implicitly tuned against.** A single,
   date-based split (:func:`split_research_holdout`) reserves the most recent
   slice of history and never lets research-phase code see it;
   :func:`final_check` is the only function allowed to score it, and it logs
   every call (refusing a silent re-run) so the holdout can't be probed
   repeatedly into overfitting by proxy.

Everything here is generic over the training panel produced by
:mod:`wharton_ml_engine.ml.dataset` — it does not assume any particular model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_REGISTRY_PATH = "reports/experiments/registry.jsonl"
DEFAULT_HOLDOUT_LOG_PATH = "reports/experiments/holdout_log.jsonl"

# Default share of a dataset's rebalance dates reserved as a locked holdout,
# used when the caller doesn't supply an explicit calendar date. A literal
# fixed date (as a specific project's PRD might specify, e.g. "2024-10-01")
# only makes sense once pinned against that project's actual loaded dataset;
# this default instead locks the *last* slice of whatever history is loaded,
# so the mechanism is correct regardless of which dataset (synthetic, a real
# fetch with a different calendar, …) it's pointed at.
DEFAULT_HOLDOUT_FRACTION = 0.15


# ---------------------------------------------------------------------------
# Newey-West HAC standard errors / t-stats (stdlib + numpy only)
# ---------------------------------------------------------------------------

def nw_lags_for_horizon(horizon_days: int, rebalance_days: int = 21) -> int:
    """Suggested Newey-West lag count for a label that overlaps by ``horizon_days``
    when observations are spaced ``rebalance_days`` apart (default: monthly).

    A ``horizon_days``-forward label observed every ``rebalance_days`` overlaps
    its neighbours for roughly ``ceil(horizon_days / rebalance_days) - 1``
    subsequent observations — that many lags of autocorrelation should be
    accounted for.
    """
    if horizon_days <= 0 or rebalance_days <= 0:
        return 0
    return max(0, math.ceil(horizon_days / rebalance_days) - 1)


def newey_west_t(values: Sequence[float], lags: Optional[int] = None) -> Dict[str, float]:
    """HAC (Newey-West, Bartlett kernel) standard error and t-stat for the mean
    of ``values`` being different from zero.

    Falls back to the plain (non-HAC) standard error when ``lags=0`` — which
    reduces to the ordinary one-sample t-test, so this function is a strict
    superset of the naive calculation, never a different one at lag 0.
    """
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = len(x)
    if n < 2:
        return {"mean": float("nan"), "nw_se": float("nan"), "nw_t": float("nan"),
                "nw_lags": 0, "n_periods": n}
    lags = 0 if lags is None else max(0, int(lags))
    lags = min(lags, n - 1)                          # can't exceed the sample

    mean = float(x.mean())
    d = x - mean
    gamma0 = float(np.dot(d, d)) / n
    long_run_var = gamma0
    for k in range(1, lags + 1):
        gamma_k = float(np.dot(d[k:], d[:-k])) / n
        weight = 1.0 - k / (lags + 1)                # Bartlett kernel
        long_run_var += 2.0 * weight * gamma_k
    long_run_var = max(long_run_var, 0.0)             # guard tiny negative rounding
    nw_se = math.sqrt(long_run_var / n) if long_run_var > 0 else float("nan")
    nw_t = mean / nw_se if nw_se and np.isfinite(nw_se) and nw_se > 0 else float("nan")
    return {"mean": mean, "nw_se": nw_se, "nw_t": nw_t, "nw_lags": lags, "n_periods": n}


def ic_summary_nw(ic_series: pd.Series, horizon_days: Optional[int] = None,
                  rebalance_days: int = 21, lags: Optional[int] = None) -> Dict[str, float]:
    """Newey-West-corrected companion to :func:`wharton_ml_engine.ml.metrics.ic_summary`.

    Returns HAC fields (``nw_t``, ``nw_se``, ``nw_lags``) alongside a
    ``n_effective_periods`` estimate (``n_periods / (nw_lags + 1)``) — the
    rough number of *independent* observations behind the t-stat, so a short
    holdout with a long horizon doesn't get over-read (the PRD's own
    instruction: report this next to the t whenever the sample is short).
    """
    ic = ic_series.dropna()
    if lags is None:
        lags = nw_lags_for_horizon(horizon_days, rebalance_days) if horizon_days else 0
    nw = newey_west_t(ic.to_numpy(), lags=lags)
    n = nw["n_periods"]
    return {
        "mean_ic": nw["mean"], "nw_se": nw["nw_se"], "nw_t": nw["nw_t"],
        "nw_lags": nw["nw_lags"], "n_periods": n,
        "n_effective_periods": (n / (nw["nw_lags"] + 1)) if n else 0,
        "hit_rate": float((ic > 0).mean()) if n else float("nan"),
    }


# ---------------------------------------------------------------------------
# Multiple-testing correction (stdlib only: erf-based normal CDF + bisection)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF via bisection (no scipy dependency)."""
    if not (0.0 < p < 1.0):
        return float("nan") if p not in (0.0, 1.0) else (float("-inf") if p == 0.0 else float("inf"))
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def two_sided_p(t: float) -> float:
    if not np.isfinite(t):
        return float("nan")
    return 2.0 * (1.0 - _norm_cdf(abs(t)))


def adjusted_significance(t: float, n_trials: int) -> Dict[str, float]:
    """Sidak-adjusted significance of a t-stat given how many variants were tried.

    ``p_adj = 1 - (1 - p) ** n_trials`` is the Sidak correction: the chance that
    *at least one* of ``n_trials`` independent tries would look this good by
    luck alone. ``t_adj`` converts that adjusted p-value back to an equivalent
    (necessarily smaller in magnitude) t-stat, so it can be compared directly
    against the un-adjusted number.
    """
    n_trials = max(1, int(n_trials))
    p = two_sided_p(t)
    if not np.isfinite(p):
        return {"t": t, "p": float("nan"), "p_adj": float("nan"), "t_adj": float("nan"),
                "n_trials": n_trials}
    p = min(max(p, 0.0), 1.0)
    p_adj = 1.0 - (1.0 - p) ** n_trials
    p_adj = min(max(p_adj, 1e-300), 1.0 - 1e-16)
    z_adj = _norm_ppf(1.0 - p_adj / 2.0)
    t_adj = math.copysign(z_adj, t) if np.isfinite(z_adj) else float("nan")
    return {"t": t, "p": p, "p_adj": p_adj, "t_adj": t_adj, "n_trials": n_trials}


# Reference bar from Harvey, Liu & Zhu (2016) for a single-tested factor under
# heavy multiple-testing assumptions across the whole literature — reported as
# a stricter external benchmark, never used as the pass/fail gate itself.
HARVEY_LIU_ZHU_T_BAR = 3.0


# ---------------------------------------------------------------------------
# Locked holdout
# ---------------------------------------------------------------------------

def split_research_holdout(
    dates: Sequence[pd.Timestamp],
    holdout_start: Optional[pd.Timestamp] = None,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split a sorted sequence of rebalance dates into (research, holdout).

    ``holdout_start`` locks an explicit calendar date (use this once you know
    your dataset's real calendar and have decided a date in advance — deciding
    it *after* looking at where the interesting results are defeats the whole
    point). Without it, the last ``holdout_fraction`` of dates is reserved.
    """
    idx = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in dates))
    if len(idx) == 0:
        return idx, idx
    if holdout_start is not None:
        cutoff = pd.Timestamp(holdout_start)
    else:
        n_holdout = max(1, int(round(len(idx) * holdout_fraction)))
        cutoff = idx[-n_holdout]
    research = idx[idx < cutoff]
    holdout = idx[idx >= cutoff]
    return research, holdout


def research_only_panel(panel: pd.DataFrame, holdout_start: Optional[pd.Timestamp] = None,
                        holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
                        date_col: str = "date") -> pd.DataFrame:
    """Drop holdout-period rows from a training panel — the default for every
    research-phase call (tuning, model comparison, feature selection)."""
    dates = panel[date_col].unique()
    research, _holdout = split_research_holdout(dates, holdout_start, holdout_fraction)
    return panel[panel[date_col].isin(set(research))].reset_index(drop=True)


def holdout_only_panel(panel: pd.DataFrame, holdout_start: Optional[pd.Timestamp] = None,
                       holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
                       date_col: str = "date") -> pd.DataFrame:
    dates = panel[date_col].unique()
    _research, holdout = split_research_holdout(dates, holdout_start, holdout_fraction)
    return panel[panel[date_col].isin(set(holdout))].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _spec_hash(spec: Dict[str, object]) -> str:
    blob = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def log_experiment(
    spec: Dict[str, object],
    metrics: Dict[str, object],
    family: str,
    registry_path: str = DEFAULT_REGISTRY_PATH,
) -> Dict[str, object]:
    """Append one experiment (evaluated variant) to the registry. Never skips a
    failed/unremarkable result — the point is counting every trial in a family,
    not just the ones worth bragging about."""
    record = {
        "id": _spec_hash({**spec, "family": family}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "family": family,
        "spec": spec,
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(registry_path) or ".", exist_ok=True)
    with open(registry_path, "a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    return record


def read_registry(registry_path: str = DEFAULT_REGISTRY_PATH) -> List[Dict[str, object]]:
    if not os.path.exists(registry_path):
        return []
    out = []
    with open(registry_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def count_trials(family: str, registry_path: str = DEFAULT_REGISTRY_PATH) -> int:
    return sum(1 for r in read_registry(registry_path) if r.get("family") == family)


# ---------------------------------------------------------------------------
# Locked-holdout final check
# ---------------------------------------------------------------------------

@dataclass
class HoldoutResult:
    spec_id: str
    metrics: Dict[str, object]
    n_holdout_dates: int
    already_checked: bool


def final_check(
    spec: Dict[str, object],
    train_panel: pd.DataFrame,
    fit_fn,
    predict_fn,
    family: str,
    holdout_start: Optional[pd.Timestamp] = None,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    horizon_days: Optional[int] = None,
    holdout_log_path: str = DEFAULT_HOLDOUT_LOG_PATH,
    force_recheck: bool = False,
) -> HoldoutResult:
    """Score the locked holdout **once** for a given model spec.

    ``fit_fn(research_panel) -> model`` trains on the research split only.
    ``predict_fn(model, holdout_panel) -> pd.Series`` (indexed like
    ``holdout_panel``) returns predictions for the holdout rows. This function
    computes rank IC per holdout date, a Newey-West t-stat, and logs the result
    to ``holdout_log_path`` keyed by a hash of ``spec`` — a second call for the
    same spec is refused (returns the prior logged result, unless
    ``force_recheck=True``, which still logs a loud second entry rather than
    overwriting the first).
    """
    from .metrics import per_date_ic

    spec_id = _spec_hash({**spec, "family": family})
    prior = [r for r in read_registry(holdout_log_path) if r.get("spec_id") == spec_id]
    if prior and not force_recheck:
        return HoldoutResult(spec_id=spec_id, metrics=prior[-1]["metrics"],
                             n_holdout_dates=prior[-1].get("n_holdout_dates", 0),
                             already_checked=True)

    research = research_only_panel(train_panel, holdout_start, holdout_fraction)
    holdout = holdout_only_panel(train_panel, holdout_start, holdout_fraction)
    if research.empty or holdout.empty:
        raise ValueError("research or holdout split is empty — check your date range")

    model = fit_fn(research)
    preds = predict_fn(model, holdout)
    frame = holdout[["date", "ticker", "fwd_return"]].copy()
    frame["pred"] = pd.Series(preds).reindex(frame.index)
    ic = per_date_ic(frame, "pred", "fwd_return", method="rank")
    nw = ic_summary_nw(ic, horizon_days=horizon_days)

    record = {
        "spec_id": spec_id, "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(), "family": family, "spec": spec,
        "metrics": nw, "n_holdout_dates": int(len(ic.dropna())),
        "forced_recheck": bool(force_recheck and prior),
    }
    os.makedirs(os.path.dirname(holdout_log_path) or ".", exist_ok=True)
    with open(holdout_log_path, "a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    if force_recheck and prior:
        print(f"WARNING: re-checked holdout for spec {spec_id} "
              f"({len(prior)} prior check(s) already logged) — this weakens "
              f"the holdout's honesty guarantee; only do this deliberately.")
    return HoldoutResult(spec_id=spec_id, metrics=nw,
                         n_holdout_dates=int(len(ic.dropna())), already_checked=False)


# ---------------------------------------------------------------------------
# IC breakdowns (G7)
# ---------------------------------------------------------------------------

def ic_breakdown_by_year(oos_frame: pd.DataFrame, pred_col: str = "pred",
                         actual_col: str = "fwd_return") -> pd.DataFrame:
    from .metrics import per_date_ic

    df = oos_frame.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    rows = []
    for year, grp in df.groupby("year"):
        ic = per_date_ic(grp, pred_col, actual_col)
        nw = ic_summary_nw(ic)
        rows.append({"year": int(year), **nw})
    return pd.DataFrame(rows).set_index("year").sort_index()


def ic_breakdown_by_group(oos_frame: pd.DataFrame, group_map: Dict[str, str],
                          pred_col: str = "pred", actual_col: str = "fwd_return",
                          group_col_name: str = "group") -> pd.DataFrame:
    """Generic breakdown by any ticker -> group mapping (sector, cap bucket, …)."""
    from .metrics import per_date_ic

    df = oos_frame.copy()
    df[group_col_name] = df["ticker"].map(group_map)
    rows = []
    for group, grp in df.groupby(group_col_name):
        if pd.isna(group):
            continue
        ic = per_date_ic(grp, pred_col, actual_col)
        nw = ic_summary_nw(ic)
        rows.append({group_col_name: group, **nw})
    return pd.DataFrame(rows).set_index(group_col_name).sort_index()


def ic_breakdown_by_regime(oos_frame: pd.DataFrame, bundle, pred_col: str = "pred",
                           actual_col: str = "fwd_return") -> pd.DataFrame:
    """IC broken down by :func:`wharton_ml_engine.signals.macro.macro_regime`'s
    label (risk-on / neutral / risk-off) as of each date — reuses the regime
    classifier already built and tested in this project rather than requiring
    a new external macro data source."""
    from ..signals.macro import macro_regime
    from .metrics import per_date_ic

    df = oos_frame.copy()
    labels = {}
    for d in df["date"].unique():
        try:
            labels[d] = macro_regime(bundle, pd.Timestamp(d)).label()
        except Exception:
            labels[d] = "unknown"
    df["regime"] = df["date"].map(labels)
    rows = []
    for regime, grp in df.groupby("regime"):
        ic = per_date_ic(grp, pred_col, actual_col)
        nw = ic_summary_nw(ic)
        rows.append({"regime": regime, **nw})
    return pd.DataFrame(rows).set_index("regime").sort_index()
