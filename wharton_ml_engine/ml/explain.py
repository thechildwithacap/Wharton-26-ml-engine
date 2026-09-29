"""Interpretability: local per-stock decompositions and global model behaviour.

Two levels:

* **Local** (:func:`explain`) — for a specific model and decision date, every
  scored stock's raw score decomposes exactly into an intercept plus one
  contribution per feature (``coef_j * z_ij`` for the linear ridge/logistic
  models this project uses). Contributions always sum to the raw score to
  numerical precision — tested, never just claimed.
* **Global** (:func:`global_explain`) — coefficient stability across
  walk-forward refits, permutation importance measured on genuinely
  out-of-sample folds (never the final model scored on its own training data),
  a single-feature IC table cross-checked against each feature's registered
  expected sign, and IC broken down by year/regime/group (reusing
  :mod:`.validation`'s breakdown functions).

Only linear models (Ridge/Logistic — the only model families this codebase
trains; see ``ml/models.py``) are supported. A tree-ensemble path (LightGBM
`pred_contrib`, as a future PRD workstream describes) is explicitly NOT
implemented here — there's no such model in this codebase to explain, and
adding one without this decomposition path would violate the project's
explainability requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.source import DataBundle
from .dataset import feature_frame, is_extended
from .features import family_of, proxy_features, spec as feature_spec
from .predict import AlphaModel
from .train import TrainedModel


# ---------------------------------------------------------------------------
# Local explanations
# ---------------------------------------------------------------------------

@dataclass
class Explanation:
    as_of: pd.Timestamp
    model_id: str
    intercept: float
    local: pd.DataFrame            # long: ticker, feature, family, raw_value, z_value, contribution, rank_in_universe
    family_rollup: pd.DataFrame    # wide: ticker x family, plus intercept + total
    is_classification: bool = False
    meta: Dict[str, object] = field(default_factory=dict)

    def for_ticker(self, ticker: str) -> pd.DataFrame:
        return self.local[self.local["ticker"] == ticker].sort_values(
            "contribution", key=lambda s: s.abs(), ascending=False)

    def top_contributors(self, ticker: str, n: int = 3, positive: bool = True) -> pd.DataFrame:
        rows = self.for_ticker(ticker)
        rows = rows[rows["contribution"] > 0] if positive else rows[rows["contribution"] < 0]
        return rows.sort_values("contribution", key=lambda s: s.abs(), ascending=False).head(n)


def _linear_contributions(trained: TrainedModel, feats: pd.DataFrame):
    """coef_j * z_ij per (ticker, feature), plus the intercept and raw total.

    Works identically for RidgeRegressor and LogisticRegressor — both expose
    ``coef_``/``intercept_`` over a strictly linear pre-activation. For a
    classification model the "raw total" is the pre-sigmoid logit; the caller
    is responsible for applying the sigmoid if a probability is wanted (kept
    separate so the additive decomposition stays exact).

    Returns ``(contributions_df, Z, intercept, raw_total)`` where
    ``contributions_df`` has one column per feature name and ``Z`` is the
    matching standardised feature matrix.
    """
    model = trained.model
    coef = np.asarray(model.coef_, dtype=float)
    intercept = float(model.intercept_)
    X = feats[trained.feature_names].to_numpy(dtype=float)
    Z = trained.scaler.transform(X)
    contributions = Z * coef[None, :]
    raw_total = contributions.sum(axis=1) + intercept
    contributions_df = pd.DataFrame(contributions, index=feats.index,
                                    columns=trained.feature_names)
    return contributions_df, Z, intercept, raw_total


def explain(alpha_model: AlphaModel, bundle: DataBundle,
           as_of: Optional[pd.Timestamp] = None,
           tickers: Optional[List[str]] = None,
           model_id: str = "model") -> Explanation:
    """Decompose ``alpha_model``'s score for every (or the given) ticker at
    ``as_of`` into named, additive feature contributions.

    Uses the model's raw, *ungated* prediction (``AlphaModel.score(gate=False)``
    territory) — an evidence-gated score is a deliberately shrunk proxy for the
    model's real output, and decomposing that would misrepresent what the model
    actually learned. The gate itself (and why it's on/off) belongs in the
    narrated explanation (see :mod:`.narrate`), not in the numeric decomposition.
    """
    trained = alpha_model.trained
    if as_of is None:
        as_of = bundle.dates()[-1]
    as_of = pd.Timestamp(as_of)
    ext = is_extended(trained.feature_names)
    feats = feature_frame(bundle, as_of, extended=ext)
    if tickers is not None:
        feats = feats.reindex([t for t in tickers if t in feats.index])
    feats = feats.dropna(how="any")

    contributions_df, Z, intercept, raw_total = _linear_contributions(trained, feats)
    fam_map = family_of(trained.feature_names)

    long_rows = []
    for j, f in enumerate(trained.feature_names):
        long_rows.append(pd.DataFrame({
            "ticker": feats.index, "feature": f, "family": fam_map[f],
            "raw_value": feats[f].to_numpy(), "z_value": Z[:, j],
            "contribution": contributions_df[f].to_numpy(),
        }))
    local = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame(
        columns=["ticker", "feature", "family", "raw_value", "z_value", "contribution"])
    raw_total_s = pd.Series(raw_total, index=feats.index)
    local["rank_in_universe"] = local["ticker"].map(
        raw_total_s.rank(ascending=False, method="min").astype(int))

    rollup = local.pivot_table(index="ticker", columns="family", values="contribution",
                               aggfunc="sum", fill_value=0.0)
    rollup["intercept"] = intercept
    family_cols = [c for c in rollup.columns if c != "intercept"]
    rollup["total"] = rollup[family_cols].sum(axis=1) + rollup["intercept"]

    return Explanation(
        as_of=as_of, model_id=model_id, intercept=intercept, local=local,
        family_rollup=rollup, is_classification=(trained.task == "classification"),
        meta={"feature_names": list(trained.feature_names),
             "proxy_features": proxy_features(trained.feature_names),
             "n_tickers": int(len(feats))},
    )


# ---------------------------------------------------------------------------
# Global explanations
# ---------------------------------------------------------------------------

def coefficient_stability(panel: pd.DataFrame, task: str,
                          feature_names: Optional[List[str]] = None,
                          n_splits: int = 5, min_train_periods: int = 12,
                          **hp) -> pd.DataFrame:
    """Sign stability of every feature's coefficient across walk-forward refits.

    ``sign_stability`` is the share of folds where the fold's coefficient sign
    matches the sign of the coefficient fit on the *entire* panel — a feature
    whose sign flips fold to fold is not a feature the model can be trusted to
    use consistently, whatever its average coefficient says.
    """
    from .dataset import ML_FEATURES
    from .train import walk_forward_evaluate, _make_model
    from .models import StandardScaler

    feature_names = feature_names or ML_FEATURES
    result = walk_forward_evaluate(panel, task=task, feature_names=feature_names,
                                   n_splits=n_splits, min_train_periods=min_train_periods,
                                   return_models=True, **hp)
    folds = result.get("folds", [])

    target_col = "outperform" if task == "classification" else "fwd_excess"
    scaler = StandardScaler().fit(panel[feature_names].to_numpy())
    Xall = scaler.transform(panel[feature_names].to_numpy())
    full_model = _make_model(task, **hp)
    full_model.fit(Xall, panel[target_col].to_numpy())
    full_sign = np.sign(full_model.coef_)

    rows = []
    coefs_by_feature = {f: [] for f in feature_names}
    for fold in folds:
        for j, f in enumerate(feature_names):
            coefs_by_feature[f].append(float(fold["model"].coef_[j]))
    for j, f in enumerate(feature_names):
        vals = np.array(coefs_by_feature[f])
        if len(vals) == 0:
            rows.append({"feature": f, "family": feature_spec(f).family,
                        "mean_coef": float("nan"), "std_coef": float("nan"),
                        "sign_stability": float("nan"), "n_folds": 0})
            continue
        matches = float(np.mean(np.sign(vals) == full_sign[j])) if full_sign[j] != 0 else float("nan")
        rows.append({"feature": f, "family": feature_spec(f).family,
                    "mean_coef": float(vals.mean()), "std_coef": float(vals.std(ddof=0)),
                    "sign_stability": matches, "n_folds": len(vals)})
    return pd.DataFrame(rows).set_index("feature")


def permutation_importance(panel: pd.DataFrame, task: str,
                           feature_names: Optional[List[str]] = None,
                           n_splits: int = 5, min_train_periods: int = 12,
                           n_repeats: int = 3, seed: int = 0,
                           by_family: bool = False, **hp) -> pd.DataFrame:
    """Drop in mean out-of-sample rank IC when a feature (or its whole family)
    is shuffled *within each test date's cross-section* of an already-fitted
    walk-forward fold.

    Genuinely out-of-sample: each fold's model was fit before its own test
    dates existed to it, and permutation happens only on that fold's held-out
    rows — never on data the fold's model was trained on. This is the honest
    "what does the model actually lean on" number, as distinct from the raw
    coefficient magnitude (which is confounded by feature scale/correlation).
    """
    from .dataset import ML_FEATURES
    from .metrics import per_date_ic
    from .train import walk_forward_evaluate

    feature_names = feature_names or ML_FEATURES
    result = walk_forward_evaluate(panel, task=task, feature_names=feature_names,
                                   n_splits=n_splits, min_train_periods=min_train_periods,
                                   return_models=True, **hp)
    folds = result.get("folds", [])
    if not folds:
        cols = ["family"] if by_family else ["feature", "family"]
        return pd.DataFrame(columns=[*cols, "baseline_ic", "mean_ic_drop", "n_folds"])

    rng = np.random.default_rng(seed)
    groups: Dict[str, List[str]] = {}
    if by_family:
        for f in feature_names:
            groups.setdefault(feature_spec(f).family, []).append(f)
    else:
        for f in feature_names:
            groups[f] = [f]

    baseline_ic_by_fold = []
    for fold in folds:
        test = panel[panel["date"].isin(fold["test_dates"])]
        if test.empty:
            continue
        Xte = fold["scaler"].transform(test[fold["feature_names"]].to_numpy())
        pred = (fold["model"].predict_proba(Xte) if task == "classification"
               else fold["model"].predict(Xte))
        rec = test[["date", "ticker", "fwd_return"]].copy()
        rec["pred"] = pred
        ic = per_date_ic(rec, "pred", "fwd_return", method="rank")
        baseline_ic_by_fold.append(float(ic.mean()) if len(ic.dropna()) else float("nan"))
    baseline_ic = float(np.nanmean(baseline_ic_by_fold)) if baseline_ic_by_fold else float("nan")

    rows = []
    for group_name, members in groups.items():
        idxs = [feature_names.index(m) for m in members]
        drops = []
        for fold in folds:
            test = panel[panel["date"].isin(fold["test_dates"])]
            if test.empty:
                continue
            for _ in range(n_repeats):
                perturbed = test.copy()
                for d, grp in perturbed.groupby("date"):
                    shuffled = grp[members].sample(frac=1.0, random_state=int(rng.integers(1e9))).to_numpy()
                    perturbed.loc[grp.index, members] = shuffled
                Xte = fold["scaler"].transform(perturbed[fold["feature_names"]].to_numpy())
                pred = (fold["model"].predict_proba(Xte) if task == "classification"
                       else fold["model"].predict(Xte))
                rec = perturbed[["date", "ticker", "fwd_return"]].copy()
                rec["pred"] = pred
                ic = per_date_ic(rec, "pred", "fwd_return", method="rank")
                perturbed_ic = float(ic.mean()) if len(ic.dropna()) else float("nan")
                if np.isfinite(perturbed_ic):
                    drops.append(baseline_ic - perturbed_ic)
        row = {"family": feature_spec(members[0]).family if by_family else feature_spec(group_name).family,
              "baseline_ic": baseline_ic,
              "mean_ic_drop": float(np.mean(drops)) if drops else float("nan"),
              "n_folds": len(baseline_ic_by_fold)}
        if not by_family:
            row = {"feature": group_name, **row}
        rows.append(row)
    df = pd.DataFrame(rows)
    sort_col = "mean_ic_drop"
    return df.sort_values(sort_col, ascending=False).reset_index(drop=True)


def single_feature_ic_table(panel: pd.DataFrame, feature_names: Optional[List[str]] = None,
                            horizon_days: Optional[int] = None,
                            actual_col: str = "fwd_return") -> pd.DataFrame:
    """Standalone rank-IC (Newey-West-corrected) for every feature on its own,
    cross-checked against the feature registry's expected sign."""
    from .dataset import ML_FEATURES
    from .metrics import per_date_ic
    from .validation import ic_summary_nw

    feature_names = feature_names or ML_FEATURES
    rows = []
    for f in feature_names:
        ic = per_date_ic(panel, f, actual_col, date_col="date")
        nw = ic_summary_nw(ic, horizon_days=horizon_days)
        s = feature_spec(f)
        realised_sign = 0 if not np.isfinite(nw["mean_ic"]) or nw["mean_ic"] == 0 else int(
            np.sign(nw["mean_ic"]))
        rows.append({
            "feature": f, "family": s.family, "expected_sign": s.expected_sign,
            "realised_sign": realised_sign,
            "sign_matches_expected": (s.expected_sign == 0 or realised_sign == s.expected_sign),
            **nw,
        })
    return pd.DataFrame(rows).set_index("feature")


def global_explain(panel: pd.DataFrame, task: str,
                   feature_names: Optional[List[str]] = None,
                   bundle: Optional[DataBundle] = None,
                   horizon_days: Optional[int] = None,
                   n_splits: int = 5, min_train_periods: int = 12,
                   **hp) -> Dict[str, object]:
    """Bundle every global-explanation view for a model's feature set into one
    dict — the payload the model card and CLI report draw from."""
    from .dataset import ML_FEATURES
    from .validation import ic_breakdown_by_year

    feature_names = feature_names or ML_FEATURES
    out: Dict[str, object] = {
        "coefficient_stability": coefficient_stability(
            panel, task, feature_names, n_splits, min_train_periods, **hp),
        "permutation_importance_by_family": permutation_importance(
            panel, task, feature_names, n_splits, min_train_periods,
            by_family=True, **hp),
        "single_feature_ic": single_feature_ic_table(
            panel, feature_names, horizon_days=horizon_days),
    }
    from .train import walk_forward_evaluate
    wf = walk_forward_evaluate(panel, task=task, feature_names=feature_names,
                               n_splits=n_splits, min_train_periods=min_train_periods, **hp)
    if not wf["oos"].empty:
        out["ic_by_year"] = ic_breakdown_by_year(wf["oos"])
        if bundle is not None:
            from .validation import ic_breakdown_by_regime
            out["ic_by_regime"] = ic_breakdown_by_regime(wf["oos"], bundle)
    return out
