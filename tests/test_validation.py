"""Validation protocol: Newey-West t-stats, multiple-testing correction, the
locked holdout, the experiment registry, and IC breakdowns."""

import math
import os

import numpy as np
import pandas as pd
import pytest

from wharton_ml_engine.ml.validation import (
    DEFAULT_HOLDOUT_FRACTION,
    HARVEY_LIU_ZHU_T_BAR,
    adjusted_significance,
    count_trials,
    final_check,
    holdout_only_panel,
    ic_breakdown_by_group,
    ic_breakdown_by_regime,
    ic_breakdown_by_year,
    ic_summary_nw,
    log_experiment,
    newey_west_t,
    nw_lags_for_horizon,
    read_registry,
    research_only_panel,
    split_research_holdout,
    two_sided_p,
)


# ------------------------------------------------------------- Newey-West
def test_nw_lags_for_horizon():
    assert nw_lags_for_horizon(21, rebalance_days=21) == 0    # no overlap
    assert nw_lags_for_horizon(126, rebalance_days=21) == 5   # ceil(126/21)-1
    assert nw_lags_for_horizon(252, rebalance_days=21) == 11
    assert nw_lags_for_horizon(0, rebalance_days=21) == 0


def test_nw_reduces_to_naive_at_zero_lags():
    rng = np.random.default_rng(0)
    x = rng.normal(0.02, 0.1, 60)
    nw = newey_west_t(x, lags=0)
    # NW uses the population-style 1/n normalisation (standard HAC convention),
    # not the ddof=1 sample variance — the two differ by sqrt(n/(n-1)).
    naive_se = x.std(ddof=0) / math.sqrt(len(x))
    assert nw["nw_se"] == pytest.approx(naive_se, rel=1e-6)
    naive_t = x.mean() / naive_se
    assert nw["nw_t"] == pytest.approx(naive_t, rel=1e-6)


def test_nw_se_grows_with_positive_autocorrelation():
    # An AR(1)-like positively autocorrelated series should get a LARGER HAC
    # standard error (smaller |t|) than i.i.d. noise with the same variance —
    # exactly the "overlapping labels inflate naive t" problem this exists to fix.
    rng = np.random.default_rng(1)
    iid = rng.normal(0.01, 0.05, 200)
    ar = np.zeros(200)
    ar[0] = rng.normal(0.01, 0.05)
    for i in range(1, 200):
        ar[i] = 0.01 + 0.85 * (ar[i - 1] - 0.01) + rng.normal(0, 0.02)
    nw_iid = newey_west_t(iid, lags=5)
    nw_ar = newey_west_t(ar, lags=5)
    assert nw_ar["nw_se"] > nw_iid["nw_se"]


def test_nw_handles_degenerate_input():
    assert math.isnan(newey_west_t([], lags=2)["nw_t"])
    assert math.isnan(newey_west_t([0.01], lags=2)["nw_t"])
    zero_var = newey_west_t([0.02, 0.02, 0.02], lags=0)
    assert math.isnan(zero_var["nw_t"])              # zero variance -> undefined t


def test_ic_summary_nw_reports_effective_periods():
    ic = pd.Series(np.random.default_rng(2).normal(0.02, 0.08, 60))
    out = ic_summary_nw(ic, horizon_days=126, rebalance_days=21)
    assert out["nw_lags"] == 5
    assert out["n_periods"] == 60
    assert out["n_effective_periods"] == pytest.approx(60 / 6)


# ------------------------------------------------- multiple-testing correction
def test_two_sided_p_matches_known_values():
    # t=1.96 -> p ~ 0.05 (two-sided, large-sample normal approximation)
    assert two_sided_p(1.96) == pytest.approx(0.05, abs=0.002)
    assert two_sided_p(0.0) == pytest.approx(1.0, abs=1e-9)


def test_sidak_correction_widens_with_more_trials():
    one = adjusted_significance(2.5, n_trials=1)
    many = adjusted_significance(2.5, n_trials=50)
    assert one["p_adj"] < many["p_adj"]               # more trials -> less credible
    assert abs(one["t_adj"]) > abs(many["t_adj"])
    assert one["t_adj"] == pytest.approx(2.5, abs=1e-6)  # 1 trial -> no correction


def test_sidak_t_adj_recovers_original_p():
    adj = adjusted_significance(3.1, n_trials=10)
    assert 0 < adj["p_adj"] <= 1
    assert np.sign(adj["t_adj"]) == np.sign(3.1)


def test_harvey_liu_zhu_bar_is_a_reference_not_a_gate():
    assert HARVEY_LIU_ZHU_T_BAR == 3.0


# --------------------------------------------------------------- locked holdout
def test_split_by_fraction():
    dates = pd.bdate_range("2020-01-01", periods=100, freq="7D")
    research, holdout = split_research_holdout(dates, holdout_fraction=0.2)
    assert len(holdout) == 20
    assert research.max() < holdout.min()
    assert len(research) + len(holdout) == len(dates)


def test_split_by_explicit_date():
    dates = pd.bdate_range("2020-01-01", periods=50, freq="7D")
    cutoff = dates[30]
    research, holdout = split_research_holdout(dates, holdout_start=cutoff)
    assert (research < cutoff).all()
    assert (holdout >= cutoff).all()


def test_research_only_and_holdout_only_panels_partition_the_data():
    dates = pd.bdate_range("2020-01-01", periods=40, freq="7D")
    panel = pd.DataFrame({"date": list(dates) * 2, "x": range(80)})
    research = research_only_panel(panel, holdout_fraction=0.25)
    holdout = holdout_only_panel(panel, holdout_fraction=0.25)
    assert len(research) + len(holdout) == len(panel)
    assert set(research["date"]).isdisjoint(set(holdout["date"]))


def test_walk_forward_evaluate_research_only_drops_holdout(tmp_path):
    from wharton_ml_engine.data import SyntheticDataSource
    from wharton_ml_engine.ml.dataset import build_training_panel
    from wharton_ml_engine.ml.train import walk_forward_evaluate

    b = SyntheticDataSource(n_tickers=25, years=6, seed=9).load()
    panel = build_training_panel(b, horizon_days=21)
    full = walk_forward_evaluate(panel, task="regression", research_only=False)
    research = walk_forward_evaluate(panel, task="regression", research_only=True,
                                     holdout_fraction=0.2)
    if not full["oos"].empty and not research["oos"].empty:
        assert research["oos"]["date"].max() <= full["oos"]["date"].max()
        assert research["summary"]["n_periods"] <= full["summary"]["n_periods"]


# --------------------------------------------------------------- experiment registry
def test_log_and_count_experiments(tmp_path):
    reg = str(tmp_path / "registry.jsonl")
    log_experiment({"model": "ridge", "horizon": 63}, {"mean_ic": 0.02}, "large_63d", reg)
    log_experiment({"model": "ridge", "horizon": 63, "alpha": 5}, {"mean_ic": 0.03},
                   "large_63d", reg)
    log_experiment({"model": "ridge", "horizon": 126}, {"mean_ic": 0.01}, "large_126d", reg)
    assert count_trials("large_63d", reg) == 2
    assert count_trials("large_126d", reg) == 1
    assert count_trials("nonexistent_family", reg) == 0
    records = read_registry(reg)
    assert len(records) == 3
    assert all("timestamp" in r and "id" in r for r in records)


def test_registry_never_skips_a_failed_variant(tmp_path):
    reg = str(tmp_path / "registry.jsonl")
    log_experiment({"model": "bad_idea"}, {"mean_ic": -0.05, "nw_t": -1.2}, "fam", reg)
    assert count_trials("fam", reg) == 1
    records = read_registry(reg)
    assert records[0]["metrics"]["mean_ic"] == -0.05


# --------------------------------------------------------------- final_check
def _toy_panel():
    dates = pd.bdate_range("2020-01-01", periods=30, freq="14D")
    rows = []
    rng = np.random.default_rng(42)
    for d in dates:
        for t in ["A", "B", "C", "D", "E", "F"]:
            x = rng.normal()
            rows.append({"date": d, "ticker": t, "feat": x,
                        "fwd_return": 0.3 * x + rng.normal(0, 0.5)})
    return pd.DataFrame(rows)


def test_final_check_scores_holdout_once_and_logs(tmp_path):
    panel = _toy_panel()
    log_path = str(tmp_path / "holdout_log.jsonl")

    def fit_fn(research_panel):
        # trivial "model": store the OLS slope of feat -> fwd_return
        x = research_panel["feat"].to_numpy()
        y = research_panel["fwd_return"].to_numpy()
        slope = float(np.dot(x, y) / np.dot(x, x))
        return slope

    def predict_fn(model_slope, holdout_panel):
        return holdout_panel["feat"] * model_slope

    result = final_check({"model": "toy_ols"}, panel, fit_fn, predict_fn,
                         family="toy_fam", holdout_fraction=0.3,
                         holdout_log_path=log_path)
    assert result.already_checked is False
    assert result.n_holdout_dates > 0
    assert os.path.exists(log_path)

    # A second call for the SAME spec must not re-score — it returns the prior result.
    again = final_check({"model": "toy_ols"}, panel, fit_fn, predict_fn,
                        family="toy_fam", holdout_fraction=0.3,
                        holdout_log_path=log_path)
    assert again.already_checked is True
    assert again.spec_id == result.spec_id
    with open(log_path) as fh:
        assert len(fh.readlines()) == 1        # still just one log line


def test_final_check_force_recheck_logs_a_second_loud_entry(tmp_path, capsys):
    panel = _toy_panel()
    log_path = str(tmp_path / "holdout_log.jsonl")

    def fit_fn(research_panel):
        return 0.1

    def predict_fn(model, holdout_panel):
        return holdout_panel["feat"] * model

    final_check({"model": "toy"}, panel, fit_fn, predict_fn, family="fam2",
               holdout_fraction=0.3, holdout_log_path=log_path)
    final_check({"model": "toy"}, panel, fit_fn, predict_fn, family="fam2",
               holdout_fraction=0.3, holdout_log_path=log_path, force_recheck=True)
    with open(log_path) as fh:
        assert len(fh.readlines()) == 2
    assert "WARNING" in capsys.readouterr().out


# --------------------------------------------------------------- IC breakdowns
def _oos_frame():
    dates = pd.bdate_range("2019-01-01", periods=24, freq="MS")
    rows = []
    rng = np.random.default_rng(7)
    for d in dates:
        for t in ["AAA", "BBB", "CCC", "DDD"]:
            pred = rng.normal()
            rows.append({"date": d, "ticker": t, "pred": pred,
                        "fwd_return": 0.2 * pred + rng.normal(0, 0.3)})
    return pd.DataFrame(rows)


def test_ic_breakdown_by_year():
    out = ic_breakdown_by_year(_oos_frame())
    assert set(out.index) == {2019, 2020}
    assert "nw_t" in out.columns and "mean_ic" in out.columns


def test_ic_breakdown_by_group():
    oos = _oos_frame()
    group_map = {"AAA": "Tech", "BBB": "Tech", "CCC": "Financials", "DDD": "Financials"}
    out = ic_breakdown_by_group(oos, group_map)
    assert set(out.index) == {"Tech", "Financials"}


def test_ic_breakdown_by_regime_runs_against_a_real_bundle():
    from wharton_ml_engine.data import SyntheticDataSource
    b = SyntheticDataSource(n_tickers=10, years=3, seed=4).load()
    dates = b.dates()[::20][:15]
    rng = np.random.default_rng(3)
    rows = []
    for d in dates:
        for t in b.tickers[:5]:
            pred = rng.normal()
            rows.append({"date": d, "ticker": t, "pred": pred,
                        "fwd_return": rng.normal(0, 0.1)})
    out = ic_breakdown_by_regime(pd.DataFrame(rows), b)
    assert not out.empty
    assert set(out.index).issubset({"risk-on", "risk-off", "neutral", "unknown"})
