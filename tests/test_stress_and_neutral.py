"""Tests for the stress-test module and sector-neutralization."""

import numpy as np
import pandas as pd
import pytest

from wharton_ml_engine import EngineConfig, SyntheticDataSource, run_engine, sample_profile
from wharton_ml_engine.risk import stress_test
from wharton_ml_engine.utils import sector_neutralize


@pytest.fixture(scope="module")
def bundle():
    return SyntheticDataSource(n_tickers=50, years=8, seed=6).load()


# --- stress module (Task 2) ---------------------------------------------

@pytest.fixture(scope="module")
def report(bundle):
    cfg = EngineConfig()
    rep = run_engine(bundle, sample_profile(), config=cfg, run_backtest=False)
    return stress_test(rep.candidate_weights, bundle, cfg.constraints), rep


def test_stress_report_structure(report):
    sr, _ = report
    d = sr.as_dict()
    assert d["n_holdings"] > 0
    assert "market_-20pct" in d["scenarios"]
    assert "rate_+100bps" in d["scenarios"]
    assert set(d["correlation"]) >= {"calm", "crisis"}
    assert "nav_impact" in d["single_name"]


def test_single_name_worst_case_equals_top_weight(report):
    sr, rep = report
    top = max(rep.candidate_weights.values())
    assert abs(sr.single_name["nav_impact"] + top) < 1e-3      # -top weight (rounded 4dp)
    # bounded by the position cap
    assert top <= EngineConfig().constraints.max_weight_per_stock + 1e-6


def test_market_scenario_uses_beta(report):
    sr, _ = report
    sc = sr.scenarios["market_-20pct"]
    assert abs(sc["portfolio_impact"] - sr.portfolio_beta * -0.20) < 1e-3


def test_stress_export_json(report, tmp_path):
    sr, _ = report
    p = sr.to_json(str(tmp_path / "stress.json"))
    import json
    d = json.load(open(p))
    assert d["scenarios"]["rate_+100bps"]["portfolio_impact"] <= 0


# --- sector-neutralization (Task 3) -------------------------------------

def test_sector_neutralize_removes_sector_level():
    # Sector A all high, sector B all low; after neutralization the cross-sector
    # level is gone (both sectors span the full rank range).
    idx = ["a1", "a2", "b1", "b2"]
    df = pd.DataFrame({"value": [90.0, 70.0, 30.0, 10.0]}, index=idx)
    sectors = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
    out = sector_neutralize(df, sectors, ["value"])
    # within A, a1>a2; within B, b1>b2 — ordering preserved inside sector
    assert out.loc["a1", "value"] > out.loc["a2", "value"]
    assert out.loc["b1", "value"] > out.loc["b2", "value"]
    # the top name in each sector should now rank similarly (sector level removed)
    assert abs(out.loc["a1", "value"] - out.loc["b1", "value"]) < 1e-9


def test_engine_runs_sector_neutral(bundle):
    profile = sample_profile()
    base = run_engine(bundle, profile, config=EngineConfig(sector_neutral=False),
                      run_backtest=False)
    neu = run_engine(bundle, profile, config=EngineConfig(sector_neutral=True),
                     run_backtest=False)
    assert neu.mandate.passed
    # neutralization should change the book (different scores -> different picks/weights)
    assert base.candidate_weights != neu.candidate_weights
