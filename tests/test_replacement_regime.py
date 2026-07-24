"""Tests for the replacement classifier (Task B) and regime-aware sizing (Task C)."""

import pandas as pd
import pytest

from wharton_ml_engine import EngineConfig, SyntheticDataSource, run_engine, sample_profile
from wharton_ml_engine.engine import (
    classify_holdings,
    regime_sizing_posture,
    replacement_candidates,
)
from wharton_ml_engine.engine.replacement import classify_holding


def _signals(rows):
    return pd.DataFrame(rows).T


# --- Task B: replacement classification ---------------------------------

def test_thesis_broken_when_component_breaks():
    sig = _signals({"X": {"value": 80, "quality": 15, "momentum": 60}})  # quality broke
    c = classify_holding("X", sig, integrated_score=40, prior_score=70, break_threshold=25)
    assert c.status == "thesis_broken"
    assert c.weakest_component == "quality"


def test_out_of_favor_when_declined_but_components_intact():
    sig = _signals({"X": {"value": 55, "quality": 60, "momentum": 45}})
    c = classify_holding("X", sig, integrated_score=48, prior_score=68, break_threshold=25)
    assert c.status == "out_of_favor"


def test_healthy_when_no_decline():
    sig = _signals({"X": {"value": 60, "quality": 65, "momentum": 55}})
    c = classify_holding("X", sig, integrated_score=64, prior_score=64, break_threshold=25)
    assert c.status == "healthy"


def test_replacement_candidates_only_broken():
    sig = _signals({
        "A": {"value": 80, "quality": 10, "momentum": 60},   # broken
        "B": {"value": 55, "quality": 60, "momentum": 50},   # ok
    })
    scored = pd.DataFrame({"integrated_score": {"A": 40, "B": 50}})
    prior = pd.DataFrame({"integrated_score": {"A": 70, "B": 62}})
    cls = classify_holdings(["A", "B"], sig, scored, prior, break_threshold=25)
    assert replacement_candidates(cls) == ["A"]


# --- Task C: regime-aware sizing ----------------------------------------

def test_posture_compounder_vs_contrarian():
    comp = regime_sizing_posture("bull", "low_vol", 1.0, 0.08)
    contra = regime_sizing_posture("bear", "low_vol", 1.0, 0.08)
    highvol = regime_sizing_posture("sideways", "high_vol", 1.0, 0.08)
    balanced = regime_sizing_posture("sideways", "low_vol", 1.0, 0.08)
    assert comp["posture"] == "compounder" and comp["concentration"] >= 1.3
    assert contra["posture"] == "contrarian" and contra["concentration"] <= 0.7
    assert contra["max_weight"] <= 0.06                 # tighter cap for contrarian
    assert highvol["posture"] == "contrarian"
    assert balanced["posture"] == "balanced" and balanced["concentration"] == 1.0


def test_engine_runs_regime_aware_sizing():
    b = SyntheticDataSource(n_tickers=45, years=6, seed=8).load()
    report = run_engine(b, sample_profile(),
                        config=EngineConfig(regime_aware_sizing=True), run_backtest=False)
    assert report.mandate.passed
    assert len(report.candidate_weights) > 0
