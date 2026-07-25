import pandas as pd

from wharton_ml_engine import EngineConfig, run_engine
from wharton_ml_engine.backtest import backtest_all_templates, classify_regime
from wharton_ml_engine.engine import run_engine as run_engine2
from wharton_ml_engine.risk import mandate_check


def _sector_weights(weights, bundle):
    out = {}
    for t, w in weights.items():
        s = bundle.meta[t].sector
        out[s] = out.get(s, 0.0) + w
    return out


def test_regime_label_valid(bundle, as_of):
    reg = classify_regime(bundle, as_of)
    assert reg.trend in {"bull", "bear", "sideways"}
    assert reg.vol_state in {"low_vol", "high_vol"}


def test_backtest_templates_produce_metrics(bundle):
    dates = bundle.dates()
    res = backtest_all_templates(bundle, dates[-500], dates[-1], top_n=15)
    assert set(res) == {"deep_value", "quality_compounder", "garp_blend",
                        "momentum_tilt", "defensive_low_vol", "hybrid_multifactor"}
    for r in res.values():
        assert "sharpe" in r.metrics
        assert len(r.returns) > 50


def test_engine_respects_constraints(bundle, profile, as_of):
    report = run_engine(bundle, profile, as_of=as_of, run_backtest=False)
    w = report.candidate_weights
    cons = profile.apply_to_constraints(report.config.constraints)

    assert len(w) <= cons.max_holdings
    assert abs(sum(w.values()) - 1.0) < 1e-6
    # Position caps.
    assert max(w.values()) <= cons.max_weight_per_stock + 1e-6
    # Sector caps.
    for sec, exp in _sector_weights(w, bundle).items():
        assert exp <= cons.max_weight_per_sector + 1e-6
    # No excluded sector present.
    for t in w:
        assert bundle.meta[t].sector not in profile.sector_avoidances


def test_engine_mandate_passes(bundle, profile, as_of):
    report = run_engine(bundle, profile, as_of=as_of, run_backtest=False)
    assert report.mandate.passed, report.mandate.violations


def test_engine_attaches_monte_carlo(bundle, profile, as_of):
    from wharton_ml_engine.reporting.export import decision_summary_dict, format_summary
    report = run_engine(bundle, profile, as_of=as_of, run_backtest=False, mc_sims=2000)
    mc = report.monte_carlo
    assert mc is not None
    assert mc.n_sims == 2000 and mc.horizon_days == report.config.ml_horizon_days
    assert 0.0 <= mc.prob_loss <= 1.0
    assert mc.percentiles["p05"] < mc.percentiles["p95"]
    # exported into both the JSON summary and the console text
    d = decision_summary_dict(report)
    assert d["monte_carlo"] is not None and "var_95" in d["monte_carlo"]
    assert "Monte Carlo" in format_summary(report)


def test_monte_carlo_can_be_disabled(bundle, profile, as_of):
    report = run_engine(bundle, profile, as_of=as_of, run_backtest=False,
                        run_monte_carlo=False)
    assert report.monte_carlo is None


def test_initial_build_rebalances(bundle, profile, as_of):
    report = run_engine(bundle, profile, as_of=as_of, run_backtest=False)
    assert report.decision.action == "rebalance"
    assert report.decision.approved


def test_no_change_holds(bundle, profile, as_of):
    # Feeding the freshly-built book back in should produce ~zero improvement
    # and therefore a HOLD.
    report1 = run_engine(bundle, profile, as_of=as_of, run_backtest=False)
    report2 = run_engine(bundle, profile, as_of=as_of, run_backtest=False,
                         current_weights=report1.candidate_weights)
    assert report2.decision.action == "hold"
    assert abs(report2.decision.net_score_improvement) < 1.0


def test_mandate_check_catches_violation(bundle, profile):
    # Build an obviously non-compliant book: one 50% position.
    t = bundle.tickers[0]
    bad = {t: 0.5, bundle.tickers[1]: 0.5}
    cons = profile.apply_to_constraints(EngineConfig().constraints)
    chk = mandate_check(bad, bundle, profile, cons)
    assert not chk.passed
    assert any("weight" in v or "holdings" in v for v in chk.violations)


def test_hard_excluded_not_in_scored_top(bundle, profile, as_of):
    report = run_engine(bundle, profile, as_of=as_of, run_backtest=False)
    # Ineligible names must have NaN integrated score.
    excl = report.client_fit[report.client_fit["hard_excluded"]].index
    for t in excl:
        assert pd.isna(report.scored.loc[t, "integrated_score"])
