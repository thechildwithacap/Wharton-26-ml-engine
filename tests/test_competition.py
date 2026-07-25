"""CompetitionRules adapter: rulebook -> engine config, and the readiness map."""

import pytest

from wharton_ml_engine import CompetitionRules, SyntheticDataSource, sample_profile
from wharton_ml_engine.engine import run_engine


def test_rules_map_onto_constraints():
    rules = CompetitionRules(
        name="Test Cup", max_holdings=25, min_holdings=12,
        max_position_pct=0.12, max_sector_pct=0.35, rebalance_frequency="weekly")
    c = rules.to_constraints()
    assert c.max_holdings == 25 and c.min_holdings == 12
    assert c.max_weight_per_stock == pytest.approx(0.12)
    assert c.max_weight_per_sector == pytest.approx(0.35)
    assert c.max_turnover_per_rebalance == pytest.approx(0.40)   # weekly cadence
    c.validate()                                                 # stays self-consistent


def test_scoring_metric_shapes_construction():
    tr = CompetitionRules(scoring_metric="total_return").to_constraints()
    dd = CompetitionRules(scoring_metric="drawdown_adjusted").to_constraints()
    vb = CompetitionRules(scoring_metric="vs_benchmark").to_constraints()
    assert tr.concentration > 1.0        # total-return -> concentrate
    assert dd.concentration < 1.0        # drawdown-adjusted -> spread out
    assert vb.benchmark_tilt > 0.0       # vs-benchmark -> partial cap-weight tilt


def test_horizon_from_contest_window():
    rules = CompetitionRules(start_date="2026-01-01", end_date="2026-04-30")
    h = rules.horizon_days()
    assert 60 < h < 100                  # ~4 months of trading days
    assert CompetitionRules().horizon_days() == 126   # default when no window


def test_engine_config_carries_rules():
    rules = CompetitionRules(max_position_pct=0.15, scoring_metric="sharpe",
                             start_date="2026-01-01", end_date="2026-06-30")
    cfg = rules.to_engine_config()
    assert cfg.constraints.max_weight_per_stock == pytest.approx(0.15)
    assert cfg.universe_filter is not None
    assert cfg.regime_aware_sizing is True          # risk-scored contest
    assert cfg.ml_horizon_days == rules.horizon_days()


def test_readiness_flags_unsupported_rules():
    # A rulebook that needs cash, shorting, a trade cap and a hold period must
    # surface all of those as needs_code (the honest gap list).
    rules = CompetitionRules(min_invested_pct=0.8, max_cash_pct=0.2,
                             allow_shorting=True, max_trades=50, min_holding_days=5,
                             allowed_asset_classes=("equity", "etf", "crypto"))
    r = rules.readiness()
    needs = {x["lever"] for x in r["needs_code"]}
    assert any("cash" in n for n in needs)
    assert any("short" in n for n in needs)
    assert any("max-trades" in n for n in needs)
    assert any("holding period" in n for n in needs)
    assert any("non-equity" in n for n in needs)
    assert len(r["enforced"]) >= 8       # the bulk is already config-driven


def test_default_rules_have_no_blocking_gaps():
    # A plain long-only, fully-invested equity contest needs no new code except
    # the (always-listed) objective-alignment note.
    r = CompetitionRules().readiness()
    needs = {x["lever"] for x in r["needs_code"]}
    assert not any("cash" in n or "short" in n or "non-equity" in n for n in needs)


def test_rules_drive_a_real_engine_run():
    bundle = SyntheticDataSource(n_tickers=50, years=5, seed=6).load()
    cfg = CompetitionRules(max_holdings=20, max_position_pct=0.10,
                           scoring_metric="total_return").to_engine_config()
    report = run_engine(bundle, sample_profile(), config=cfg, run_backtest=False,
                        run_monte_carlo=False)
    w = report.candidate_weights
    assert len(w) <= 20
    assert max(w.values()) <= 0.10 + 1e-6           # honours the contest position cap
