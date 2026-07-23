"""Tests for the paper-trading portfolio and stepping logic."""

import numpy as np
import pandas as pd
import pytest

from wharton_ml_engine import SyntheticDataSource, sample_profile
from wharton_ml_engine.engine import (
    PaperPortfolio,
    paper_trade_step,
    performance_summary,
)


@pytest.fixture(scope="module")
def bundle():
    return SyntheticDataSource(n_tickers=40, years=6, seed=9).load()


def _prices(bundle, as_of=None):
    idx = bundle.prices.index[-1] if as_of is None else as_of
    return bundle.prices.loc[bundle.prices.index <= idx].iloc[-1]


# --- portfolio mechanics -------------------------------------------------

def test_new_portfolio_all_cash(bundle):
    pf = PaperPortfolio.new(capital=100_000)
    px = _prices(bundle)
    assert pf.nav(px) == 100_000
    assert pf.weights(px) == {}


def test_rebalance_hits_targets_and_charges_cost(bundle):
    pf = PaperPortfolio.new(capital=100_000)
    px = _prices(bundle)
    tickers = list(bundle.prices.columns[:4])
    target = {t: 0.25 for t in tickers}
    trade = pf.rebalance_to(target, px, cost_fraction=0.001)
    w = pf.weights(px)
    # weights land near target (fractional shares -> near-exact)
    for t in tickers:
        assert abs(w[t] - 0.25) < 0.02
    assert trade["n_trades"] == 4
    assert trade["cost"] > 0
    # NAV drops only by the cost charged
    assert pf.nav(px) < 100_000
    assert pf.nav(px) > 100_000 * 0.99


def test_rebalance_then_reduce_turnover(bundle):
    pf = PaperPortfolio.new(capital=100_000)
    px = _prices(bundle)
    t = list(bundle.prices.columns[:5])
    pf.rebalance_to({x: 0.2 for x in t}, px, 0.0)
    # rebalancing to the same weights is ~zero turnover
    trade = pf.rebalance_to({x: 0.2 for x in t}, px, 0.001)
    assert trade["turnover"] < 0.01


def test_mark_records_history(bundle):
    pf = PaperPortfolio.new(capital=50_000)
    px = _prices(bundle)
    rec = pf.mark(bundle.prices.index[-1], px, benchmark=4000.0, note="init")
    assert len(pf.history) == 1
    assert rec["nav"] == 50_000
    assert rec["benchmark"] == 4000.0


def test_save_load_roundtrip(bundle, tmp_path):
    pf = PaperPortfolio.new(capital=100_000)
    px = _prices(bundle)
    pf.rebalance_to({t: 0.2 for t in bundle.prices.columns[:5]}, px, 0.001)
    pf.mark(bundle.prices.index[-1], px, benchmark=4000.0)
    p = tmp_path / "state.json"
    pf.save(str(p))
    back = PaperPortfolio.load(str(p))
    assert back.cash == pf.cash
    assert back.positions == pf.positions
    assert back.history == pf.history


# --- stepping via the engine --------------------------------------------

def test_paper_step_marks_and_acts(bundle):
    pf = PaperPortfolio.new(capital=100_000)
    r = paper_trade_step(bundle, sample_profile(), pf, force_rebalance=True)
    assert r.action == "rebalance"
    assert len(pf.positions) >= 15               # built a diversified book
    assert len(pf.history) == 1
    assert r.nav > 0


def test_performance_summary_after_walk(bundle):
    pf = PaperPortfolio.new(capital=100_000)
    profile = sample_profile()
    dates = bundle.dates()
    for d in [dates[300], dates[360], dates[420], dates[-1]]:
        paper_trade_step(bundle, profile, pf, as_of=d)
    s = performance_summary(pf)
    assert s["marks"] == 4
    assert "total_return" in s
    assert np.isfinite(s["total_return"])
    assert "benchmark_return" in s
