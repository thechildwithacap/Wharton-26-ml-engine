"""API service layer: index registry, per-index engine runs, JSON serialisation.

Runs fully offline against a small synthetic bundle saved to a temp dir (no
network, no cached dataset), with the ML model disabled for speed.
"""

import json

import pytest

from wharton_ml_engine.api.service import EngineService, report_to_dict
from wharton_ml_engine.data import (
    SyntheticDataSource,
    available_universe,
    list_indices,
    save_bundle,
)


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    # Synthetic universe includes AAPL/MSFT/JPM... via the ticker generator? No —
    # synthetic tickers are SYN0.. so index membership won't match. We therefore
    # test the *registry mechanics* on real tickers separately, and the engine
    # run on the synthetic bundle via the SPX ("whole bundle") path.
    b = SyntheticDataSource(n_tickers=40, years=5, seed=3).load()
    d = str(tmp_path_factory.mktemp("ds"))
    save_bundle(b, d)
    return d


@pytest.fixture(scope="module")
def service(dataset_dir):
    return EngineService(dataset_dir=dataset_dir, train_model=False)


@pytest.fixture(scope="module")
def service_with_model(dataset_dir):
    return EngineService(dataset_dir=dataset_dir, train_model=True)


def test_index_registry_intersects_available_names():
    # With a universe that contains real index names, membership resolves.
    tickers = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "ZZZZ"]
    sectors = {t: "Technology" for t in tickers}
    ndx = available_universe("NDX", tickers, sectors)
    assert "AAPL" in ndx and "MSFT" in ndx and "NVDA" in ndx
    assert "ZZZZ" not in ndx          # not an index member
    spx = available_universe("SPX", tickers, sectors)
    assert spx == tickers             # SPX = whole bundle
    infos = {i.id: i for i in list_indices(tickers, sectors)}
    assert infos["NDX"].n_available >= 3


def test_service_indices_endpoint(service):
    out = service.indices()
    assert "as_of" in out and out["n_total_names"] == 40
    ids = {i["id"] for i in out["indices"]}
    assert "SPX" in ids               # whole-bundle index always available


def test_service_report_spx(service):
    r = service.report(index_id="SPX", objective="growth", risk_tolerance=4, mc_sims=1500)
    assert r["index"] == "SPX"
    assert r["n_holdings"] > 0
    assert r["decision"]["action"] in ("buy", "rebalance", "hold", "trim")
    assert len(r["holdings"]) == r["n_holdings"]
    # every holding carries weight + factor scorecard
    h0 = r["holdings"][0]
    assert 0 < h0["weight"] <= 1 and isinstance(h0["factors"], dict)
    assert r["monte_carlo"] is not None and "var_95" in r["monte_carlo"]


def test_report_is_json_serialisable(service):
    r = service.report(index_id="SPX", objective="balanced", risk_tolerance=3, mc_sims=800)
    s = json.dumps(r)               # must not raise (NaN/inf cleaned to null)
    assert '"NaN"' not in s and "Infinity" not in s


def test_unknown_index_raises(service):
    with pytest.raises((ValueError, KeyError)):
        service.report(index_id="NOPE")


def test_objective_changes_portfolio(service):
    growth = service.report(index_id="SPX", objective="growth", risk_tolerance=4, mc_sims=500)
    income = service.report(index_id="SPX", objective="income", risk_tolerance=2, mc_sims=500)
    # Different mandates should not produce identical books.
    assert {h["ticker"] for h in growth["holdings"]} != {h["ticker"] for h in income["holdings"]} \
        or growth["style_weights"] != income["style_weights"]


def test_no_model_means_no_ml_view(service):
    r = service.report(index_id="SPX", objective="balanced", risk_tolerance=3, mc_sims=500)
    assert r["ml_view"] is None
    assert "ml_score" not in r["holdings"][0]
    assert "ml_reason" not in r["holdings"][0]


def test_trained_model_produces_ml_view(service_with_model):
    r = service_with_model.report(index_id="SPX", objective="balanced", risk_tolerance=3,
                                  mc_sims=500)
    assert r["ml_view"] is not None
    mv = r["ml_view"]
    assert 0.0 <= mv["confidence"] <= 1.0
    assert mv["gated_off"] == (mv["confidence"] <= 0.05)
    assert "research_metrics" in mv and "known_limitations" in mv
    assert isinstance(mv["proxy_features"], list)
    # every holding carries its own ml_score/ml_reason
    for h in r["holdings"]:
        assert "ml_score" in h and "ml_reason" in h
        assert isinstance(h["ml_reason"], str) and len(h["ml_reason"]) > 0


def test_ml_view_is_json_serialisable(service_with_model):
    r = service_with_model.report(index_id="SPX", objective="growth", risk_tolerance=4,
                                  mc_sims=500)
    s = json.dumps(r)
    assert '"NaN"' not in s and "Infinity" not in s


def test_ml_view_reflects_gate_when_weak(service_with_model):
    r = service_with_model.report(index_id="SPX", objective="balanced", risk_tolerance=3,
                                  mc_sims=500)
    mv = r["ml_view"]
    if mv["gated_off"]:
        # A gated-off model must show every holding at the neutral score.
        assert all(h["ml_score"] == pytest.approx(50.0) for h in r["holdings"])
