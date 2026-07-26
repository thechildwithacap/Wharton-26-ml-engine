"""Engine service: load data once, run the engine per index, serialise to JSON.

Kept framework-free so it works with the stdlib server (:mod:`.server`) or any
other host.  The bundle and the trained alpha model are loaded once and cached;
each request restricts the universe to the chosen index and runs the full
pipeline (signals -> construction -> risk -> Monte Carlo).
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..client.profile import ClientProfile
from ..config import EngineConfig
from ..data import available_universe, list_indices, load_bundle
from ..data.source import DataBundle
from ..engine import run_engine
from ..risk.portfolio import sector_exposure

# Factor columns to surface for the per-holding heatmap (order = display order).
_FACTOR_COLS = [
    "value", "intrinsic", "quality", "growth", "garp", "discipline", "momentum",
    "low_vol", "size", "factor", "macro_tilt", "macro_fit", "analyst",
]

_DATASET_CANDIDATES = ["datasets/us_sample_sec", "datasets/us_sample"]


def _clean(x):
    """JSON-safe: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(x, (np.floating, float)):
        return None if (x is None or math.isnan(x) or math.isinf(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


class EngineService:
    def __init__(self, dataset_dir: Optional[str] = None, train_model: bool = True,
                 config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.dataset_dir = dataset_dir or self._find_dataset()
        self.bundle: DataBundle = load_bundle(self.dataset_dir)
        self.alpha_model = None
        if train_model:
            self._train()

    @staticmethod
    def _find_dataset() -> str:
        for d in _DATASET_CANDIDATES:
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "prices.csv")):
                return d
        raise FileNotFoundError(
            "no dataset found; expected one of " + ", ".join(_DATASET_CANDIDATES))

    def _train(self) -> None:
        try:
            from ..ml import AlphaModel, train_alpha_model
            trained = train_alpha_model(
                self.bundle, task="regression",
                horizon_days=self.config.ml_horizon_days, n_splits=6)
            self.alpha_model = AlphaModel(trained)
        except Exception:
            self.alpha_model = None      # UI still works without the ML overlay

    # ---- endpoints ----------------------------------------------------------
    def indices(self) -> Dict[str, object]:
        infos = list_indices(self.bundle.tickers, self.bundle.sectors)
        return {
            "as_of": str(self.bundle.dates()[-1].date()),
            "dataset": os.path.basename(self.dataset_dir),
            "n_total_names": len(self.bundle.tickers),
            "indices": [
                {"id": i.id, "name": i.name, "n_members": i.n_members,
                 "n_available": i.n_available, "benchmark": i.benchmark}
                for i in infos if i.n_available > 0
            ],
        }

    def report(self, index_id: str = "SPX", objective: str = "balanced",
               risk_tolerance: int = 3, mc_sims: int = 8000,
               mc_annual_drift: Optional[float] = None) -> Dict[str, object]:
        universe = available_universe(index_id, self.bundle.tickers, self.bundle.sectors)
        if not universe:
            raise ValueError(f"no available constituents for index {index_id!r}")
        sub = self.bundle.restrict_universe(universe)
        profile = ClientProfile(objective=objective,
                                risk_tolerance=int(risk_tolerance))
        report = run_engine(
            sub, profile, config=self.config, alpha_model=self.alpha_model,
            run_backtest=False, mc_sims=mc_sims, mc_annual_drift=mc_annual_drift)
        return report_to_dict(report, index_id, sub)


def report_to_dict(report, index_id: str, bundle: DataBundle) -> Dict[str, object]:
    """Serialise an :class:`EngineReport` to a JSON-safe dict for the UI."""
    weights = report.candidate_weights
    scored = report.scored
    signals = report.signals
    names = {t: bundle.meta[t].name for t in bundle.tickers}
    sectors = bundle.sectors

    holdings: List[Dict[str, object]] = []
    for t, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        row = {
            "ticker": t, "name": names.get(t, t), "sector": sectors.get(t, ""),
            "weight": _clean(w),
            "integrated_score": _clean(scored["integrated_score"].get(t)
                                       if "integrated_score" in scored else None),
            "client_fit": _clean(scored["client_fit"].get(t)
                                 if "client_fit" in scored else None),
            "risk_score": _clean(scored["risk_score"].get(t)
                                 if "risk_score" in scored else None),
            "factors": {c: _clean(signals[c].get(t)) for c in _FACTOR_COLS
                        if c in signals.columns},
        }
        holdings.append(row)

    reg = report.regime
    mc = report.monte_carlo.to_json() if report.monte_carlo else None
    sec_exp = sector_exposure(weights, sectors)

    return {
        "index": index_id,
        "as_of": str(report.as_of.date()),
        "client": {"objective": report.profile.objective,
                   "risk": report.profile.risk_label},
        "regime": {"label": reg.label,
                   "features": {k: _clean(v) for k, v in reg.features.items()}},
        "decision": {
            "action": report.decision.action,
            "approved": bool(report.decision.approved),
            "net_score_improvement": _clean(report.decision.net_score_improvement),
            "reasons": list(report.decision.reasons),
        },
        "style_weights": {k: _clean(v) for k, v in report.style_recommendation.weights.items()},
        "signal_confidence": _clean(report.signal_confidence),
        "ml_metrics": ({k: _clean(v) for k, v in report.ml_metrics.items()}
                       if report.ml_metrics else None),
        "risk": {
            "volatility": _clean(report.candidate_risk.volatility),
            "beta": _clean(report.candidate_risk.beta),
            "top5_weight": _clean(report.candidate_risk.top5_weight),
            "effective_n": _clean(report.candidate_risk.effective_n),
            "sector_exposure": {k: _clean(v) for k, v in sec_exp.items()},
        },
        "concentration_alerts": list(report.concentration_alerts),
        "n_holdings": len(weights),
        "n_universe": len(bundle.tickers),
        "holdings": holdings,
        "monte_carlo": mc,
        "factor_columns": [c for c in _FACTOR_COLS if c in signals.columns],
    }
