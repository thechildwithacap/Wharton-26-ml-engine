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
# "analyst" is deliberately excluded: EngineService never supplies real ratings
# to run_engine (there's no ratings source wired into the API), so it would
# always be signals/hybrid.py's "deterministic placeholder from fundamentals"
# (its own docstring's words) — a proxy re-derived from roic/accruals/gross
# margin, which are already shown honestly as value/quality. Presenting it
# alongside real factor scores would look like independent information it
# isn't. It stays a real ML feature internally (registered with
# is_proxy=True, flagged in ml_view.proxy_features / known_limitations) —
# only the misleading general-scorecard presentation is removed.
_FACTOR_COLS = [
    "value", "intrinsic", "quality", "growth", "garp", "discipline", "momentum",
    "low_vol", "size", "factor", "macro_tilt", "macro_fit",
]

_DATASET_CANDIDATES = ["datasets/us_wide", "datasets/us_sample_sec",
                       "datasets/us_sample"]


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
        ml_view = self._ml_view(sub, report.as_of)
        return report_to_dict(report, index_id, sub, ml_view=ml_view)

    def _ml_view(self, sub, as_of):
        """Build the ML explanation view for this request's universe, or None
        if there's no trained model (the UI degrades gracefully either way)."""
        if self.alpha_model is None:
            return None
        try:
            from ..ml import build_ml_view
            return build_ml_view(self.alpha_model, sub, as_of=as_of,
                                 model_id=os.path.basename(self.dataset_dir))
        except Exception:
            return None


def report_to_dict(report, index_id: str, bundle: DataBundle,
                   ml_view=None) -> Dict[str, object]:
    """Serialise an :class:`EngineReport` to a JSON-safe dict for the UI.

    ``ml_view`` (an :class:`~wharton_ml_engine.ml.view.MLView`, when a model is
    trained) is computed over the *full* universe so its percentile ranks stay
    honest, but only the recommended holdings' scores/reasons are attached to
    the payload — dumping all ~300 universe names' reasons on every request
    would bloat the response for no UI benefit.
    """
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
        if ml_view is not None:
            row["ml_score"] = _clean(ml_view.score.get(t))
            row["ml_reason"] = ml_view.reasons.get(t)
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
        "ml_view": _ml_view_summary(ml_view),
    }


def _ml_view_summary(ml_view) -> Optional[Dict[str, object]]:
    """Compact, universe-independent summary of the ML model's own evidence —
    the per-holding score/reason already live on each holdings row; this is
    the "how much should you trust this at all" block for the risk panel."""
    if ml_view is None:
        return None
    return {
        "model_id": ml_view.model_id,
        "confidence": _clean(ml_view.confidence),
        "gated_off": ml_view.confidence <= 0.05,
        "research_metrics": {k: _clean(v) for k, v in ml_view.card.research_metrics.items()},
        "proxy_features": list(ml_view.card.proxy_features),
        "known_limitations": list(ml_view.card.known_limitations),
        "holdout": dict(ml_view.card.holdout),
    }
