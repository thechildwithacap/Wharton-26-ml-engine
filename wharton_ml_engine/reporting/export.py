"""Reporting: CSV exports and a console summary (PRD sections 7.3 & 10).

Everything the Excel / Google Sheets front-end needs is written as flat CSVs
per decision date, plus a human-readable decision summary for the IPS / Final
Report.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import pandas as pd

from ..engine.pipeline import EngineReport


def export_report(report: EngineReport, outdir: str) -> Dict[str, str]:
    """Write the full report to ``outdir`` as CSV/JSON.  Returns paths written."""
    os.makedirs(outdir, exist_ok=True)
    stamp = report.as_of.strftime("%Y%m%d")
    written: Dict[str, str] = {}

    def _write(name: str, df: pd.DataFrame) -> None:
        path = os.path.join(outdir, f"{name}_{stamp}.csv")
        df.to_csv(path)
        written[name] = path

    _write("signals", report.signals.round(4))
    _write("scores", report.scored.round(4))
    _write("client_fit", report.client_fit.round(4))
    _write("stock_risk", report.stock_risk.round(4))
    _write("liquidity", report.liquidity.round(4))
    _write("portfolio", report.portfolio_table().round(6))
    _write("trades", report.decision.trades.round(6))
    if report.templates_summary is not None:
        _write("templates_summary", report.templates_summary.round(4))

    # Style weights and regime.
    sw = pd.DataFrame({"style_weight": report.style_recommendation.weights})
    _write("style_weights", sw.round(4))

    # Portfolio risk summary.
    risk_df = pd.DataFrame([report.candidate_risk.as_dict()]).T
    risk_df.columns = ["value"]
    _write("portfolio_risk", risk_df.round(6))

    # Decision summary as JSON.
    summary = decision_summary_dict(report)
    jpath = os.path.join(outdir, f"decision_{stamp}.json")
    with open(jpath, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    written["decision"] = jpath

    return written


def decision_summary_dict(report: EngineReport) -> Dict[str, object]:
    d = report.decision
    return {
        "as_of": str(report.as_of.date()),
        "client": report.profile.name,
        "objective": report.profile.objective,
        "risk_tolerance": report.profile.risk_label,
        "regime": report.regime.label,
        "regime_features": {k: round(v, 4) for k, v in report.regime.features.items()},
        "style_weights": {k: round(v, 3) for k, v in report.style_recommendation.weights.items()},
        "best_template": report.style_recommendation.best_template,
        "signal_confidence": round(report.signal_confidence, 3),
        "ml_alpha_metrics": ({k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in report.ml_metrics.items()}
                             if report.ml_metrics else None),
        "data_quality": report.data_quality.status,
        "n_holdings": len(report.candidate_weights),
        "portfolio_volatility": round(report.candidate_risk.volatility, 4),
        "portfolio_beta": round(report.candidate_risk.beta, 3),
        "top5_weight": round(report.candidate_risk.top5_weight, 3),
        "effective_n": round(report.candidate_risk.effective_n, 1),
        "concentration_alerts": report.concentration_alerts,
        "crowding_flags": report.crowding.get("crowding_flags", []),
        "mandate_passed": report.mandate.passed,
        "mandate_violations": report.mandate.violations,
        "turnover": round(report.turnover.turnover, 4),
        "cost_estimate": round(report.turnover.cost_estimate, 5),
        "decision": d.action,
        "approved": d.approved,
        "net_score_improvement": round(d.net_score_improvement, 3),
        "checks": d.checks,
        "reasons": d.reasons,
    }


def _fmt_pct(x: float) -> str:
    return f"{x:.1%}" if pd.notna(x) else "n/a"


def format_summary(report: EngineReport) -> str:
    """Return a multi-line console summary of the decision."""
    L: List[str] = []
    p = report.profile
    L.append("=" * 72)
    L.append(f" WHARTON ML INVESTMENT ENGINE  —  decision as of {report.as_of.date()}")
    L.append("=" * 72)
    L.append(f" Client: {p.name}  |  objective={p.objective}  |  "
             f"risk={p.risk_label}  |  horizon={p.horizon}")
    L.append(f" Market regime: {report.regime.label}  "
             f"(breadth {report.regime.features.get('breadth', float('nan')):.0%}, "
             f"6m {report.regime.features.get('ret_6m', float('nan')):+.1%})")
    L.append(f" Best backtested template: {report.style_recommendation.best_template}")
    sw = report.style_recommendation.weights
    L.append(" Recommended style weights: " +
             ", ".join(f"{k} {v:.0%}" for k, v in sorted(sw.items(), key=lambda kv: -kv[1])))
    L.append(f" Signal confidence: {report.signal_confidence:.2f}   "
             f"Data quality: {report.data_quality.status}")
    if report.ml_metrics:
        m = report.ml_metrics
        L.append(f" ML alpha model: out-of-sample IC {m.get('mean_ic', float('nan')):.3f} "
                 f"(t={m.get('ic_t_stat', float('nan')):.2f}, "
                 f"hit-rate {m.get('hit_rate', float('nan')):.0%})")
    L.append("-" * 72)
    L.append(f" Candidate portfolio: {len(report.candidate_weights)} holdings")
    L.append(f"   volatility {_fmt_pct(report.candidate_risk.volatility)}  |  "
             f"beta {report.candidate_risk.beta:.2f}  |  "
             f"top-5 {_fmt_pct(report.candidate_risk.top5_weight)}  |  "
             f"eff-N {report.candidate_risk.effective_n:.1f}")
    if report.concentration_alerts:
        L.append("   concentration alerts: " + "; ".join(report.concentration_alerts))
    if report.crowding.get("crowding_flags"):
        L.append("   crowding: " + "; ".join(report.crowding["crowding_flags"]))
    L.append(f"   mandate: {'PASS' if report.mandate.passed else 'FAIL'}"
             + ("" if report.mandate.passed else " — " + "; ".join(report.mandate.violations[:3])))
    L.append("-" * 72)
    pt = report.portfolio_table()
    L.append(" Top holdings:")
    for t, row in pt.head(10).iterrows():
        L.append(f"   {t:<5} {row['weight']:>6.2%}  {row.get('sector','')[:22]:<22} "
                 f"score {row.get('integrated_score', float('nan')):.1f}")
    L.append("-" * 72)
    L.append(f" DECISION: {report.decision.action.upper()}  "
             f"(net score improvement {report.decision.net_score_improvement:+.2f})")
    for r in report.decision.reasons:
        L.append(f"   • {r}")
    L.append("=" * 72)
    return "\n".join(L)


def print_summary(report: EngineReport) -> None:
    print(format_summary(report))
