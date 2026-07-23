"""Stress-testing module (PRD §9 / audit deliverable).

Turns the ad-hoc crisis numbers into a first-class, **exportable** report:
historical crisis drawdowns (holding the current book through 2020 / 2022),
model-based hypothetical shocks (market and rate, applied via estimated factor
sensitivities since pre-2019 data doesn't exist), a crisis-vs-calm correlation
panel, and single-name worst-case exposure under the concentration caps.

Everything returns chart-ready data (``StressReport.as_dict`` / ``to_json``) so
it drops straight into the pitch deck rather than living in a console log.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import PortfolioConstraints
from ..data.source import DataBundle
from ..utils import max_drawdown


# Named historical windows (start, end).  Add more as data allows.
DEFAULT_WINDOWS: Dict[str, Tuple[str, str]] = {
    "covid_2020": ("2020-02-15", "2020-04-30"),
    "selloff_2022": ("2022-01-01", "2022-10-31"),
}
CALM_WINDOW: Tuple[str, str] = ("2021-01-01", "2021-12-31")


@dataclass
class StressReport:
    as_of: str
    n_holdings: int
    portfolio_beta: float
    historical: Dict[str, Dict[str, float]] = field(default_factory=dict)
    scenarios: Dict[str, Dict[str, object]] = field(default_factory=dict)
    correlation: Dict[str, float] = field(default_factory=dict)
    single_name: Dict[str, object] = field(default_factory=dict)
    coverage_note: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "as_of": self.as_of, "n_holdings": self.n_holdings,
            "portfolio_beta": round(self.portfolio_beta, 3),
            "historical": self.historical, "scenarios": self.scenarios,
            "correlation": self.correlation, "single_name": self.single_name,
            "coverage_note": self.coverage_note,
        }

    def to_json(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.as_dict(), fh, indent=2, default=str)
        return path

    def summary(self) -> str:
        L = [f"STRESS TEST — {self.n_holdings} holdings, beta {self.portfolio_beta:.2f}"]
        for k, v in self.historical.items():
            L.append(f"  {k:14} book {v['book_drawdown']:+.1%}  vs S&P {v['benchmark_drawdown']:+.1%}")
        for k, v in self.scenarios.items():
            L.append(f"  {k:14} {v['shock']}  ->  est. {v['portfolio_impact']:+.1%}  ({v['method']})")
        L.append(f"  correlation    calm {self.correlation.get('calm', float('nan')):.2f} "
                 f"-> crisis {self.correlation.get('crisis', float('nan')):.2f}")
        sn = self.single_name
        L.append(f"  single-name    worst -100% shock = {sn.get('nav_impact', 0):+.1%} "
                 f"({sn.get('ticker', '?')} at {sn.get('weight', 0):.1%})")
        return "\n".join(L)


def _weighted_daily_returns(bundle: DataBundle, weights: Dict[str, float],
                            start: str, end: str) -> Tuple[pd.Series, float]:
    """Return the (fixed-weight) daily return series over a window + name coverage."""
    w = pd.Series(weights, dtype=float)
    present = [t for t in w.index if t in bundle.prices.columns]
    px = bundle.prices.loc[(bundle.prices.index >= pd.Timestamp(start)) &
                           (bundle.prices.index <= pd.Timestamp(end)), present]
    # Only names with data in the window (some may have IPO'd later).
    px = px.dropna(axis=1, how="all")
    if px.shape[1] == 0 or len(px) < 2:
        return pd.Series(dtype=float), 0.0
    ww = w[px.columns]
    ww = ww / ww.sum()
    rets = px.pct_change().dropna(how="all").mul(ww, axis=1).sum(axis=1)
    coverage = float(ww.sum() and len(px.columns) / max(1, len(present)))
    return rets, coverage


def _portfolio_beta(bundle: DataBundle, weights: Dict[str, float],
                    as_of: pd.Timestamp, window: int = 252, index: str = "SPX") -> float:
    if index not in bundle.benchmarks.columns:
        return 1.0
    px = bundle.prices.loc[bundle.prices.index <= as_of].tail(window + 1)
    mkt = bundle.benchmarks[index].reindex(px.index).pct_change().dropna()
    var = mkt.var(ddof=0)
    if not np.isfinite(var) or var == 0:
        return 1.0
    betas = {}
    for t in weights:
        if t in px.columns:
            r = px[t].reindex(mkt.index).pct_change()
            betas[t] = r.cov(mkt) / var
    if not betas:
        return 1.0
    w = pd.Series({t: weights[t] for t in betas}); w = w / w.sum()
    return float((w * pd.Series(betas)).sum())


def _avg_pairwise_corr(bundle: DataBundle, tickers: List[str],
                       start: str, end: str) -> float:
    present = [t for t in tickers if t in bundle.prices.columns]
    px = bundle.prices.loc[(bundle.prices.index >= pd.Timestamp(start)) &
                           (bundle.prices.index <= pd.Timestamp(end)), present]
    rets = px.pct_change().dropna(how="all")
    if rets.shape[1] < 2:
        return float("nan")
    c = rets.corr().values
    iu = np.triu_indices_from(c, 1)
    return float(np.nanmean(c[iu]))


def stress_test(
    weights: Dict[str, float],
    bundle: DataBundle,
    constraints: PortfolioConstraints,
    as_of: Optional[pd.Timestamp] = None,
    market_shock: float = -0.20,
    rate_shock_bps: float = 100.0,
    windows: Optional[Dict[str, Tuple[str, str]]] = None,
) -> StressReport:
    """Run the full stress suite on a portfolio and return an exportable report."""
    as_of = bundle.dates()[-1] if as_of is None else pd.Timestamp(as_of)
    windows = windows or DEFAULT_WINDOWS
    w = {t: v for t, v in weights.items() if v > 0}
    tickers = list(w)
    beta = _portfolio_beta(bundle, w, as_of)

    # --- historical: hold today's book through each crisis window ------------
    historical: Dict[str, Dict[str, float]] = {}
    covs = []
    for name, (s, e) in windows.items():
        book, cov = _weighted_daily_returns(bundle, w, s, e)
        covs.append(cov)
        bench = pd.Series(dtype=float)
        if "SPX" in bundle.benchmarks.columns:
            b = bundle.benchmarks["SPX"]
            bench = b[(b.index >= pd.Timestamp(s)) & (b.index <= pd.Timestamp(e))].pct_change().dropna()
        historical[name] = {
            "window": f"{s}..{e}",
            "book_drawdown": round(max_drawdown(book), 4) if not book.empty else None,
            "benchmark_drawdown": round(max_drawdown(bench), 4) if not bench.empty else None,
        }

    # --- hypothetical scenarios via estimated sensitivities ------------------
    fund = bundle.fundamentals_asof(as_of)
    scenarios: Dict[str, Dict[str, object]] = {}
    # Market shock: portfolio_beta * shock.
    scenarios["market_-20pct"] = {
        "shock": f"{market_shock:+.0%} equity market",
        "portfolio_impact": round(beta * market_shock, 4),
        "method": f"portfolio beta {beta:.2f} x market shock",
    }
    # Rate shock: characteristic 'equity duration' proxy from valuation (high P/E
    # = longer duration = more rate-sensitive).  Illustrative, not a fitted beta.
    pe = fund["pe"].reindex(tickers)
    pe_rank = pe.rank(pct=True).fillna(0.5)
    duration = 2.0 + 8.0 * pe_rank                 # ~2y (cheap) .. ~10y (expensive)
    wser = pd.Series(w); wser = wser / wser.sum()
    port_dur = float((wser * duration.reindex(wser.index).fillna(6.0)).sum())
    rate_impact = -(rate_shock_bps / 10000.0) * port_dur
    scenarios["rate_+100bps"] = {
        "shock": f"+{rate_shock_bps:.0f}bps rates",
        "portfolio_impact": round(rate_impact, 4),
        "method": f"weighted equity-duration proxy {port_dur:.1f}y (from valuation)",
    }

    # --- crisis vs calm correlation -----------------------------------------
    crisis_win = windows.get("covid_2020", list(windows.values())[0])
    correlation = {
        "calm": round(_avg_pairwise_corr(bundle, tickers, *CALM_WINDOW), 3),
        "crisis": round(_avg_pairwise_corr(bundle, tickers, *crisis_win), 3),
        "calm_window": f"{CALM_WINDOW[0]}..{CALM_WINDOW[1]}",
        "crisis_window": f"{crisis_win[0]}..{crisis_win[1]}",
    }

    # --- single-name worst case ---------------------------------------------
    top = max(w.items(), key=lambda kv: kv[1]) if w else ("", 0.0)
    single_name = {"ticker": top[0], "weight": round(top[1], 4),
                   "nav_impact": round(-top[1], 4),
                   "cap": constraints.max_weight_per_stock,
                   "top5_weight": round(sum(sorted(w.values(), reverse=True)[:5]), 4)}

    cov = float(np.nanmin(covs)) if covs else 1.0
    note = ("historical windows hold the *current* book through past prices; "
            f"min name-coverage {cov:.0%} (names that IPO'd later are dropped). "
            "no pre-2019 (2008) data — see hypothetical scenarios.")

    return StressReport(as_of=str(as_of.date()), n_holdings=len(w),
                        portfolio_beta=beta, historical=historical,
                        scenarios=scenarios, correlation=correlation,
                        single_name=single_name, coverage_note=note)
