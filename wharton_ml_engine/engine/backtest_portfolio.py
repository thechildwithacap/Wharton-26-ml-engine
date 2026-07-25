"""Head-to-head portfolio backtest: ML-driven engine vs. rule-only engine.

Walks forward through an out-of-sample window, rebalancing monthly.  At each
rebalance it builds the portfolio with the *full* engine (integrated score →
constrained construction) **twice** — once using only the hand-built signals
(rule-only) and once with the trained ``ml_alpha`` folded in — then holds each
book and stitches net-of-cost daily returns.  The benchmark is included for
reference.

Leakage control
---------------
The ML model is trained on ``bundle.before(train_end)`` and the backtest runs on
``[train_end, end]``.  Features at each rebalance are point-in-time, and the
model never saw any return realised inside the test window.  The two strategies
share identical style weights, client fit, risk and liquidity inputs on every
date — the *only* difference is whether ``ml_alpha`` enters the integrated
score — so the comparison isolates the ML contribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..client.fit import compute_client_fit
from ..client.profile import ClientProfile
from ..config import EngineConfig
from ..data.source import DataBundle
from ..risk.checks import model_robustness_scores, turnover_and_cost
from ..risk.stock import liquidity_model, stock_risk_model
from ..signals import compute_signals
from ..utils import annualise_return, annualise_vol, max_drawdown, sharpe_ratio
from .construct import construct_portfolio
from .integrate import integrate_scores


@dataclass
class StrategyResult:
    name: str
    returns: pd.Series                       # net-of-cost daily returns
    gross_returns: pd.Series
    weights_history: Dict[pd.Timestamp, Dict[str, float]] = field(default_factory=dict)
    turnover: List[float] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def equity_curve(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()


@dataclass
class ComparisonResult:
    strategies: Dict[str, StrategyResult]
    benchmark: pd.Series
    summary: pd.DataFrame
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    ml_metrics: Dict[str, float] = field(default_factory=dict)
    retrain_dates: List[pd.Timestamp] = field(default_factory=list)

    def equity_curves(self) -> pd.DataFrame:
        curves = {n: s.equity_curve for n, s in self.strategies.items()}
        curves["benchmark"] = (1.0 + self.benchmark).cumprod()
        return pd.DataFrame(curves).dropna(how="all")


def _rebalance_dates(bundle: DataBundle, start, end) -> pd.DatetimeIndex:
    idx = bundle.prices.index
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    window = idx[(idx >= start) & (idx <= end)]
    if len(window) == 0:
        return pd.DatetimeIndex([])
    first = pd.Series(window, index=window).groupby([window.year, window.month]).first()
    return pd.DatetimeIndex(sorted(first.values))


def _metrics(returns: pd.Series, benchmark: Optional[pd.Series] = None,
             turnover: Optional[List[float]] = None) -> Dict[str, float]:
    r = returns.dropna()
    if r.empty:
        return {k: float("nan") for k in
                ["ann_return", "ann_vol", "sharpe", "max_drawdown", "worst_year",
                 "total_return", "avg_turnover", "info_ratio"]}
    yearly = (1 + r).groupby(r.index.year).prod() - 1.0
    out = {
        "ann_return": annualise_return(r),
        "ann_vol": annualise_vol(r),
        "sharpe": sharpe_ratio(r),
        "max_drawdown": max_drawdown(r),
        "worst_year": float(yearly.min()) if len(yearly) else float("nan"),
        "total_return": float((1 + r).prod() - 1.0),
        "avg_turnover": float(np.mean(turnover)) if turnover else float("nan"),
    }
    if benchmark is not None:
        active = (r - benchmark.reindex(r.index)).dropna()
        te = active.std(ddof=0) * np.sqrt(252)
        out["info_ratio"] = float(annualise_return(active) / te) if te else float("nan")
    else:
        out["info_ratio"] = float("nan")
    return out


def _train_bundle(bundle: DataBundle, as_of: pd.Timestamp,
                  lookback_days: Optional[int]) -> DataBundle:
    """Leak-free training bundle ending at ``as_of``.

    Expanding window by default; if ``lookback_days`` is set, a rolling window
    of that length (so the model forgets stale regimes).
    """
    sub = bundle.before(as_of)
    if not lookback_days:
        return sub
    start = pd.Timestamp(as_of) - pd.Timedelta(days=int(lookback_days))
    fd = sub.fundamentals.index.get_level_values(0)
    return DataBundle(
        prices=sub.prices.loc[sub.prices.index >= start],
        benchmarks=sub.benchmarks.loc[sub.benchmarks.index >= start],
        fundamentals=sub.fundamentals.loc[fd >= start],
        meta=dict(sub.meta),
    )


def _segment_returns(bundle: DataBundle, weights: Dict[str, float],
                     d: pd.Timestamp, nxt: pd.Timestamp) -> pd.Series:
    if not weights:
        return pd.Series(dtype=float)
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    px = bundle.prices
    mask = (px.index > d) & (px.index <= nxt)
    seg = px.loc[mask, list(w.index)]
    if len(seg) < 2:
        return pd.Series(dtype=float)
    return seg.pct_change().dropna(how="all").mul(w, axis=1).sum(axis=1)


def backtest_ml_vs_rules(
    bundle: DataBundle,
    profile: ClientProfile,
    train_end,
    end=None,
    config: Optional[EngineConfig] = None,
    alpha_model: Optional[object] = None,
    ml_task: str = "regression",
    ml_horizon: Optional[int] = None,
    retrain_every: Optional[int] = None,
    train_lookback_days: Optional[int] = None,
    extended_features: bool = False,
    portfolio_value: float = 1_000_000.0,
    benchmark_index: str = "SPX",
) -> ComparisonResult:
    """Train (leak-free) then backtest the ML-driven vs rule-only engine.

    ``retrain_every`` — if set, retrain the ML model every *N* rebalances
    (walk-forward) using only data available at that rebalance, instead of a
    single fit at ``train_end``.  ``train_lookback_days`` makes each retrain use
    a rolling window of that length (default: expanding).  A caller-supplied
    ``alpha_model`` is used fixed and disables retraining.
    """
    config = config or EngineConfig()
    if ml_horizon is None:
        ml_horizon = config.ml_horizon_days
    constraints = profile.apply_to_constraints(config.constraints)
    train_end = pd.Timestamp(train_end)
    end = bundle.dates()[-1] if end is None else pd.Timestamp(end)

    from ..ml import AlphaModel, train_alpha_model
    from ..backtest.style_weights import recommend_style_weights

    walk_forward = alpha_model is None and bool(retrain_every) and retrain_every > 0

    # --- initial model (single-fit mode) trained strictly on pre-test data ---
    ml_metrics: Dict[str, float] = {}
    retrain_dates: List[pd.Timestamp] = []
    retrain_metrics: List[Dict[str, float]] = []
    if alpha_model is None and not walk_forward:
        trained = train_alpha_model(bundle.before(train_end), task=ml_task,
                                    horizon_days=ml_horizon, extended=extended_features)
        alpha_model = AlphaModel(trained)
        ml_metrics = dict(trained.metrics)
    elif alpha_model is not None:
        ml_metrics = dict(getattr(alpha_model, "metrics", {}) or {})

    def _maybe_retrain(d: pd.Timestamp) -> None:
        """Walk-forward refit on data available at ``d`` (leak-free)."""
        nonlocal alpha_model
        tb = _train_bundle(bundle, d, train_lookback_days)
        try:
            trained = train_alpha_model(tb, task=ml_task, horizon_days=ml_horizon,
                                        extended=extended_features)
        except ValueError:
            if alpha_model is None:                       # cannot proceed yet
                raise
            return
        alpha_model = AlphaModel(trained)
        retrain_dates.append(d)
        retrain_metrics.append(dict(trained.metrics))

    rdates = _rebalance_dates(bundle, train_end, end)
    strategies = {
        "rule_only": StrategyResult("rule_only", pd.Series(dtype=float), pd.Series(dtype=float)),
        "ml_driven": StrategyResult("ml_driven", pd.Series(dtype=float), pd.Series(dtype=float)),
    }
    prev_w = {"rule_only": {}, "ml_driven": {}}
    gross_acc = {"rule_only": [], "ml_driven": []}
    net_acc = {"rule_only": [], "ml_driven": []}

    for i, d in enumerate(rdates):
        nxt = rdates[i + 1] if i + 1 < len(rdates) else end

        if walk_forward and (i == 0 or i % retrain_every == 0):
            _maybe_retrain(d)

        # ---- inputs shared by both strategies (computed once) --------------
        base = compute_signals(bundle, profile, d)                 # no ml_alpha
        fit = compute_client_fit(bundle, profile, d)
        srisk = stock_risk_model(bundle, d)
        liq = liquidity_model(bundle, d, constraints, portfolio_value)
        robustness = model_robustness_scores(base)
        style = recommend_style_weights(bundle, profile, d, template_results=None).weights
        market_cap = bundle.fundamentals_asof(d).get("market_cap")

        variants = {"rule_only": base}
        ml_sig = base.copy()
        ml_sig["ml_alpha"] = alpha_model.score(bundle, d).reindex(base.index)
        variants["ml_driven"] = ml_sig

        for name, sig in variants.items():
            scored = integrate_scores(sig, fit, srisk, liq, style, profile, robustness)
            w = construct_portfolio(scored, bundle, constraints, liq, market_cap).weights
            strategies[name].weights_history[d] = w
            strategies[name].turnover.append(
                turnover_and_cost(prev_w[name], w, config.costs).turnover)

            seg = _segment_returns(bundle, w, d, nxt)
            if not seg.empty:
                cost = turnover_and_cost(prev_w[name], w, config.costs).cost_estimate
                net = seg.copy()
                net.iloc[0] = net.iloc[0] - cost            # charge costs at rebalance
                gross_acc[name].append(seg)
                net_acc[name].append(net)
            prev_w[name] = w

    # --- benchmark over the test window --------------------------------------
    b = bundle.benchmarks[benchmark_index]
    test_start = rdates[0] if len(rdates) else train_end
    b = b[(b.index >= test_start) & (b.index <= end)]
    bench_ret = b.pct_change().dropna()

    for name in strategies:
        gr = (pd.concat(gross_acc[name]).sort_index() if gross_acc[name] else pd.Series(dtype=float))
        nr = (pd.concat(net_acc[name]).sort_index() if net_acc[name] else pd.Series(dtype=float))
        gr = gr[~gr.index.duplicated(keep="first")]
        nr = nr[~nr.index.duplicated(keep="first")]
        strategies[name].gross_returns = gr
        strategies[name].returns = nr
        strategies[name].metrics = _metrics(nr, bench_ret, strategies[name].turnover)

    # Aggregate walk-forward training metrics across all refits.
    if retrain_metrics:
        def _avg(key):
            vals = [m.get(key) for m in retrain_metrics
                    if m.get(key) is not None and np.isfinite(m.get(key))]
            return float(np.mean(vals)) if vals else float("nan")
        ml_metrics = {"mean_ic": _avg("mean_ic"), "ic_t_stat": _avg("ic_t_stat"),
                      "hit_rate": _avg("hit_rate"), "n_retrains": len(retrain_metrics)}

    summary = pd.DataFrame({n: s.metrics for n, s in strategies.items()}).T
    summary.loc["benchmark"] = _metrics(bench_ret)
    return ComparisonResult(
        strategies=strategies, benchmark=bench_ret, summary=summary,
        train_end=train_end, test_start=test_start, test_end=end,
        ml_metrics=ml_metrics, retrain_dates=retrain_dates,
    )


def format_comparison(result: ComparisonResult) -> str:
    L: List[str] = []
    L.append("=" * 74)
    L.append(" ML-DRIVEN vs RULE-ONLY ENGINE — out-of-sample portfolio backtest")
    L.append("=" * 74)
    if result.retrain_dates:
        L.append(f" Training     : WALK-FORWARD, {len(result.retrain_dates)} refits "
                 f"(first {result.retrain_dates[0].date()}, "
                 f"last {result.retrain_dates[-1].date()})")
    else:
        L.append(f" Training     : single fit through {result.train_end.date()}")
    L.append(f" Test window  : {result.test_start.date()} .. {result.test_end.date()}")
    if result.ml_metrics:
        m = result.ml_metrics
        label = "avg train IC" if result.retrain_dates else "train IC"
        L.append(f" ML model     : {label} {m.get('mean_ic', float('nan')):.3f} "
                 f"(t={m.get('ic_t_stat', float('nan')):.2f}, "
                 f"hit {m.get('hit_rate', float('nan')):.0%})")
    L.append("-" * 74)
    cols = ["ann_return", "ann_vol", "sharpe", "max_drawdown", "info_ratio", "avg_turnover"]
    header = f" {'strategy':<12}" + "".join(f"{c:>14}" for c in cols)
    L.append(header)
    for name in list(result.strategies) + ["benchmark"]:
        row = result.summary.loc[name]
        cells = []
        for c in cols:
            v = row.get(c, float("nan"))
            if c in ("ann_return", "ann_vol", "max_drawdown", "avg_turnover"):
                cells.append(f"{v:>13.1%} " if pd.notna(v) else f"{'n/a':>14}")
            else:
                cells.append(f"{v:>14.2f}" if pd.notna(v) else f"{'n/a':>14}")
        L.append(f" {name:<12}" + "".join(cells))
    L.append("-" * 74)
    rule, ml = result.summary.loc["rule_only"], result.summary.loc["ml_driven"]
    dret = ml["ann_return"] - rule["ann_return"]
    dsharpe = ml["sharpe"] - rule["sharpe"]
    verdict = ("ML ADDED value" if dsharpe > 0.05 else
               "ML HURT" if dsharpe < -0.05 else "roughly NEUTRAL")
    L.append(f" ML vs rules  : ann-return {dret:+.1%},  Sharpe {dsharpe:+.2f}  ->  {verdict}")
    L.append("=" * 74)
    return "\n".join(L)
