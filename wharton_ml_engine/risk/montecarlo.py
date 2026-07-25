"""Monte Carlo simulation of portfolio outcomes (PRD §6.2 risk extension).

Given a book's weights and the daily-return history of its holdings, this
projects the distribution of outcomes over a forward horizon — the range a
point backtest number can't show.  It answers the questions that actually
matter for a competition: *what's the chance I'm down at the deadline, how bad
is the tail, and how deep a drawdown should I expect along the way?*

Three engines, all numpy-only:

* ``block_bootstrap`` (default) — resample blocks of consecutive historical
  days, preserving fat tails, volatility clustering and cross-asset correlation
  exactly as they occurred.  The most honest with real return data.
* ``bootstrap`` — i.i.d. daily resampling (no autocorrelation).
* ``normal`` / ``t`` — parametric multivariate draws from the sample mean vector
  and covariance (``t`` adds fat tails via a Student-t with ``df``).

An optional ``annual_drift`` override replaces the historical mean with an
expected return you supply (e.g. the ML model's), keeping the historical
volatility and correlation structure — useful for "if I expect X%, what's the
spread?" scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    horizon_days: int
    n_sims: int
    method: str
    trading_days_per_year: int
    mean_return: float
    median_return: float
    std_return: float
    prob_loss: float
    percentiles: Dict[str, float]           # p05,p25,p50,p75,p95 terminal return
    var_95: float                           # 95% VaR (loss, positive number)
    cvar_95: float                          # expected shortfall beyond VaR
    max_drawdown_median: float
    max_drawdown_p95: float
    ann_return_median: float
    ann_vol: float
    meta: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> Dict[str, object]:
        return {
            "horizon_days": self.horizon_days, "n_sims": self.n_sims,
            "method": self.method,
            "mean_return": round(self.mean_return, 4),
            "median_return": round(self.median_return, 4),
            "std_return": round(self.std_return, 4),
            "prob_loss": round(self.prob_loss, 4),
            "percentiles": {k: round(v, 4) for k, v in self.percentiles.items()},
            "var_95": round(self.var_95, 4), "cvar_95": round(self.cvar_95, 4),
            "max_drawdown_median": round(self.max_drawdown_median, 4),
            "max_drawdown_p95": round(self.max_drawdown_p95, 4),
            "ann_return_median": round(self.ann_return_median, 4),
            "ann_vol": round(self.ann_vol, 4),
            "meta": self.meta,
        }

    def summary(self) -> str:
        p = self.percentiles
        return (f"MC[{self.method}, {self.n_sims}x{self.horizon_days}d] "
                f"median {self.median_return:+.1%}  "
                f"P(loss) {self.prob_loss:.0%}  "
                f"5-95% [{p['p05']:+.1%}, {p['p95']:+.1%}]  "
                f"VaR95 {self.var_95:.1%}  CVaR95 {self.cvar_95:.1%}  "
                f"median maxDD {self.max_drawdown_median:.1%}")


def _portfolio_daily(returns: pd.DataFrame, weights: pd.Series) -> np.ndarray:
    w = weights.reindex(returns.columns).fillna(0.0).to_numpy()
    s = w.sum()
    if s > 0:
        w = w / s
    return returns.fillna(0.0).to_numpy() @ w


def _paths_stats(paths: np.ndarray, tdays: int, horizon: int):
    """paths: (n_sims, horizon) daily simple returns -> terminal + drawdown arrays."""
    growth = np.cumprod(1.0 + paths, axis=1)
    terminal = growth[:, -1] - 1.0
    running_max = np.maximum.accumulate(growth, axis=1)
    dd = (growth / running_max - 1.0).min(axis=1)      # most negative per path
    return terminal, dd


def simulate_portfolio(
    returns: pd.DataFrame,
    weights: Union[pd.Series, Dict[str, float], Sequence[float]],
    horizon_days: int = 63,
    n_sims: int = 10000,
    method: str = "block_bootstrap",
    block: int = 5,
    annual_drift: Optional[float] = None,
    t_df: float = 5.0,
    trading_days_per_year: int = 252,
    seed: int = 0,
) -> MonteCarloResult:
    """Simulate the forward return distribution of a weighted portfolio.

    ``returns`` is the daily simple-return history of the holdings (columns =
    tickers).  ``weights`` may be a Series/dict keyed by ticker or a plain
    sequence aligned to the columns; it is renormalised to sum to 1.
    """
    if isinstance(weights, dict):
        weights = pd.Series(weights)
    elif not isinstance(weights, pd.Series):
        weights = pd.Series(list(weights), index=returns.columns)

    rets = returns.dropna(how="all").copy()
    if rets.shape[0] < 30 or rets.shape[1] == 0:
        raise ValueError("need >=30 rows of return history to simulate")
    rng = np.random.default_rng(seed)
    H = int(horizon_days)

    port = _portfolio_daily(rets, weights)          # historical daily port returns
    mu_d = float(np.mean(port))
    # Optional drift override: shift daily mean to hit the target annual return.
    drift_shift = 0.0
    if annual_drift is not None:
        target_mu_d = (1.0 + annual_drift) ** (1.0 / trading_days_per_year) - 1.0
        drift_shift = target_mu_d - mu_d

    if method in ("block_bootstrap", "bootstrap"):
        if method == "bootstrap":
            idx = rng.integers(0, len(port), size=(n_sims, H))
            paths = port[idx] + drift_shift
        else:
            b = max(1, int(block))
            n_blocks = int(np.ceil(H / b))
            starts = rng.integers(0, max(1, len(port) - b + 1), size=(n_sims, n_blocks))
            offs = np.arange(b)
            # (n_sims, n_blocks, b) -> (n_sims, n_blocks*b) -> trim to H
            block_idx = (starts[:, :, None] + offs[None, None, :])
            paths = port[block_idx].reshape(n_sims, n_blocks * b)[:, :H] + drift_shift
    elif method in ("normal", "t"):
        # Parametric at the asset level to keep the correlation structure.
        w = weights.reindex(rets.columns).fillna(0.0).to_numpy()
        sw = w.sum()
        w = w / sw if sw > 0 else w
        mu = rets.mean().to_numpy()
        cov = np.cov(rets.fillna(0.0).to_numpy(), rowvar=False)
        cov = np.atleast_2d(cov)
        try:
            L = np.linalg.cholesky(cov + 1e-12 * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            L = np.diag(np.sqrt(np.clip(np.diag(cov), 0, None)))
        z = rng.standard_normal(size=(n_sims * H, cov.shape[0]))
        if method == "t":
            g = rng.chisquare(t_df, size=(n_sims * H, 1)) / t_df
            z = z / np.sqrt(g)                       # multivariate-t scaling
        asset_paths = (mu + z @ L.T).reshape(n_sims, H, cov.shape[0])
        paths = asset_paths @ w + drift_shift
    else:
        raise ValueError(f"unknown method: {method!r}")

    terminal, dd = _paths_stats(paths, trading_days_per_year, H)
    pct = {f"p{q:02d}": float(np.percentile(terminal, q))
           for q in (5, 25, 50, 75, 95)}
    var95 = float(-np.percentile(terminal, 5))       # loss as a positive number
    tail = terminal[terminal <= np.percentile(terminal, 5)]
    cvar95 = float(-tail.mean()) if tail.size else var95
    years = H / trading_days_per_year
    median_term = float(np.median(terminal))
    ann_ret_med = float((1.0 + median_term) ** (1.0 / years) - 1.0) if years > 0 else median_term
    ann_vol = float(np.std(terminal) / np.sqrt(years)) if years > 0 else float(np.std(terminal))

    return MonteCarloResult(
        horizon_days=H, n_sims=int(n_sims), method=method,
        trading_days_per_year=trading_days_per_year,
        mean_return=float(np.mean(terminal)), median_return=median_term,
        std_return=float(np.std(terminal)),
        prob_loss=float(np.mean(terminal < 0.0)),
        percentiles=pct, var_95=var95, cvar_95=cvar95,
        max_drawdown_median=float(np.median(dd)),
        max_drawdown_p95=float(np.percentile(dd, 5)),   # worst 5% drawdown
        ann_return_median=ann_ret_med, ann_vol=ann_vol,
        meta={"n_assets": int(rets.shape[1]), "hist_days": int(rets.shape[0]),
              "annual_drift_override": annual_drift,
              "hist_ann_return": round((1 + mu_d) ** trading_days_per_year - 1, 4)},
    )


def simulate_from_bundle(
    bundle,
    weights: Union[pd.Series, Dict[str, float]],
    as_of: Optional[pd.Timestamp] = None,
    lookback_days: int = 504,
    **kw,
) -> MonteCarloResult:
    """Convenience wrapper: pull each holding's daily returns from the bundle
    (point-in-time, ``lookback_days`` up to ``as_of``) and simulate."""
    if isinstance(weights, dict):
        weights = pd.Series(weights)
    as_of = pd.Timestamp(as_of) if as_of is not None else bundle.dates()[-1]
    held = [t for t in weights.index if t in bundle.prices.columns]
    px = bundle.prices.loc[bundle.prices.index <= as_of, held].tail(lookback_days + 1)
    rets = px.pct_change().dropna(how="all")
    return simulate_portfolio(rets, weights.reindex(held), **kw)
