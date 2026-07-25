"""Macro-regime pod: market-wide state + a regime-conditioned stock factor.

The existing quant pod measures a *static* beta (``macro_tilt`` = low beta is
defensive).  In a trending market that is just an inverse-volatility bet, and
the factor audit showed it running persistently negative.  Macro information
only helps a **cross-sectional** stock model when it *changes sign with the
regime*: high-beta / cyclical names should score well when the tape is risk-on
and badly when it is risk-off.  That is what :func:`macro_factor_model` builds.

Everything is derived from the benchmark series already in the bundle (SPX, and
NDX as a growth-leadership proxy) — no external macro feed — and is strictly
point-in-time (only prices up to ``as_of`` are read).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from ..data.source import DataBundle
from ..utils import clip_score, pct_rank


@dataclass
class MacroRegime:
    """Market-wide state at a decision date (all point-in-time)."""

    trend: float            # SPX 12-1 momentum (risk appetite proxy)
    vol_annual: float       # SPX trailing 63d realised vol, annualised
    vol_z: float            # that vol vs its own 1y history (>0 = elevated)
    drawdown: float         # SPX drawdown from trailing 1y peak (<= 0)
    growth_leadership: float  # NDX minus SPX, trailing 126d (growth vs broad)
    risk_on: float          # 0..1 composite; 1 = full risk-on

    def label(self) -> str:
        if self.risk_on >= 0.66:
            return "risk-on"
        if self.risk_on <= 0.34:
            return "risk-off"
        return "neutral"

    def to_json(self) -> Dict[str, float]:
        return {
            "trend": round(self.trend, 4), "vol_annual": round(self.vol_annual, 4),
            "vol_z": round(self.vol_z, 3), "drawdown": round(self.drawdown, 4),
            "growth_leadership": round(self.growth_leadership, 4),
            "risk_on": round(self.risk_on, 3), "label": self.label(),
        }


def _bench(bundle: DataBundle, name: str, upto: pd.Timestamp) -> pd.Series:
    if name not in bundle.benchmarks.columns:
        return pd.Series(dtype=float)
    b = bundle.benchmarks[name]
    return b.loc[b.index <= upto].dropna()


def macro_regime(bundle: DataBundle, as_of: pd.Timestamp) -> MacroRegime:
    """Derive the market-wide macro regime from the index benchmarks."""
    spx = _bench(bundle, "SPX", as_of)
    if len(spx) < 130:
        return MacroRegime(0.0, 0.15, 0.0, 0.0, 0.0, 0.5)

    # 12-1 trend (skip the last month, like price momentum).
    p_252 = spx.iloc[-min(252, len(spx) - 1)]
    p_21 = spx.iloc[-min(21, len(spx) - 1)]
    trend = float(p_21 / p_252 - 1.0)

    rets = spx.pct_change().dropna()
    vol_annual = float(rets.tail(63).std(ddof=0) * np.sqrt(252))
    # rolling 63d vol history over the last year, for a z-score of current vol.
    roll = rets.rolling(63).std(ddof=0).dropna().tail(252) * np.sqrt(252)
    vol_z = float((vol_annual - roll.mean()) / roll.std(ddof=0)) if len(roll) > 20 \
        and roll.std(ddof=0) > 0 else 0.0

    peak = spx.tail(252).cummax().iloc[-1]
    drawdown = float(spx.iloc[-1] / peak - 1.0) if peak > 0 else 0.0

    ndx = _bench(bundle, "NDX", as_of)
    growth_leadership = 0.0
    if len(ndx) >= 130:
        n0, n1 = ndx.iloc[-min(126, len(ndx) - 1)], ndx.iloc[-1]
        s0, s1 = spx.iloc[-min(126, len(spx) - 1)], spx.iloc[-1]
        growth_leadership = float((n1 / n0) - (s1 / s0))

    # Composite risk-on: positive trend, benign vol, shallow drawdown.
    risk_on = float(np.clip(
        0.5
        + 1.2 * np.tanh(4.0 * trend)
        - 0.35 * np.tanh(vol_z)
        + 3.0 * drawdown,           # drawdown is negative -> reduces risk_on
        0.0, 1.0))
    return MacroRegime(trend, vol_annual, vol_z, drawdown, growth_leadership, risk_on)


def macro_factor_model(bundle: DataBundle, as_of: pd.Timestamp,
                       window: int = 252) -> pd.DataFrame:
    """Per-stock macro exposures + a **regime-conditioned** ``macro_fit`` score.

    * ``beta`` — sensitivity to SPX (cyclicality).
    * ``growth_beta`` — sensitivity to the NDX-minus-SPX growth-leadership factor.
    * ``macro_fit`` — 0-100: how well the name's exposure matches *today's*
      regime.  Risk-on rewards high beta / growth exposure; risk-off rewards the
      opposite.  Because the sign flips with the regime, this is not a static
      low-vol tilt — it only pays when the exposure suits the environment.
    """
    tickers = bundle.tickers
    out = pd.DataFrame(index=tickers)
    px = bundle.prices_upto(as_of).tail(window + 1)
    reg = macro_regime(bundle, as_of)

    if len(px) < 60 or "SPX" not in bundle.benchmarks.columns:
        out["beta"] = 1.0
        out["growth_beta"] = 0.0
        out["macro_fit"] = 50.0
        out["regime_risk_on"] = reg.risk_on
        return out

    mkt = bundle.benchmarks["SPX"].reindex(px.index).pct_change()
    ndx = (bundle.benchmarks["NDX"].reindex(px.index).pct_change()
           if "NDX" in bundle.benchmarks.columns else mkt)
    growth_spread = (ndx - mkt)
    rets = px.pct_change()

    common = mkt.dropna().index.intersection(rets.dropna(how="all").index)
    mkt_c = mkt.reindex(common)
    gs_c = growth_spread.reindex(common)
    var_m = mkt_c.var(ddof=0)
    var_g = gs_c.var(ddof=0)

    betas, gbetas = {}, {}
    for t in tickers:
        r = rets[t].reindex(common)
        cov_m = r.cov(mkt_c)
        betas[t] = cov_m / var_m if (np.isfinite(cov_m) and var_m) else 1.0
        cov_g = r.cov(gs_c)
        gbetas[t] = cov_g / var_g if (np.isfinite(cov_g) and var_g) else 0.0
    out["beta"] = pd.Series(betas)
    out["growth_beta"] = pd.Series(gbetas)

    # Regime tilt in [-1, 1]: +1 fully risk-on, -1 fully risk-off.
    tilt = 2.0 * reg.risk_on - 1.0
    # Standardise exposures cross-sectionally, then align with the regime.
    def _z(s):
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd > 0 else s * 0.0
    beta_z = _z(out["beta"])
    gbeta_z = _z(out["growth_beta"])
    growth_sign = float(np.tanh(50.0 * reg.growth_leadership))  # -1..1
    alignment = tilt * beta_z + 0.5 * tilt * growth_sign * gbeta_z
    out["macro_fit"] = clip_score(pct_rank(alignment, ascending=True))
    out["regime_risk_on"] = reg.risk_on
    return out
