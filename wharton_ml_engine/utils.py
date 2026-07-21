"""Cross-sectional scoring utilities shared by every signal model.

Convention used throughout the engine:

* Every model emits a score on a **0-100** scale where **higher is more
  attractive** (cheaper value name, higher quality, safer risk, etc.).
* Scores are produced by ranking a raw metric *cross-sectionally* (across the
  universe on a single decision date), which makes them comparable and
  robust to outliers and changing scale over time.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd


def winsorize(s: pd.Series, lower: float = 0.02, upper: float = 0.98) -> pd.Series:
    """Clip a series to the given quantiles to tame outliers."""
    s = s.astype(float)
    if s.notna().sum() < 3:
        return s
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def zscore(s: pd.Series) -> pd.Series:
    """Standardise (mean 0, std 1), robust to all-NaN / zero-variance input."""
    s = s.astype(float)
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def pct_rank(s: pd.Series, ascending: bool = True) -> pd.Series:
    """Percentile-rank a series onto 0-100.

    ``ascending=True`` means *larger raw value -> higher score*.  Set
    ``ascending=False`` for metrics where *smaller is better* (e.g. P/E,
    debt/equity, volatility).
    """
    s = s.astype(float)
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=s.index)
    if valid.nunique() == 1:
        # No dispersion: everyone is neutral.
        out = pd.Series(np.nan, index=s.index)
        out.loc[valid.index] = 50.0
        return out
    ranks = valid.rank(ascending=ascending, method="average", pct=True)
    out = pd.Series(np.nan, index=s.index)
    out.loc[ranks.index] = ranks * 100.0
    return out


def score_lower_is_better(s: pd.Series) -> pd.Series:
    return pct_rank(s, ascending=False)


def score_higher_is_better(s: pd.Series) -> pd.Series:
    return pct_rank(s, ascending=True)


def weighted_blend(
    components: Mapping[str, pd.Series],
    weights: Optional[Mapping[str, float]] = None,
) -> pd.Series:
    """Blend several 0-100 sub-scores into one, ignoring missing components.

    Weights are renormalised over whatever sub-scores are present for each
    row, so a stock missing one input is scored fairly on the rest rather
    than being pushed toward zero.
    """
    if not components:
        raise ValueError("weighted_blend requires at least one component")

    frame = pd.DataFrame(components)
    if weights is None:
        weights = {c: 1.0 for c in frame.columns}
    w = pd.Series({c: float(weights.get(c, 0.0)) for c in frame.columns})

    mask = frame.notna()
    weight_matrix = mask.mul(w, axis=1)
    denom = weight_matrix.sum(axis=1)
    numer = (frame.fillna(0.0) * weight_matrix).sum(axis=1)

    out = pd.Series(np.nan, index=frame.index)
    ok = denom > 0
    out.loc[ok] = numer.loc[ok] / denom.loc[ok]
    return out


def neutral_fill(s: pd.Series, value: float = 50.0) -> pd.Series:
    """Fill missing scores with a neutral value (used before final ranking)."""
    return s.fillna(value)


def clip_score(s: pd.Series) -> pd.Series:
    return s.clip(lower=0.0, upper=100.0)


def annualise_return(daily_returns: pd.Series, periods: int = 252) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return float("nan")
    growth = (1.0 + r).prod()
    years = len(r) / periods
    if years <= 0:
        return float("nan")
    return float(growth ** (1.0 / years) - 1.0)


def annualise_vol(daily_returns: pd.Series, periods: int = 252) -> float:
    r = daily_returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=0) * np.sqrt(periods))


def sharpe_ratio(daily_returns: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    ann_ret = annualise_return(daily_returns, periods)
    ann_vol = annualise_vol(daily_returns, periods)
    if not np.isfinite(ann_vol) or ann_vol == 0:
        return float("nan")
    return (ann_ret - rf) / ann_vol


def max_drawdown(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return float("nan")
    curve = (1.0 + r).cumprod()
    running_max = curve.cummax()
    dd = curve / running_max - 1.0
    return float(dd.min())
