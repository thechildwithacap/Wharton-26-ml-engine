"""Quant factor pod: price-based factors and macro sensitivity (PRD 6.1.2).

These models read the price history up to the decision date (no look-ahead)
and produce factor exposures plus 0-100 scores.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from ..data.source import DataBundle
from ..utils import clip_score, pct_rank, score_higher_is_better, score_lower_is_better, weighted_blend


def _returns(bundle: DataBundle, as_of: pd.Timestamp, lookback: int) -> pd.DataFrame:
    px = bundle.prices_upto(as_of).tail(lookback + 1)
    return px.pct_change().dropna(how="all")


def price_factor_model(bundle: DataBundle, as_of: pd.Timestamp) -> pd.DataFrame:
    """Momentum, size, low-volatility -> exposures + composite Factor Score."""
    px = bundle.prices_upto(as_of)
    tickers = bundle.tickers
    out = pd.DataFrame(index=tickers)

    if len(px) < 40:
        # Not enough history: everything neutral.
        out["momentum_12_1"] = np.nan
        out["reversal_1m"] = np.nan
        out["low_vol"] = np.nan
        out["size"] = np.nan
        out["factor"] = 50.0
        return out

    last = px.iloc[-1]
    # 12-1 momentum: total return from ~252d ago to ~21d ago (skip last month).
    p_252 = px.iloc[-min(252, len(px) - 1)]
    p_21 = px.iloc[-min(21, len(px) - 1)]
    mom = (p_21 / p_252) - 1.0
    # Short-term reversal: last ~21d return (low is attractive).
    rev = (last / p_21) - 1.0
    # Trailing volatility (annualised) over ~126d.
    rets = px.pct_change().tail(126)
    vol = rets.std(ddof=0) * np.sqrt(252)

    fund = bundle.fundamentals_asof(as_of)
    market_cap = fund["market_cap"].reindex(tickers)

    out["momentum_12_1"] = mom.reindex(tickers)
    out["reversal_1m"] = rev.reindex(tickers)
    out["volatility"] = vol.reindex(tickers)
    out["market_cap"] = market_cap

    mom_s = score_higher_is_better(out["momentum_12_1"])
    rev_s = score_lower_is_better(out["reversal_1m"])          # buy recent losers a touch
    lowvol_s = score_lower_is_better(out["volatility"])        # low vol premium
    size_s = score_lower_is_better(out["market_cap"])          # small-cap premium

    out["momentum"] = clip_score(mom_s)
    out["low_vol"] = clip_score(lowvol_s)
    out["size"] = clip_score(size_s)
    out["reversal"] = clip_score(rev_s)

    out["factor"] = clip_score(
        weighted_blend(
            {"momentum": mom_s, "low_vol": lowvol_s, "size": size_s, "reversal": rev_s},
            {"momentum": 0.45, "low_vol": 0.30, "size": 0.15, "reversal": 0.10},
        )
    )
    return out


def macro_sensitivity_model(bundle: DataBundle, as_of: pd.Timestamp,
                            window: int = 252) -> pd.DataFrame:
    """Beta to the index and a cyclical-vs-defensive Macro Tilt Score.

    ``macro_tilt`` is 0-100 where **higher = more defensive** (low beta,
    resilient), which the regime layer can lean on in risk-off environments.
    """
    px = bundle.prices_upto(as_of).tail(window + 1)
    tickers = bundle.tickers
    out = pd.DataFrame(index=tickers)

    if len(px) < 40 or "SPX" not in bundle.benchmarks.columns:
        out["beta"] = 1.0
        out["macro_tilt"] = 50.0
        return out

    mkt = bundle.benchmarks["SPX"].reindex(px.index).pct_change().dropna()
    rets = px.pct_change().reindex(mkt.index)
    var = mkt.var(ddof=0)
    betas = {}
    for t in tickers:
        cov = rets[t].cov(mkt)
        betas[t] = cov / var if (np.isfinite(cov) and var) else 1.0
    out["beta"] = pd.Series(betas)
    # Defensive = low beta.
    out["macro_tilt"] = clip_score(score_lower_is_better(out["beta"]))
    return out


def quant_scores(bundle: DataBundle, as_of: pd.Timestamp) -> pd.DataFrame:
    pf = price_factor_model(bundle, as_of)
    macro = macro_sensitivity_model(bundle, as_of)
    out = pd.DataFrame(index=bundle.tickers)
    for col in ["momentum", "low_vol", "size", "factor"]:
        out[col] = pf[col]
    out["beta"] = macro["beta"]
    out["macro_tilt"] = macro["macro_tilt"]
    out["volatility"] = pf["volatility"]
    return out
