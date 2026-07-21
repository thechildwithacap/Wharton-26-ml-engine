"""Stock-level risk and liquidity models (PRD 6.2).

* Stock Risk Model — volatility, beta, max drawdown and drawdown frequency,
  distilled into a 1-5 Risk Band, a 0-100 ``risk_score`` (higher = safer) and
  a tail-risk flag.
* Liquidity & Tradability Model — average daily traded value into a Liquidity
  Score and a maximum safe trade size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PortfolioConstraints
from ..data.source import DataBundle
from ..utils import clip_score, max_drawdown, pct_rank, score_lower_is_better


def stock_risk_model(bundle: DataBundle, as_of: pd.Timestamp,
                     window: int = 252) -> pd.DataFrame:
    px = bundle.prices_upto(as_of).tail(window + 1)
    tickers = bundle.tickers
    out = pd.DataFrame(index=tickers)

    rets = px.pct_change().dropna(how="all")
    out["volatility"] = rets.std(ddof=0) * np.sqrt(252)

    if "SPX" in bundle.benchmarks.columns and len(rets) > 20:
        mkt = bundle.benchmarks["SPX"].reindex(px.index).pct_change().reindex(rets.index)
        var = mkt.var(ddof=0)
        out["beta"] = pd.Series(
            {t: (rets[t].cov(mkt) / var if var else 1.0) for t in tickers}
        )
    else:
        out["beta"] = 1.0

    mdd, dd_freq = {}, {}
    for t in tickers:
        r = rets[t].dropna()
        mdd[t] = max_drawdown(r)
        if not r.empty:
            curve = (1 + r).cumprod()
            dd = curve / curve.cummax() - 1.0
            dd_freq[t] = float((dd < -0.10).mean())    # share of days > 10% under water
        else:
            dd_freq[t] = np.nan
    out["max_drawdown"] = pd.Series(mdd)
    out["drawdown_freq"] = pd.Series(dd_freq)

    # Composite safety score (higher = safer) from vol, |mdd| and dd frequency.
    vol_s = score_lower_is_better(out["volatility"])
    mdd_s = score_lower_is_better(out["max_drawdown"].abs())
    freq_s = score_lower_is_better(out["drawdown_freq"])
    out["risk_score"] = clip_score(0.5 * vol_s + 0.3 * mdd_s + 0.2 * freq_s)

    # Risk Band 1 (safest) .. 5 (riskiest) from vol quintiles.
    try:
        out["risk_band"] = pd.qcut(out["volatility"].rank(method="first"),
                                   5, labels=[1, 2, 3, 4, 5]).astype(int)
    except ValueError:
        out["risk_band"] = 3

    # Tail-risk flag: very high vol or a deep historical drawdown.
    vol_hi = out["volatility"] > out["volatility"].quantile(0.85)
    dd_deep = out["max_drawdown"] < -0.40
    out["tail_risk_flag"] = (vol_hi | dd_deep)
    return out


def liquidity_model(bundle: DataBundle, as_of: pd.Timestamp,
                    constraints: PortfolioConstraints,
                    portfolio_value: float = 1_000_000.0) -> pd.DataFrame:
    fund = bundle.fundamentals_asof(as_of)
    tickers = list(fund.index)
    out = pd.DataFrame(index=tickers)

    adv = fund["adv_usd"].reindex(tickers)
    out["adv_usd"] = adv
    out["liquidity_score"] = clip_score(pct_rank(adv, ascending=True))

    # Max safe trade = participation cap * ADV.
    out["max_safe_trade_usd"] = adv * constraints.max_participation_of_adv
    # As a fraction of the book, what's the largest position we can hold and
    # still be able to exit within the participation limit.
    out["max_weight_by_liquidity"] = (out["max_safe_trade_usd"] / portfolio_value).clip(upper=1.0)
    # Flag names too thin to reach even the minimum position size.
    out["illiquid_flag"] = out["max_weight_by_liquidity"] < constraints.min_weight_per_stock
    return out
