"""Data model and the pluggable data-source seam.

The whole engine reads market data through :class:`DataBundle`, which is
produced by a :class:`DataSource`.  The default implementation
(:mod:`wharton_ml_engine.data.synthetic`) fabricates a deterministic, fully
offline universe so the pipeline runs end-to-end with no API keys.  A real
run swaps in a source backed by the FinancialDataset API, CSV exports, or the
approved-universe file — nothing downstream changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Fundamental fields every source must provide (point-in-time, as-of a date).
FUNDAMENTAL_FIELDS: List[str] = [
    "market_cap",       # USD
    "pe",               # price / earnings
    "pb",               # price / book
    "ev_ebit",          # enterprise value / EBIT
    "fcf_yield",        # free cash flow / market cap
    "accruals",         # lower is better (earnings quality)
    "roe",              # return on equity
    "roic",             # return on invested capital
    "gross_margin",
    "earnings_vol",     # stdev of yearly earnings growth (lower = stabler)
    "debt_equity",      # leverage (lower is better)
    "interest_coverage",# EBIT / interest (higher is better)
    "revenue_growth",   # yoy
    "eps_growth",       # yoy
    "growth_stability", # 0-1, consistency of growth
    "dividend_yield",
    "payout_ratio",     # dividends / earnings
    "adv_usd",          # average daily traded value, USD
]


@dataclass
class SecurityMeta:
    """Static descriptive metadata for one security."""

    ticker: str
    name: str
    sector: str
    themes: List[str] = field(default_factory=list)
    pays_dividend: bool = False


@dataclass
class DataBundle:
    """Everything the engine needs about the market.

    * ``prices`` — daily adjusted close, index = dates, columns = tickers.
    * ``benchmarks`` — daily levels for indices (``SPX``, ``NDX``) and sector
      ETFs, index = dates, columns = names.
    * ``fundamentals`` — a point-in-time panel indexed by ``(date, ticker)``.
      Values are sampled on a monthly grid and read via :meth:`fundamentals_asof`
      to avoid look-ahead bias.
    * ``meta`` — per-ticker :class:`SecurityMeta`.
    """

    prices: pd.DataFrame
    benchmarks: pd.DataFrame
    fundamentals: pd.DataFrame            # MultiIndex (date, ticker)
    meta: Dict[str, SecurityMeta]

    # -- convenience accessors ------------------------------------------------

    @property
    def tickers(self) -> List[str]:
        return list(self.prices.columns)

    @property
    def sectors(self) -> Dict[str, str]:
        return {t: m.sector for t, m in self.meta.items()}

    def dates(self) -> pd.DatetimeIndex:
        return self.prices.index

    def fundamentals_asof(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Return the latest fundamentals available *on or before* ``as_of``.

        Guarantees no look-ahead: only rows dated <= ``as_of`` are considered,
        and the most recent snapshot per ticker is returned, indexed by ticker.
        """
        as_of = pd.Timestamp(as_of)
        level_dates = self.fundamentals.index.get_level_values(0)
        visible = self.fundamentals.loc[level_dates <= as_of]
        if visible.empty:
            return pd.DataFrame(columns=FUNDAMENTAL_FIELDS)
        # Keep the last snapshot per ticker.
        visible = visible.reset_index()
        latest = (
            visible.sort_values("date")
            .groupby("ticker", as_index=True)
            .tail(1)
            .set_index("ticker")
        )
        return latest[FUNDAMENTAL_FIELDS]

    def prices_upto(self, as_of: pd.Timestamp) -> pd.DataFrame:
        as_of = pd.Timestamp(as_of)
        return self.prices.loc[self.prices.index <= as_of]

    def before(self, as_of: pd.Timestamp) -> "DataBundle":
        """Return a copy truncated to data on or before ``as_of``.

        Used to build a leak-free *training* bundle: a model fit on
        ``bundle.before(train_end)`` can never see prices or fundamentals from
        the out-of-sample backtest window.
        """
        as_of = pd.Timestamp(as_of)
        fund_dates = self.fundamentals.index.get_level_values(0)
        return DataBundle(
            prices=self.prices.loc[self.prices.index <= as_of].copy(),
            benchmarks=self.benchmarks.loc[self.benchmarks.index <= as_of].copy(),
            fundamentals=self.fundamentals.loc[fund_dates <= as_of].copy(),
            meta=dict(self.meta),
        )

    def restrict_universe(self, tickers: List[str]) -> "DataBundle":
        """Return a copy filtered to ``tickers`` (the approved list, PRD 4.2)."""
        keep = [t for t in tickers if t in self.prices.columns]
        fund = self.fundamentals.loc[
            self.fundamentals.index.get_level_values(1).isin(keep)
        ]
        return DataBundle(
            prices=self.prices[keep].copy(),
            benchmarks=self.benchmarks.copy(),
            fundamentals=fund.copy(),
            meta={t: self.meta[t] for t in keep},
        )


class DataSource:
    """Abstract base for anything that can produce a :class:`DataBundle`."""

    def load(self) -> DataBundle:  # pragma: no cover - interface
        raise NotImplementedError
