"""Free real-data source: provider prices + SEC EDGAR fundamentals (PRD §4).

Combines daily prices from a free-tier provider (TwelveData or FMP) with
fundamentals from SEC EDGAR into a :class:`DataBundle`, so the whole engine —
signals, risk, backtest, ML — runs on real US-equity data at no cost.

Only the price provider needs a (free, no-card) API key; SEC EDGAR needs none.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .sec_edgar import build_sec_fundamentals
from .source import DataBundle, DataSource, SecurityMeta
from .web_prices import PRICE_FETCHERS, default_get

# index/ETF label -> provider ticker
DEFAULT_BENCHMARKS: Dict[str, str] = {
    "SPX": "SPY", "NDX": "QQQ", "ETF_TECH": "XLK", "ETF_HEAL": "XLV",
    "ETF_FINA": "XLF", "ETF_ENER": "XLE",
}


class WebDataSource(DataSource):
    def __init__(
        self,
        tickers: List[str],
        start: str,
        end: str,
        price_provider: str = "twelvedata",
        price_api_key: Optional[str] = None,
        benchmarks: Optional[Dict[str, str]] = None,
        themes_map: Optional[Dict[str, List[str]]] = None,
        price_get: Optional[Callable[[str], dict]] = None,
        sec_fetch: Optional[Callable[[str], dict]] = None,
        throttle_s: float = 0.15,
    ) -> None:
        if price_provider not in PRICE_FETCHERS:
            raise ValueError(f"unknown price_provider {price_provider!r}; "
                             f"choose from {sorted(PRICE_FETCHERS)}")
        self.tickers = [t.upper() for t in tickers]
        self.start, self.end = start, end
        self.price_provider = price_provider
        self.price_api_key = price_api_key
        self.benchmarks = benchmarks if benchmarks is not None else DEFAULT_BENCHMARKS
        self.themes_map = themes_map or {}
        self.price_get = price_get or default_get()
        self.sec_fetch = sec_fetch
        self.throttle_s = throttle_s

    def _prices(self, symbol: str) -> Optional[pd.DataFrame]:
        fetch = PRICE_FETCHERS[self.price_provider]
        df = fetch(symbol, self.start, self.end, self.price_api_key or "", self.price_get)
        if self.throttle_s:
            time.sleep(self.throttle_s)      # be polite to free-tier rate limits
        return df

    def load(self) -> DataBundle:
        close: Dict[str, pd.Series] = {}
        adv: Dict[str, float] = {}
        for t in self.tickers:
            df = self._prices(t)
            if df is not None and len(df) > 20:
                close[t] = df["close"]
                adv[t] = float((df["close"] * df["volume"]).tail(63).mean())
        if not close:
            raise RuntimeError("no price data returned — check API key / credits / symbols")
        prices = pd.DataFrame(close).sort_index()

        bench_cols: Dict[str, pd.Series] = {}
        for label, sym in self.benchmarks.items():
            df = self._prices(sym)
            if df is not None and len(df) > 20:
                bench_cols[label] = df["close"]
        benchmarks = (pd.DataFrame(bench_cols).reindex(prices.index).ffill()
                      if bench_cols else pd.DataFrame(index=prices.index))

        fundamentals, sectors, names = build_sec_fundamentals(
            list(close), prices, self.start, self.end, adv=adv, fetch=self.sec_fetch)

        meta: Dict[str, SecurityMeta] = {}
        last_div = {}
        if not fundamentals.empty and "dividend_yield" in fundamentals.columns:
            last_div = (fundamentals.reset_index().sort_values("date")
                        .groupby("ticker")["dividend_yield"].last().to_dict())
        for t in close:
            meta[t] = SecurityMeta(
                ticker=t, name=names.get(t, t),
                sector=sectors.get(t, "Unknown"),
                themes=list(self.themes_map.get(t, [])),
                pays_dividend=bool(last_div.get(t, 0) and last_div.get(t, 0) > 0),
            )
        return DataBundle(prices=prices, benchmarks=benchmarks,
                          fundamentals=fundamentals, meta=meta)
