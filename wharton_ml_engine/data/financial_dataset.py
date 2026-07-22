"""Real-data source backed by the financialdatasets.ai API (PRD §4).

``FinancialDatasetSource`` pulls daily prices, historical financial metrics and
company facts for a chosen universe and assembles them into a
:class:`DataBundle` — the same structure the synthetic source produces, so the
signal / risk / backtest / ML layers work unchanged.

Design notes
------------
* **API key.** Reads ``FINANCIAL_DATASETS_API_KEY`` from the environment (or is
  passed explicitly).  Requests set the ``X-API-KEY`` header.
* **Runs in-process.** Data is fetched by the Python process, not routed through
  any chat/tool context, so large histories are fine.
* **Point-in-time.** Fundamentals are dated at ``report_period + report_lag``
  (default 45 days) to approximate when the numbers were actually public, and
  are read via ``fundamentals_asof`` — no look-ahead.
* **Testable.** The HTTP layer is a single injectable ``fetch_json`` callable,
  so the assembly logic is unit-tested offline with fixture payloads shaped
  exactly like the real API.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import SECTORS
from .source import FUNDAMENTAL_FIELDS, DataBundle, DataSource, SecurityMeta

API_BASE = "https://api.financialdatasets.ai"

# Benchmark index proxies and a handful of sector ETFs (index -> ETF ticker).
DEFAULT_BENCHMARKS: Dict[str, str] = {
    "SPX": "SPY",
    "NDX": "QQQ",
    "ETF_TECH": "XLK",
    "ETF_HEAL": "XLV",
    "ETF_FINA": "XLF",
    "ETF_ENER": "XLE",
    "ETF_INDU": "XLI",
}

# Map financialdatasets.ai financial-metrics fields -> our FUNDAMENTAL_FIELDS.
_METRIC_MAP: Dict[str, str] = {
    "market_cap": "market_cap",
    "price_to_earnings_ratio": "pe",
    "price_to_book_ratio": "pb",
    "enterprise_value_to_ebitda_ratio": "ev_ebit",
    "free_cash_flow_yield": "fcf_yield",
    "return_on_equity": "roe",
    "return_on_invested_capital": "roic",
    "gross_margin": "gross_margin",
    "debt_to_equity": "debt_equity",
    "interest_coverage": "interest_coverage",
    "revenue_growth": "revenue_growth",
    "earnings_per_share_growth": "eps_growth",
    "payout_ratio": "payout_ratio",
    "dividend_yield": "dividend_yield",
}

# Normalise assorted sector spellings to our canonical SECTORS list.
_SECTOR_ALIASES: Dict[str, str] = {
    "information technology": "Technology",
    "technology": "Technology",
    "healthcare": "Health Care",
    "health care": "Health Care",
    "financial services": "Financials",
    "financials": "Financials",
    "consumer cyclical": "Consumer Discretionary",
    "consumer discretionary": "Consumer Discretionary",
    "consumer defensive": "Consumer Staples",
    "consumer staples": "Consumer Staples",
    "industrials": "Industrials",
    "energy": "Energy",
    "basic materials": "Materials",
    "materials": "Materials",
    "utilities": "Utilities",
    "communication services": "Communication Services",
    "real estate": "Real Estate",
}


def _to_float(x) -> float:
    try:
        if x is None or x == "":
            return float("nan")
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _normalise_sector(raw: Optional[str]) -> str:
    if not raw:
        return "Unknown"
    key = str(raw).strip().lower()
    if key in _SECTOR_ALIASES:
        return _SECTOR_ALIASES[key]
    for k, v in _SECTOR_ALIASES.items():
        if k in key:
            return v
    return raw if raw in SECTORS else "Unknown"


def _default_fetch_json(api_key: str) -> Callable[[str, Dict[str, object]], dict]:
    """Return a fetcher that GETs ``API_BASE/path?params`` with the API key.

    Honours ``HTTPS_PROXY`` (via urllib's env proxy support) and the agent CA
    bundle when present.
    """
    ca = "/root/.ccr/ca-bundle.crt"
    ctx = ssl.create_default_context(cafile=ca) if os.path.exists(ca) else \
        ssl.create_default_context()
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(urllib.request.getproxies()),
        urllib.request.HTTPSHandler(context=ctx),
    )

    # A browser-like User-Agent is required: the endpoint sits behind Cloudflare
    # bot management, which rejects the default ``Python-urllib`` client with a
    # 403 (Cloudflare error 1010) before the request ever reaches the API.
    headers = {
        "X-API-KEY": api_key,
        "Accept": "application/json",
        "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    }

    def fetch(path: str, params: Dict[str, object]) -> dict:
        qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{API_BASE}/{path.lstrip('/')}?{qs}"
        req = urllib.request.Request(url, headers=headers)
        with opener.open(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    return fetch


class FinancialDatasetSource(DataSource):
    def __init__(
        self,
        tickers: List[str],
        start: str,
        end: str,
        api_key: Optional[str] = None,
        benchmarks: Optional[Dict[str, str]] = None,
        themes_map: Optional[Dict[str, List[str]]] = None,
        report_lag_days: int = 45,
        fetch_json: Optional[Callable[[str, Dict[str, object]], dict]] = None,
    ) -> None:
        self.tickers = list(tickers)
        self.start = start
        self.end = end
        self.benchmarks = benchmarks if benchmarks is not None else DEFAULT_BENCHMARKS
        self.themes_map = themes_map or {}
        self.report_lag_days = report_lag_days

        if fetch_json is not None:
            self.fetch_json = fetch_json
        else:
            key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
            if not key:
                raise ValueError(
                    "No API key: set FINANCIAL_DATASETS_API_KEY or pass api_key= "
                    "(or inject fetch_json= for testing)."
                )
            self.fetch_json = _default_fetch_json(key)

    # ------------------------------------------------------------------ load
    def load(self) -> DataBundle:
        price_frames: Dict[str, pd.Series] = {}
        adv: Dict[str, float] = {}
        for t in self.tickers:
            s = self._prices(t)
            if s is not None and len(s) > 20:
                price_frames[t] = s["close"]
                adv[t] = float((s["close"] * s["volume"]).tail(63).mean())
        if not price_frames:
            raise RuntimeError("no price data returned for any ticker")
        prices = pd.DataFrame(price_frames).sort_index()

        bench_cols: Dict[str, pd.Series] = {}
        for name, etf in self.benchmarks.items():
            s = self._prices(etf)
            if s is not None and len(s) > 20:
                bench_cols[name] = s["close"]
        benchmarks = (pd.DataFrame(bench_cols).reindex(prices.index).ffill()
                      if bench_cols else pd.DataFrame(index=prices.index))

        fundamentals = self._fundamentals(list(price_frames), adv)
        meta = self._meta(list(price_frames), fundamentals)
        return DataBundle(prices=prices, benchmarks=benchmarks,
                          fundamentals=fundamentals, meta=meta)

    # --------------------------------------------------------------- fetchers
    def _prices(self, ticker: str) -> Optional[pd.DataFrame]:
        try:
            payload = self.fetch_json("prices/", {
                "ticker": ticker, "interval": "day", "interval_multiplier": 1,
                "start_date": self.start, "end_date": self.end,
            })
        except Exception:
            return None
        rows = payload.get("prices") or payload.get("data") or []
        if not rows:
            return None
        recs = []
        for r in rows:
            t = r.get("time") or r.get("date") or r.get("time_milliseconds")
            recs.append({
                "date": pd.to_datetime(t, utc=True, errors="coerce"),
                "close": _to_float(r.get("close")),
                "volume": _to_float(r.get("volume")),
            })
        df = pd.DataFrame(recs).dropna(subset=["date"]).sort_values("date")
        df["date"] = df["date"].dt.tz_localize(None).dt.normalize()
        return df.set_index("date")[["close", "volume"]].dropna(subset=["close"])

    def _metrics(self, ticker: str) -> List[dict]:
        try:
            payload = self.fetch_json("financial-metrics/", {
                "ticker": ticker, "period": "quarterly", "limit": 100,
                "report_period_gte": self.start, "report_period_lte": self.end,
            })
        except Exception:
            return []
        rows = payload.get("financial_metrics") or payload.get("data") or []
        return sorted(rows, key=lambda r: r.get("report_period", ""))

    def _company_facts(self, ticker: str) -> dict:
        try:
            payload = self.fetch_json("company/facts/", {"ticker": ticker})
        except Exception:
            return {}
        return payload.get("company_facts") or payload.get("data") or {}

    # ------------------------------------------------------------- assembly
    def _fundamentals(self, tickers: List[str], adv: Dict[str, float]) -> pd.DataFrame:
        index_tuples, rows = [], []
        for t in tickers:
            eps_hist: List[float] = []
            for m in self._metrics(t):
                rp = pd.to_datetime(m.get("report_period"), errors="coerce")
                if pd.isna(rp):
                    continue
                fields = {f: float("nan") for f in FUNDAMENTAL_FIELDS}
                for src, dst in _METRIC_MAP.items():
                    if src in m:
                        fields[dst] = _to_float(m[src])
                # Derive earnings volatility / growth stability from history
                # available *up to this report* (expanding, no look-ahead).
                eg = fields.get("eps_growth")
                if eg is not None and np.isfinite(eg):
                    eps_hist.append(eg)
                if len(eps_hist) >= 3:
                    vol = float(np.std(eps_hist, ddof=0))
                    fields["earnings_vol"] = vol
                    fields["growth_stability"] = float(1.0 / (1.0 + max(vol, 0.0)))
                fields["adv_usd"] = adv.get(t, float("nan"))
                if not np.isfinite(fields.get("dividend_yield", float("nan"))):
                    fields["dividend_yield"] = 0.0
                eff_date = rp + pd.Timedelta(days=self.report_lag_days)
                index_tuples.append((eff_date, t))
                rows.append(fields)
        if not rows:
            # No fundamentals available: emit an empty, well-formed panel.
            idx = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
            return pd.DataFrame(columns=FUNDAMENTAL_FIELDS, index=idx)
        idx = pd.MultiIndex.from_tuples(index_tuples, names=["date", "ticker"])
        return pd.DataFrame(rows, index=idx)[FUNDAMENTAL_FIELDS].sort_index()

    def _meta(self, tickers: List[str], fundamentals: pd.DataFrame) -> Dict[str, SecurityMeta]:
        meta: Dict[str, SecurityMeta] = {}
        last_div = {}
        if not fundamentals.empty and "dividend_yield" in fundamentals.columns:
            last_div = (fundamentals.reset_index().sort_values("date")
                        .groupby("ticker")["dividend_yield"].last().to_dict())
        for t in tickers:
            facts = self._company_facts(t)
            sector = _normalise_sector(facts.get("sector"))
            name = facts.get("name") or f"{t}"
            meta[t] = SecurityMeta(
                ticker=t, name=str(name), sector=sector,
                themes=list(self.themes_map.get(t, [])),
                pays_dividend=bool(last_div.get(t, 0) and last_div.get(t, 0) > 0),
            )
        return meta
