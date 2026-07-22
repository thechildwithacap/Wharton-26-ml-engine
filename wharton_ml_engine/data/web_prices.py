"""Daily price fetchers for free-tier providers (TwelveData, FMP).

Each returns a per-symbol DataFrame indexed by date with ``close`` (adjusted
where available) and ``volume``.  The HTTP layer is an injectable ``get``
callable so the parsing is unit-tested offline with fixture payloads.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.parse
import urllib.request
from typing import Callable, Optional

import pandas as pd

_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def default_get() -> Callable[[str], dict]:
    ca = "/root/.ccr/ca-bundle.crt"
    ctx = ssl.create_default_context(cafile=ca) if os.path.exists(ca) else \
        ssl.create_default_context()
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(urllib.request.getproxies()),
        urllib.request.HTTPSHandler(context=ctx),
    )

    def get(url: str) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": _UA,
                                                   "Accept": "application/json"})
        with opener.open(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    return get


def fetch_prices_twelvedata(symbol: str, start: str, end: str, api_key: str,
                            get: Callable[[str], dict]) -> Optional[pd.DataFrame]:
    q = urllib.parse.urlencode({
        "symbol": symbol, "interval": "1day", "start_date": start, "end_date": end,
        "outputsize": 5000, "order": "ASC", "apikey": api_key,
    })
    try:
        data = get(f"https://api.twelvedata.com/time_series?{q}")
    except Exception:
        return None
    if str(data.get("status")) == "error" or "values" not in data:
        return None
    rows = []
    for v in data["values"]:
        rows.append({"date": pd.to_datetime(v["datetime"]),
                     "close": float(v["close"]),
                     "volume": float(v.get("volume") or "nan")})
    if not rows:
        return None
    return pd.DataFrame(rows).set_index("date").sort_index()


def fetch_prices_fmp(symbol: str, start: str, end: str, api_key: str,
                     get: Callable[[str], dict]) -> Optional[pd.DataFrame]:
    q = urllib.parse.urlencode({"from": start, "to": end, "apikey": api_key})
    try:
        data = get(f"https://financialmodelingprep.com/api/v3/historical-price-full/"
                   f"{symbol}?{q}")
    except Exception:
        return None
    hist = data.get("historical") if isinstance(data, dict) else None
    if not hist:
        return None
    rows = []
    for h in hist:
        rows.append({"date": pd.to_datetime(h["date"]),
                     "close": float(h.get("adjClose", h.get("close"))),
                     "volume": float(h.get("volume") or "nan")})
    return pd.DataFrame(rows).set_index("date").sort_index()


PRICE_FETCHERS = {
    "twelvedata": fetch_prices_twelvedata,
    "fmp": fetch_prices_fmp,
}
