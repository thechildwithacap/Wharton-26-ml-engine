"""SEC Form 4 insider transactions — free, from EDGAR.

Officers, directors and >=10% holders must report their trades on **Form 4**
within two business days.  Insider *buying* is the informative side: a manager
who buys with their own money is signalling confidence, whereas a sale can mean
nothing more than diversification or a tax bill.  The article calls this out
explicitly, and SEC publishes it for free.

Where the data comes from
-------------------------
``data.sec.gov/submissions/CIK##########.json`` lists every filing a company has
made, including each Form 4 with its **filing date** and accession number.  That
gives an honest, point-in-time count of insider filings per name per period with
no extra credentials — only a descriptive ``SEC_USER_AGENT``.

Honest limitation (stated, not hidden): the submissions index gives filing
*counts and dates*, not transaction size or direction.  Getting buy-vs-sell and
dollar value requires fetching each Form 4 XML document.  ``fetch_detail=True``
does that; left off (the default) the module produces a filing-*activity*
signal rather than a true net-buying signal, and ``has_detail`` on the result
says which you got — so the signal is never silently overstated.
"""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

SEC_BASE = "https://data.sec.gov"
SEC_WWW = "https://www.sec.gov"

# Form 4 transaction codes.  "P" = open-market purchase (the meaningful buy),
# "S" = open-market sale.  Awards/exercises (A, M, G, F) are compensation, not
# conviction, so they are tracked separately and excluded from net buying.
BUY_CODES = {"P"}
SELL_CODES = {"S"}
COMP_CODES = {"A", "M", "G", "F"}


@dataclass
class InsiderActivity:
    """Per-ticker insider activity over a lookback window."""

    counts: pd.DataFrame            # index=ticker; filings/buys/sells/values
    has_detail: bool = False
    as_of: Optional[pd.Timestamp] = None
    meta: Dict[str, object] = field(default_factory=dict)


def _default_fetch() -> Callable[[str], object]:
    ca = "/root/.ccr/ca-bundle.crt"
    ctx = ssl.create_default_context(cafile=ca) if os.path.exists(ca) else \
        ssl.create_default_context()
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(urllib.request.getproxies()),
        urllib.request.HTTPSHandler(context=ctx),
    )
    ua = os.environ.get("SEC_USER_AGENT",
                        "wharton-ml-engine research (set SEC_USER_AGENT)")

    def fetch(url: str):
        req = urllib.request.Request(url, headers={"User-Agent": ua,
                                                   "Accept-Encoding": "gzip, deflate"})
        with opener.open(req, timeout=60) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                import gzip
                raw = gzip.decompress(raw)
            text = raw.decode("utf-8", errors="replace")
        if url.endswith(".json"):
            return json.loads(text)
        return text

    return fetch


def _parse_form4_xml(text: str) -> Dict[str, float]:
    """Pull transaction code / shares / price out of a Form 4 XML document.

    Deliberately regex-based and tolerant: Form 4 XML varies between filers and
    a parse failure must degrade to "no detail", never crash the pipeline.
    """
    codes = re.findall(r"<transactionCode>\s*([A-Z])\s*</transactionCode>", text)
    shares = re.findall(
        r"<transactionShares>\s*<value>\s*([\d.]+)\s*</value>", text)
    prices = re.findall(
        r"<transactionPricePerShare>\s*<value>\s*([\d.]+)\s*</value>", text)

    buys = sells = 0
    buy_value = sell_value = 0.0
    for i, c in enumerate(codes):
        sh = float(shares[i]) if i < len(shares) else 0.0
        pr = float(prices[i]) if i < len(prices) else 0.0
        val = sh * pr
        if c in BUY_CODES:
            buys += 1
            buy_value += val
        elif c in SELL_CODES:
            sells += 1
            sell_value += val
    return {"buys": buys, "sells": sells,
            "buy_value": buy_value, "sell_value": sell_value}


def fetch_insider_activity(
    tickers: List[str],
    cik_map: Dict[str, str],
    as_of: pd.Timestamp,
    lookback_days: int = 180,
    fetch: Optional[Callable[[str], object]] = None,
    fetch_detail: bool = False,
    max_detail_per_ticker: int = 25,
) -> InsiderActivity:
    """Count Form 4 activity per ticker in the ``lookback_days`` before ``as_of``.

    Strictly point-in-time: only filings dated on or before ``as_of`` count, so
    the signal can be used in a backtest without look-ahead.
    """
    fetch = fetch or _default_fetch()
    as_of = pd.Timestamp(as_of)
    start = as_of - pd.Timedelta(days=lookback_days)

    rows: Dict[str, Dict[str, float]] = {}
    detail_ok = False
    for t in tickers:
        cik = cik_map.get(t.upper())
        if not cik:
            continue
        try:
            sub = fetch(f"{SEC_BASE}/submissions/CIK{cik}.json")
        except Exception:
            continue
        recent = (sub or {}).get("filings", {}).get("recent", {})
        forms = recent.get("form", []) or []
        dates = recent.get("filingDate", []) or []
        accs = recent.get("accessionNumber", []) or []

        rec = {"filings": 0, "buys": 0, "sells": 0,
               "buy_value": 0.0, "sell_value": 0.0}
        hits: List[str] = []
        for i, form in enumerate(forms):
            if str(form).strip() != "4":
                continue
            d = pd.to_datetime(dates[i], errors="coerce") if i < len(dates) else pd.NaT
            if pd.isna(d) or not (start <= d <= as_of):
                continue
            rec["filings"] += 1
            if i < len(accs):
                hits.append(str(accs[i]))

        if fetch_detail and hits:
            for acc in hits[:max_detail_per_ticker]:
                a = acc.replace("-", "")
                url = (f"{SEC_WWW}/Archives/edgar/data/{int(cik)}/{a}/"
                       f"{acc}.txt")
                try:
                    text = fetch(url)
                except Exception:
                    continue
                if not isinstance(text, str):
                    continue
                d = _parse_form4_xml(text)
                for k in ("buys", "sells", "buy_value", "sell_value"):
                    rec[k] += d[k]
                detail_ok = True
        rows[t] = rec

    counts = (pd.DataFrame(rows).T if rows else
              pd.DataFrame(columns=["filings", "buys", "sells",
                                    "buy_value", "sell_value"]))
    return InsiderActivity(counts=counts, has_detail=detail_ok, as_of=as_of,
                           meta={"lookback_days": lookback_days,
                                 "n_tickers": len(rows)})


def insider_score(activity: InsiderActivity, index=None) -> pd.DataFrame:
    """Turn raw Form 4 activity into 0-100 cross-sectional scores.

    With transaction detail, ``insider`` reflects **net buying intensity**
    (buys vs sells, value-weighted).  Without it, the score falls back to
    filing *activity* and is explicitly flagged via ``detail`` so a
    filing-count proxy is never mistaken for a conviction signal.
    """
    from ..utils import clip_score, pct_rank

    c = activity.counts
    idx = index if index is not None else c.index
    out = pd.DataFrame(index=idx)
    if c.empty:
        out["insider_filings"] = np.nan
        out["insider_net"] = np.nan
        out["insider"] = 50.0
        out["detail"] = activity.has_detail
        return out

    c = c.reindex(idx)
    out["insider_filings"] = c["filings"]
    if activity.has_detail:
        buys, sells = c["buys"].fillna(0), c["sells"].fillna(0)
        bval, sval = c["buy_value"].fillna(0), c["sell_value"].fillna(0)
        # Net buying ratio in [-1, 1]; value-weighted when values are present.
        denom_n = (buys + sells).replace(0, np.nan)
        denom_v = (bval + sval).replace(0, np.nan)
        net_n = (buys - sells) / denom_n
        net_v = (bval - sval) / denom_v
        net = net_v.where(net_v.notna(), net_n)
        out["insider_net"] = net
        # No insider activity at all is neutral, not bearish.
        out["insider"] = clip_score(pct_rank(net, ascending=True)).fillna(50.0)
    else:
        out["insider_net"] = np.nan
        out["insider"] = clip_score(
            pct_rank(c["filings"], ascending=True)).fillna(50.0)
    out["detail"] = activity.has_detail
    return out
