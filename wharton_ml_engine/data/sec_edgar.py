"""Free, no-key fundamentals from SEC EDGAR (PRD §4).

SEC's XBRL ``companyfacts`` API exposes every US filer's reported financials,
and — crucially — stamps each fact with the date it was **filed**, which gives
true point-in-time availability (no look-ahead).  This module maps those raw
concepts to the engine's ``FUNDAMENTAL_FIELDS``.

Design choices for robustness:

* **Annual (FY) flows.**  Income-statement / cash-flow items are taken from
  10-K (fiscal-year) figures, which are unambiguous, instead of stitching
  quarterly YTD/Q4 values.
* **Instantaneous stocks.**  Balance-sheet items (equity, assets, liabilities,
  cash, debt, shares) use the latest point available as-of each date.
* **Price-aware ratios monthly.**  P/E, P/B, market cap and the yields are
  recomputed on a monthly grid using each month's price, so valuation signals
  track the market instead of going stale between filings.

SEC asks for a descriptive User-Agent with contact info; set ``SEC_USER_AGENT``
to override the default.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import SECTORS
from .source import FUNDAMENTAL_FIELDS

SEC_BASE = "https://data.sec.gov"
SEC_WWW = "https://www.sec.gov"


def _default_sec_fetch() -> Callable[[str], dict]:
    ca = "/root/.ccr/ca-bundle.crt"
    ctx = ssl.create_default_context(cafile=ca) if os.path.exists(ca) else \
        ssl.create_default_context()
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(urllib.request.getproxies()),
        urllib.request.HTTPSHandler(context=ctx),
    )
    ua = os.environ.get("SEC_USER_AGENT",
                        "wharton-ml-engine research (contact: set SEC_USER_AGENT)")

    def fetch(url: str) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": ua,
                                                   "Accept-Encoding": "gzip, deflate"})
        with opener.open(req, timeout=60) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                import gzip
                raw = gzip.decompress(raw)
            return json.loads(raw.decode("utf-8"))

    return fetch


# ---- SIC (SEC industry code) -> our canonical sector --------------------
def _sic_to_sector(sic: Optional[str]) -> str:
    """Map an SEC SIC code to a canonical GICS-ish sector.

    Ordered most-specific-first (first match wins), because narrow industry
    ranges (e.g. pharma 2833-2836, computers 3570-3579) sit inside broader
    manufacturing bands and must be tested before them.
    """
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return "Unknown"
    specific = [
        (2833, 2836, "Health Care"),        # pharma / biologics
        (2080, 2085, "Consumer Staples"),   # beverages
        (2100, 2199, "Consumer Staples"),   # tobacco / food
        (2840, 2844, "Consumer Staples"),   # soap, cosmetics, cleaning
        (3570, 3579, "Technology"),         # computers & office equipment
        (3661, 3669, "Technology"),         # communications equipment
        (3670, 3679, "Technology"),         # electronic components / semis
        (3820, 3829, "Technology"),         # measuring / control instruments
        (3840, 3851, "Health Care"),        # medical devices
        (7370, 7379, "Technology"),         # software & IT services
        (3760, 3769, "Industrials"),        # guided missiles / defense
    ]
    broad = [
        (100, 999, "Materials"), (1000, 1499, "Energy"), (1500, 1799, "Industrials"),
        (2000, 2079, "Consumer Staples"), (2200, 2399, "Consumer Discretionary"),
        (2400, 2799, "Industrials"), (2800, 2899, "Materials"),
        (2900, 2999, "Energy"), (3000, 3299, "Consumer Discretionary"),
        (3300, 3499, "Materials"), (3500, 3599, "Industrials"),
        (3600, 3699, "Technology"), (3700, 3799, "Consumer Discretionary"),
        (3800, 3899, "Health Care"), (4000, 4799, "Industrials"),
        (4800, 4899, "Communication Services"), (4900, 4999, "Utilities"),
        (5000, 5199, "Industrials"), (5200, 5999, "Consumer Discretionary"),
        (6000, 6499, "Financials"), (6500, 6599, "Real Estate"),
        (6600, 6799, "Financials"), (7000, 7299, "Consumer Discretionary"),
        (7300, 7399, "Technology"), (7400, 7999, "Communication Services"),
        (8000, 8099, "Health Care"),
    ]
    for lo, hi, sec in specific + broad:
        if lo <= code <= hi:
            return sec if sec in SECTORS else "Unknown"
    return "Unknown"


def ticker_cik_map(fetch: Callable[[str], dict]) -> Dict[str, str]:
    data = fetch(f"{SEC_WWW}/files/company_tickers.json")
    out: Dict[str, str] = {}
    for row in data.values():
        out[str(row["ticker"]).upper()] = f"{int(row['cik_str']):010d}"
    return out


def _rows(gaap: dict, names: List[str], unit_hint: Optional[str] = None) -> List[dict]:
    """Return the fact rows for the first present concept among ``names``."""
    for n in names:
        if n in gaap:
            units = gaap[n]["units"]
            unit = unit_hint if (unit_hint and unit_hint in units) else list(units)[0]
            return units[unit]
    return []


def _annual(gaap: dict, names: List[str], unit_hint: Optional[str] = None) -> pd.DataFrame:
    """Latest-filed fiscal-year value per fiscal year (columns: fy, val, filed)."""
    rows = [r for r in _rows(gaap, names, unit_hint)
            if r.get("fp") == "FY" and str(r.get("form", "")).startswith("10-K")
            and r.get("val") is not None and r.get("fy")]
    if not rows:
        return pd.DataFrame(columns=["fy", "val", "filed", "end"])
    df = pd.DataFrame([{"fy": int(r["fy"]), "val": float(r["val"]),
                        "filed": pd.Timestamp(r["filed"]), "end": pd.Timestamp(r["end"])}
                       for r in rows])
    return df.sort_values("filed").drop_duplicates("fy", keep="last").sort_values("fy")


def _instant(gaap: dict, names: List[str], unit_hint: Optional[str] = None) -> pd.DataFrame:
    """Balance-sheet point values as a filed-date-sorted series."""
    rows = [r for r in _rows(gaap, names, unit_hint) if r.get("val") is not None]
    if not rows:
        return pd.DataFrame(columns=["filed", "val"])
    df = pd.DataFrame([{"filed": pd.Timestamp(r["filed"]), "val": float(r["val"])}
                       for r in rows])
    return df.sort_values("filed").drop_duplicates("filed", keep="last")


def _asof(df: pd.DataFrame, date: pd.Timestamp, col: str = "val") -> float:
    if df.empty:
        return float("nan")
    vis = df[df["filed"] <= date]
    return float(vis[col].iloc[-1]) if not vis.empty else float("nan")


def _annual_asof(df: pd.DataFrame, date: pd.Timestamp):
    """Return (latest, previous) annual values visible at ``date``."""
    if df.empty:
        return float("nan"), float("nan")
    vis = df[df["filed"] <= date]
    if vis.empty:
        return float("nan"), float("nan")
    last = float(vis["val"].iloc[-1])
    prev = float(vis["val"].iloc[-2]) if len(vis) >= 2 else float("nan")
    return last, prev


def _yoy_instant(df: pd.DataFrame, date: pd.Timestamp) -> float:
    """Year-over-year change of a balance-sheet series, point-in-time.

    Compares the latest value visible at ``date`` with the value visible ~one
    year earlier.  Returns NaN when either point is unavailable or the base is
    non-positive.  Used for net share issuance and asset growth.
    """
    cur = _asof(df, date)
    prev = _asof(df, date - pd.Timedelta(days=365))
    if not (np.isfinite(cur) and np.isfinite(prev)) or prev <= 0:
        return float("nan")
    return cur / prev - 1.0


def build_sec_fundamentals(
    tickers: List[str],
    prices: pd.DataFrame,
    start: str,
    end: str,
    adv: Optional[Dict[str, float]] = None,
    fetch: Optional[Callable[[str], dict]] = None,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    """Return (fundamentals_panel, sectors, names) from SEC EDGAR.

    ``prices`` provides the monthly price used for valuation ratios.  The panel
    is indexed by (date, ticker) on a monthly grid; every value is point-in-time
    (only facts filed on/before the row date are used).
    """
    fetch = fetch or _default_sec_fetch()
    adv = adv or {}
    cik_map = ticker_cik_map(fetch)

    months = pd.bdate_range(start=start, end=end, freq="BME")
    index_tuples: List[tuple] = []
    rows: List[Dict[str, float]] = []
    sectors: Dict[str, str] = {}
    names: Dict[str, str] = {}

    for t in tickers:
        cik = cik_map.get(t.upper())
        if not cik:
            continue
        try:
            cf = fetch(f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik}.json")
        except Exception:
            continue
        gaap = cf.get("facts", {}).get("us-gaap", {})
        dei = cf.get("facts", {}).get("dei", {})
        names[t] = cf.get("entityName", t)

        # sector via submissions (sic)
        try:
            sub = fetch(f"{SEC_BASE}/submissions/CIK{cik}.json")
            sectors[t] = _sic_to_sector(sub.get("sic"))
        except Exception:
            sectors[t] = "Unknown"

        # annual flows
        rev = _annual(gaap, ["RevenueFromContractWithCustomerExcludingAssessedTax",
                             "Revenues", "SalesRevenueNet"])
        ni = _annual(gaap, ["NetIncomeLoss"])
        eps = _annual(gaap, ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
                      unit_hint="USD/shares")
        gross = _annual(gaap, ["GrossProfit"])
        opinc = _annual(gaap, ["OperatingIncomeLoss"])
        ocf = _annual(gaap, ["NetCashProvidedByUsedInOperatingActivities"])
        capex = _annual(gaap, ["PaymentsToAcquirePropertyPlantAndEquipment"])
        div = _annual(gaap, ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"])
        interest = _annual(gaap, ["InterestExpense", "InterestExpenseDebt"])

        # instantaneous stocks
        equity = _instant(gaap, ["StockholdersEquity",
                                 "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"])
        liab = _instant(gaap, ["Liabilities"])
        assets = _instant(gaap, ["Assets"])
        cash = _instant(gaap, ["CashAndCashEquivalentsAtCarryingValue"])
        ltdebt = _instant(gaap, ["LongTermDebtNoncurrent", "LongTermDebt"])
        shares = _instant(dei, ["EntityCommonStockSharesOutstanding"], unit_hint="shares")
        # Weighted diluted share count (income-statement) is a cleaner issuance
        # series than the cover-page count when available.
        shares_wtd = _annual(gaap, ["WeightedAverageNumberOfDilutedSharesOutstanding",
                                    "WeightedAverageNumberOfSharesOutstandingBasic"],
                             unit_hint="shares")

        px = prices[t] if t in prices.columns else None
        if px is None:
            continue

        eps_growth_hist: List[float] = []
        for m in months:
            price = px[px.index <= m]
            if price.empty:
                continue
            p = float(price.iloc[-1])

            rev_l, rev_p = _annual_asof(rev, m)
            ni_l, _ = _annual_asof(ni, m)
            eps_l, eps_p = _annual_asof(eps, m)
            gross_l, _ = _annual_asof(gross, m)
            op_l, _ = _annual_asof(opinc, m)
            ocf_l, _ = _annual_asof(ocf, m)
            capex_l, _ = _annual_asof(capex, m)
            div_l, _ = _annual_asof(div, m)
            int_l, _ = _annual_asof(interest, m)
            eq = _asof(equity, m)
            lb = _asof(liab, m)
            ast = _asof(assets, m)
            csh = _asof(cash, m)
            ltd = _asof(ltdebt, m)
            sh = _asof(shares, m)

            mktcap = p * sh if np.isfinite(sh) else float("nan")
            rec = {f: float("nan") for f in FUNDAMENTAL_FIELDS}
            rec["market_cap"] = mktcap
            rec["pe"] = (p / eps_l) if (np.isfinite(eps_l) and eps_l > 0) else float("nan")
            rec["pb"] = (mktcap / eq) if (np.isfinite(mktcap) and np.isfinite(eq) and eq > 0) else float("nan")
            if np.isfinite(mktcap) and np.isfinite(op_l) and op_l > 0:
                ev = mktcap + (ltd if np.isfinite(ltd) else 0) - (csh if np.isfinite(csh) else 0)
                rec["ev_ebit"] = ev / op_l
            if np.isfinite(mktcap) and mktcap > 0:
                fcf = (ocf_l - capex_l) if (np.isfinite(ocf_l) and np.isfinite(capex_l)) else float("nan")
                rec["fcf_yield"] = (fcf / mktcap) if np.isfinite(fcf) else float("nan")
                rec["dividend_yield"] = (div_l / mktcap) if np.isfinite(div_l) else 0.0
            rec["roe"] = (ni_l / eq) if (np.isfinite(ni_l) and np.isfinite(eq) and eq > 0) else float("nan")
            if np.isfinite(ni_l) and np.isfinite(eq) and (eq + (ltd if np.isfinite(ltd) else 0)) > 0:
                rec["roic"] = ni_l / (eq + (ltd if np.isfinite(ltd) else 0))
            rec["gross_margin"] = (gross_l / rev_l) if (np.isfinite(gross_l) and np.isfinite(rev_l) and rev_l) else float("nan")
            rec["debt_equity"] = (lb / eq) if (np.isfinite(lb) and np.isfinite(eq) and eq > 0) else float("nan")
            rec["interest_coverage"] = (op_l / int_l) if (np.isfinite(op_l) and np.isfinite(int_l) and int_l > 0) else float("nan")
            rec["revenue_growth"] = (rev_l / rev_p - 1.0) if (np.isfinite(rev_l) and np.isfinite(rev_p) and rev_p > 0) else float("nan")
            eg = (eps_l / eps_p - 1.0) if (np.isfinite(eps_l) and np.isfinite(eps_p) and eps_p > 0) else float("nan")
            rec["eps_growth"] = eg
            rec["payout_ratio"] = (div_l / ni_l) if (np.isfinite(div_l) and np.isfinite(ni_l) and ni_l > 0) else 0.0
            # -- capital-discipline / financing signals (independent of the above) --
            # Sloan accruals: earnings not backed by cash, scaled by assets.
            if np.isfinite(ni_l) and np.isfinite(ocf_l) and np.isfinite(ast) and ast > 0:
                rec["accruals"] = (ni_l - ocf_l) / ast
            # Net share issuance: prefer weighted diluted count YoY, else cover-page.
            wtd_l, wtd_p = _annual_asof(shares_wtd, m)
            if np.isfinite(wtd_l) and np.isfinite(wtd_p) and wtd_p > 0:
                rec["net_issuance"] = wtd_l / wtd_p - 1.0
            else:
                rec["net_issuance"] = _yoy_instant(shares, m)
            # Asset growth: the investment/over-expansion anomaly.
            rec["asset_growth"] = _yoy_instant(assets, m)
            if np.isfinite(eg):
                eps_growth_hist.append(eg)
            if len(eps_growth_hist) >= 3:
                vol = float(np.std(eps_growth_hist, ddof=0))
                rec["earnings_vol"] = vol
                rec["growth_stability"] = 1.0 / (1.0 + max(vol, 0.0))
            rec["adv_usd"] = adv.get(t, float("nan"))

            index_tuples.append((m, t))
            rows.append(rec)

    if not rows:
        idx = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
        return pd.DataFrame(columns=FUNDAMENTAL_FIELDS, index=idx), sectors, names
    idx = pd.MultiIndex.from_tuples(index_tuples, names=["date", "ticker"])
    panel = pd.DataFrame(rows, index=idx)[FUNDAMENTAL_FIELDS].sort_index()
    return panel, sectors, names
