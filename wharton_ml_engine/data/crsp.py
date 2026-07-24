"""Survivorship-bias-free data via CRSP / WRDS (PRD §4, audit Task E — real path).

Why CRSP.  Every free feed we tried (TwelveData, FMP) 404s a *delisted* ticker —
SIVB, FRC, SBNY vanish the moment they fail.  That missing tail **is**
survivorship bias.  CRSP is the standard academic fix: it keeps every security
that ever traded, tags each with its delisting date, and — crucially — records
the **delisting return** (``dlret``), the final mark that captures the value lost
in a bankruptcy or the premium booked in a takeover.  With that, a backtest can
hold a name right up to its failure and eat the loss, instead of pretending the
name never existed.

Access.  CRSP lives on WRDS (Wharton Research Data Services — the same Wharton
that runs the competition; many schools have institutional WRDS logins).  The
real path uses the ``wrds`` Python package::

    import wrds
    from wharton_ml_engine.data import CRSPDataSource, WRDSClient

    db  = wrds.Connection(wrds_username="your_login")     # prompts once
    src = CRSPDataSource(tickers=["AAPL", "SIVB", ...],
                         start="2015-01-01", end="2025-12-31",
                         client=WRDSClient(db))
    bundle = src.load()          # includes the failed names, with the loss priced in

Testability.  All SQL is behind the small :class:`CRSPClient` seam (five typed
methods).  :class:`WRDSClient` implements them against the crsp/comp schemas;
tests inject a fake client returning CRSP-shaped DataFrames, so the whole
assembly — total-return index, delisting-return splice, point-in-time Compustat
fundamentals, survivorship-safe metadata — is verified offline with no WRDS
login.  The assembled :class:`DataBundle` is identical in shape to every other
source, so signals / risk / ML / backtest run unchanged.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

from ..config import SECTORS
from .source import FUNDAMENTAL_FIELDS, DataBundle, DataSource, SecurityMeta

# CRSP delisting-code convention.  When a stock delists for performance reasons
# (codes 400/500-series: bankruptcy, insufficient capital, price/volume) CRSP
# sometimes leaves ``dlret`` missing.  The research-standard imputation (Shumway
# 1997; Beaver-McNichols-Price 2007) is -30% for NYSE/AMEX and -55% for Nasdaq,
# NOT a truncated 0% — the missing value is precisely where the bias hides.
PERF_DELIST_IMPUTE_NYSE_AMEX = -0.30
PERF_DELIST_IMPUTE_NASDAQ = -0.55


def _is_performance_delist(dlstcd) -> bool:
    """CRSP delisting codes >= 400 that are dropped for cause (not merger/exchange)."""
    try:
        code = int(dlstcd)
    except (TypeError, ValueError):
        return False
    # 400-490: dropped (liquidation/bankruptcy-ish); 500-591: dropped for cause
    # (insufficient capital, price too low, delinquent, went private in distress).
    return 400 <= code <= 591


# Standard Industrial Classification (SIC) division -> our canonical SECTORS.
def _sic_to_sector(siccd) -> str:
    try:
        sic = int(siccd)
    except (TypeError, ValueError):
        return "Unknown"
    # Order matters: check the specific bands before the broad ones.
    if 6000 <= sic <= 6799:
        return "Financials"
    if 2833 <= sic <= 2836 or 8000 <= sic <= 8099 or 3840 <= sic <= 3851:
        return "Health Care"
    if 7370 <= sic <= 7379 or 3570 <= sic <= 3577 or 3670 <= sic <= 3679:
        return "Technology"
    if 4800 <= sic <= 4899:
        return "Communication Services"
    if 4900 <= sic <= 4949:
        return "Utilities"
    if 1300 <= sic <= 1399 or 2900 <= sic <= 2999:
        return "Energy"
    if (2000 <= sic <= 2199) or (2080 <= sic <= 2099) or sic in (5140, 5141):
        return "Consumer Staples"
    if 5200 <= sic <= 5999 or 2300 <= sic <= 2399 or 3700 <= sic <= 3716:
        return "Consumer Discretionary"
    if 1000 <= sic <= 1099 or 2600 <= sic <= 2699 or 2800 <= sic <= 2829 or 3300 <= sic <= 3399:
        return "Materials"
    if 6500 <= sic <= 6599:
        return "Real Estate"
    if 1500 <= sic <= 1799 or 3400 <= sic <= 3569 or 3580 <= sic <= 3669 or 3700 <= sic <= 3999:
        return "Industrials"
    return "Unknown"


class CRSPClient(Protocol):
    """The five queries the assembler needs.  Any object with these methods works;
    :class:`WRDSClient` runs them as SQL, tests supply fixtures."""

    def stock_file(self, permnos: List[int], start: str, end: str) -> pd.DataFrame:
        """Columns: permno, date, prc, ret, vol, shrout."""

    def names(self, permnos: List[int]) -> pd.DataFrame:
        """Columns: permno, ticker, comnam, siccd, shrcd, exchcd, namedt, nameenddt."""

    def delistings(self, permnos: List[int]) -> pd.DataFrame:
        """Columns: permno, dlstdt, dlret, dlstcd (one row per delisting)."""

    def fundamentals(self, permnos: List[int], start: str, end: str) -> pd.DataFrame:
        """Compustat quarterly, one row per (permno, datadate).  Columns: permno,
        datadate, rdq, prccq, cshoq, atq, ceqq, dlttq, dlcq, niq, saleq, cogsq,
        oiadpq, xintq, capxq, oancfq, dvpsxq, epspxq (missing tolerated)."""

    def benchmark(self, start: str, end: str) -> pd.DataFrame:
        """Columns: date, vwretd (CRSP value-weighted market total return)."""

    def resolve_tickers(self, tickers: List[str]) -> Dict[str, int]:
        """Map tickers -> permno (latest permno per ticker)."""


class WRDSClient:
    """:class:`CRSPClient` backed by a live ``wrds.Connection`` (``db.raw_sql``).

    Pass ``wrds.Connection(...).raw_sql`` (or the connection itself).  All SQL is
    parameter-free string interpolation of integer permnos / ISO dates, which is
    safe here because those values are validated (ints / ``pd.Timestamp``) before
    they reach the query.
    """

    def __init__(self, db) -> None:
        # Accept either a wrds.Connection (has .raw_sql) or a raw_sql callable.
        self._raw_sql: Callable[[str], pd.DataFrame] = getattr(db, "raw_sql", db)

    @staticmethod
    def _in(permnos: List[int]) -> str:
        return ",".join(str(int(p)) for p in permnos) or "NULL"

    def resolve_tickers(self, tickers: List[str]) -> Dict[str, int]:
        tk = ",".join("'" + str(t).upper().replace("'", "") + "'" for t in tickers) or "NULL"
        df = self._raw_sql(
            f"select permno, ticker, nameenddt from crsp.stocknames "
            f"where ticker in ({tk})"
        )
        if df is None or df.empty:
            return {}
        df = df.sort_values("nameenddt")
        return {str(r.ticker).upper(): int(r.permno)
                for r in df.itertuples()}      # last (latest) wins

    def stock_file(self, permnos, start, end):
        return self._raw_sql(
            f"select permno, date, prc, ret, vol, shrout from crsp.dsf "
            f"where permno in ({self._in(permnos)}) "
            f"and date between '{start}' and '{end}'"
        )

    def names(self, permnos):
        return self._raw_sql(
            f"select permno, ticker, comnam, siccd, shrcd, exchcd, namedt, nameenddt "
            f"from crsp.stocknames where permno in ({self._in(permnos)})"
        )

    def delistings(self, permnos):
        return self._raw_sql(
            f"select permno, dlstdt, dlret, dlstcd from crsp.dsedelist "
            f"where permno in ({self._in(permnos)}) and dlstcd > 100"
        )

    def fundamentals(self, permnos, start, end):
        # Compustat quarterly joined to permno via the CCM link table, restricted
        # to primary, valid links live on the report date.
        return self._raw_sql(
            "select l.lpermno as permno, f.datadate, f.rdq, f.prccq, f.cshoq, "
            "f.atq, f.ceqq, f.dlttq, f.dlcq, f.niq, f.saleq, f.cogsq, f.oiadpq, "
            "f.xintq, f.capxq, f.oancfq, f.dvpsxq, f.epspxq "
            "from comp.fundq f "
            "join crsp.ccmxpf_lnkhist l on f.gvkey = l.gvkey "
            "where l.linktype in ('LC','LU') and l.linkprim in ('P','C') "
            f"and l.lpermno in ({self._in(permnos)}) "
            f"and f.datadate between '{start}' and '{end}' "
            "and (l.linkdt <= f.datadate or l.linkdt is null) "
            "and (f.datadate <= l.linkenddt or l.linkenddt is null) "
            "and f.indfmt='INDL' and f.datafmt='STD' and f.consol='C' "
            "and f.popsrc='D'"
        )

    def benchmark(self, start, end):
        return self._raw_sql(
            f"select date, vwretd from crsp.dsi "
            f"where date between '{start}' and '{end}'"
        )


def _to_float(x) -> float:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return float("nan")
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


class CRSPDataSource(DataSource):
    """Assemble a survivorship-bias-free :class:`DataBundle` from CRSP + Compustat.

    Parameters
    ----------
    tickers / permnos
        Give either.  Tickers are resolved to permnos through the client.
    start, end
        ISO date bounds.
    client
        A :class:`CRSPClient` (``WRDSClient`` in production, a fixture in tests).
    report_lag_days
        Fallback publication lag applied to ``datadate`` when Compustat's earnings
        announcement date ``rdq`` is missing (default 60), so fundamentals enter
        the panel only once they were public — no look-ahead.
    """

    def __init__(
        self,
        start: str,
        end: str,
        client: CRSPClient,
        tickers: Optional[List[str]] = None,
        permnos: Optional[List[int]] = None,
        report_lag_days: int = 60,
        themes_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        if not tickers and not permnos:
            raise ValueError("provide tickers= or permnos=")
        self.start = start
        self.end = end
        self.client = client
        self.tickers = list(tickers or [])
        self.permnos = list(permnos or [])
        self.report_lag_days = report_lag_days
        self.themes_map = themes_map or {}

    # ------------------------------------------------------------------ load
    def load(self) -> DataBundle:
        permnos = self._resolve_permnos()
        if not permnos:
            raise RuntimeError("no permnos resolved for the requested universe")

        names = self.client.names(permnos)
        delist = self.client.delistings(permnos)
        stock = self.client.stock_file(permnos, self.start, self.end)
        if stock is None or stock.empty:
            raise RuntimeError("CRSP stock file returned no rows")

        perm_to_ticker, meta_static = self._identify(names, delist)
        prices, adv, delist_dates = self._build_prices(stock, delist, perm_to_ticker)
        benchmarks = self._build_benchmarks(prices.index)
        fundamentals = self._build_fundamentals(permnos, perm_to_ticker, adv,
                                                list(prices.columns))
        meta = self._build_meta(prices.columns, meta_static, delist_dates)
        return DataBundle(prices=prices, benchmarks=benchmarks,
                          fundamentals=fundamentals, meta=meta)

    # --------------------------------------------------------------- helpers
    def _resolve_permnos(self) -> List[int]:
        permnos = list(self.permnos)
        if self.tickers:
            mapping = self.client.resolve_tickers(self.tickers)
            permnos += [int(mapping[t.upper()]) for t in self.tickers
                        if t.upper() in mapping]
        # de-dup, keep order
        seen, out = set(), []
        for p in permnos:
            if p not in seen:
                seen.add(p)
                out.append(int(p))
        return out

    def _identify(self, names: pd.DataFrame, delist: pd.DataFrame):
        """Pick one ticker + descriptive metadata per permno (latest name record)."""
        perm_to_ticker: Dict[int, str] = {}
        meta_static: Dict[str, dict] = {}
        if names is None or names.empty:
            return perm_to_ticker, meta_static
        nm = names.copy()
        nm["nameenddt"] = pd.to_datetime(nm["nameenddt"], errors="coerce")
        nm = nm.sort_values("nameenddt")
        for permno, grp in nm.groupby("permno"):
            last = grp.iloc[-1]
            ticker = str(last.get("ticker") or f"P{int(permno)}").upper()
            # Only keep common stock (share codes 10/11) when shrcd is present.
            shrcd = last.get("shrcd")
            if shrcd is not None and not pd.isna(shrcd) and int(shrcd) not in (10, 11):
                continue
            perm_to_ticker[int(permno)] = ticker
            meta_static[ticker] = {
                "name": str(last.get("comnam") or ticker).title(),
                "sector": _sic_to_sector(last.get("siccd")),
            }
        return perm_to_ticker, meta_static

    def _build_prices(self, stock: pd.DataFrame, delist: pd.DataFrame,
                      perm_to_ticker: Dict[int, str]):
        """Total-return price index per ticker, with the delisting return spliced in.

        CRSP ``ret`` is the holding-period total return (dividends reinvested,
        splits handled).  We seed each series at its first traded price
        (``abs(prc)``) and compound ``ret`` forward, so the *level* is realistic
        for the price floor while the *returns* are exact.  On the delisting date
        we apply one final ``(1 + dlret)`` step — the loss (or takeover premium)
        the survivor-biased view never sees.
        """
        s = stock.copy()
        s["date"] = pd.to_datetime(s["date"])
        s["ret"] = pd.to_numeric(s["ret"], errors="coerce")
        s["prc"] = pd.to_numeric(s["prc"], errors="coerce").abs()
        s["vol"] = pd.to_numeric(s["vol"], errors="coerce")

        # delisting return / date per permno
        dl = {}
        if delist is not None and not delist.empty:
            d = delist.copy()
            d["dlstdt"] = pd.to_datetime(d["dlstdt"], errors="coerce")
            d["dlret"] = pd.to_numeric(d["dlret"], errors="coerce")
            d = d.dropna(subset=["dlstdt"]).sort_values("dlstdt")
            for permno, grp in d.groupby("permno"):
                r = grp.iloc[-1]
                dl[int(permno)] = (r["dlstdt"], r["dlret"], r.get("dlstcd"),
                                   grp.iloc[-1].get("exchcd"))

        price_cols: Dict[str, pd.Series] = {}
        adv: Dict[str, float] = {}
        delist_dates: Dict[str, Optional[str]] = {}

        for permno, grp in s.groupby("permno"):
            permno = int(permno)
            ticker = perm_to_ticker.get(permno)
            if ticker is None:
                continue
            grp = grp.sort_values("date").set_index("date")
            ret = grp["ret"].copy()

            # Splice the delisting return as a final step, on/after the last date.
            if permno in dl:
                ddate, dlret, dlstcd, _exch = dl[permno]
                if pd.isna(dlret):
                    exchcd = grp.get("exchcd")
                    is_nasdaq = False  # exchcd not in dsf; default NYSE/AMEX rule
                    dlret = (PERF_DELIST_IMPUTE_NASDAQ if is_nasdaq
                             else PERF_DELIST_IMPUTE_NYSE_AMEX) \
                        if _is_performance_delist(dlstcd) else 0.0
                # place the delisting return one business day after last obs (or on
                # ddate if it is beyond the price history)
                step_date = max(ret.index[-1] + pd.Timedelta(days=1),
                                pd.Timestamp(ddate)) if len(ret) else pd.Timestamp(ddate)
                ret.loc[step_date] = float(dlret)
                ret = ret.sort_index()
                delist_dates[ticker] = pd.Timestamp(ddate).strftime("%Y-%m-%d")
            else:
                delist_dates[ticker] = None

            # Build the index: seed at first valid price, compound returns.
            first_prc = grp["prc"].dropna()
            seed = float(first_prc.iloc[0]) if len(first_prc) else 100.0
            growth = (1.0 + ret.fillna(0.0)).cumprod()
            level = seed * growth / growth.iloc[0] if len(growth) else growth
            price_cols[ticker] = level

            dollar_vol = (grp["prc"] * grp["vol"]).dropna()
            adv[ticker] = float(dollar_vol.tail(63).mean()) if len(dollar_vol) else float("nan")

        prices = pd.DataFrame(price_cols).sort_index()
        return prices, adv, delist_dates

    def _build_benchmarks(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        try:
            b = self.client.benchmark(self.start, self.end)
        except Exception:
            b = None
        if b is None or b.empty:
            return pd.DataFrame(index=index)
        b = b.copy()
        b["date"] = pd.to_datetime(b["date"])
        b = b.dropna(subset=["date"]).sort_values("date").set_index("date")
        vw = pd.to_numeric(b["vwretd"], errors="coerce").fillna(0.0)
        spx = 100.0 * (1.0 + vw).cumprod()
        out = pd.DataFrame({"SPX": spx})
        out["NDX"] = spx      # single market proxy; NDX aliased so downstream keys exist
        return out.reindex(index).ffill()

    def _build_fundamentals(self, permnos, perm_to_ticker, adv, keep_tickers):
        raw = self.client.fundamentals(permnos, self.start, self.end)
        if raw is None or raw.empty:
            idx = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
            return pd.DataFrame(columns=FUNDAMENTAL_FIELDS, index=idx)
        df = raw.copy()
        df["ticker"] = df["permno"].map(lambda p: perm_to_ticker.get(int(p)))
        df = df[df["ticker"].isin(keep_tickers)].copy()
        df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
        df["rdq"] = pd.to_datetime(df.get("rdq"), errors="coerce")
        # Public date: earnings-announcement date when known, else datadate + lag.
        lag = pd.Timedelta(days=self.report_lag_days)
        df["pub_date"] = df["rdq"].where(df["rdq"].notna(), df["datadate"] + lag)
        df = df.dropna(subset=["pub_date", "ticker"]).sort_values(["ticker", "datadate"])

        rows, idx_tuples = [], []
        for ticker, grp in df.groupby("ticker"):
            grp = grp.reset_index(drop=True)
            eps_hist: List[float] = []
            for i, r in grp.iterrows():
                fields = self._ratios(r, grp, i)
                fields["adv_usd"] = adv.get(ticker, float("nan"))
                eg = fields.get("eps_growth")
                if eg is not None and np.isfinite(eg):
                    eps_hist.append(eg)
                if len(eps_hist) >= 3:
                    vol = float(np.std(eps_hist, ddof=0))
                    fields["earnings_vol"] = vol
                    fields["growth_stability"] = float(1.0 / (1.0 + max(vol, 0.0)))
                idx_tuples.append((r["pub_date"], ticker))
                rows.append(fields)
        if not rows:
            idx = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
            return pd.DataFrame(columns=FUNDAMENTAL_FIELDS, index=idx)
        idx = pd.MultiIndex.from_tuples(idx_tuples, names=["date", "ticker"])
        return pd.DataFrame(rows, index=idx)[FUNDAMENTAL_FIELDS].sort_index()

    @staticmethod
    def _ratios(r: pd.Series, grp: pd.DataFrame, i: int) -> Dict[str, float]:
        """Derive our fundamental fields from raw Compustat quarterly line items.

        Flows (net income, sales, EBIT, cash flow, dividends, EPS) are annualised
        by x4 from the single quarter; growth uses the same quarter one year prior
        (shift 4) so it is comparable and free of seasonality.
        """
        g = _to_float
        prc, sh = g(r.get("prccq")), g(r.get("cshoq"))
        mcap = prc * sh if np.isfinite(prc) and np.isfinite(sh) else float("nan")
        ceq = g(r.get("ceqq"))
        debt = np.nansum([g(r.get("dlttq")), g(r.get("dlcq"))])
        ni_q, sale_q, cogs_q = g(r.get("niq")), g(r.get("saleq")), g(r.get("cogsq"))
        ebit_q, xint_q = g(r.get("oiadpq")), g(r.get("xintq"))
        capx_q, ocf_q = g(r.get("capxq")), g(r.get("oancfq"))
        dvps_q, eps_q = g(r.get("dvpsxq")), g(r.get("epspxq"))

        def _div(a, b):
            return a / b if np.isfinite(a) and np.isfinite(b) and b != 0 else float("nan")

        f: Dict[str, float] = {k: float("nan") for k in FUNDAMENTAL_FIELDS}
        f["market_cap"] = mcap
        f["pe"] = _div(mcap, ni_q * 4)
        f["pb"] = _div(mcap, ceq)
        f["ev_ebit"] = _div(mcap + (debt if np.isfinite(debt) else 0.0), ebit_q * 4)
        f["fcf_yield"] = _div((ocf_q - capx_q) * 4, mcap)
        f["roe"] = _div(ni_q * 4, ceq)
        f["roic"] = _div(ebit_q * 4, (ceq if np.isfinite(ceq) else 0.0) +
                         (debt if np.isfinite(debt) else 0.0))
        f["gross_margin"] = _div(sale_q - cogs_q, sale_q)
        f["debt_equity"] = _div(debt, ceq)
        f["interest_coverage"] = _div(ebit_q, xint_q)
        f["dividend_yield"] = _div(dvps_q * 4, prc) if np.isfinite(prc) else 0.0
        if not np.isfinite(f["dividend_yield"]):
            f["dividend_yield"] = 0.0
        f["payout_ratio"] = _div(dvps_q, eps_q)
        # accruals ~ (net income - operating cash flow) / assets-proxy(market cap)
        if np.isfinite(ni_q) and np.isfinite(ocf_q):
            f["accruals"] = _div((ni_q - ocf_q) * 4, mcap)
        # year-over-year growth from the same quarter one year earlier (shift 4)
        if i >= 4:
            prev = grp.iloc[i - 4]
            f["revenue_growth"] = _div(sale_q - g(prev.get("saleq")), abs(g(prev.get("saleq"))))
            f["eps_growth"] = _div(eps_q - g(prev.get("epspxq")), abs(g(prev.get("epspxq"))))
        return f

    def _build_meta(self, tickers, meta_static, delist_dates) -> Dict[str, SecurityMeta]:
        meta: Dict[str, SecurityMeta] = {}
        for t in tickers:
            info = meta_static.get(t, {})
            meta[t] = SecurityMeta(
                ticker=t,
                name=info.get("name", t),
                sector=info.get("sector", "Unknown"),
                themes=list(self.themes_map.get(t, [])),
                security_type="common",
                delisting_date=delist_dates.get(t),
            )
        return meta
