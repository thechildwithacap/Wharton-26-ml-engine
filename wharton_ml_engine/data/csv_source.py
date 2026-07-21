"""CSV-backed data source and bundle (de)serialisation.

Lets any real dataset — the cached output of a live API pull, a broker export,
or the Wharton approved-list file — be loaded into a :class:`DataBundle`, and
lets a bundle be cached to disk so training/backtests run offline afterwards.

Layout of a dataset directory::

    prices.csv        index=date, columns=tickers (adjusted close)
    benchmarks.csv    index=date, columns=SPX,NDX,ETF_*
    fundamentals.csv  long: date,ticker,<fundamental fields>
    meta.csv          ticker,name,sector,themes,pays_dividend
"""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from .source import FUNDAMENTAL_FIELDS, DataBundle, DataSource, SecurityMeta


def save_bundle(bundle: DataBundle, dirpath: str) -> Dict[str, str]:
    """Write a bundle to ``dirpath`` as CSVs.  Returns the paths written."""
    os.makedirs(dirpath, exist_ok=True)
    paths: Dict[str, str] = {}

    p = os.path.join(dirpath, "prices.csv")
    bundle.prices.to_csv(p, index_label="date")
    paths["prices"] = p

    b = os.path.join(dirpath, "benchmarks.csv")
    bundle.benchmarks.to_csv(b, index_label="date")
    paths["benchmarks"] = b

    f = os.path.join(dirpath, "fundamentals.csv")
    bundle.fundamentals.reset_index().to_csv(f, index=False)
    paths["fundamentals"] = f

    m = os.path.join(dirpath, "meta.csv")
    rows = [
        {
            "ticker": t,
            "name": sm.name,
            "sector": sm.sector,
            "themes": "|".join(sm.themes),
            "pays_dividend": int(sm.pays_dividend),
        }
        for t, sm in bundle.meta.items()
    ]
    pd.DataFrame(rows).to_csv(m, index=False)
    paths["meta"] = m
    return paths


def load_bundle(dirpath: str) -> DataBundle:
    prices = pd.read_csv(os.path.join(dirpath, "prices.csv"),
                         index_col="date", parse_dates=["date"])
    benchmarks = pd.read_csv(os.path.join(dirpath, "benchmarks.csv"),
                             index_col="date", parse_dates=["date"])

    fund = pd.read_csv(os.path.join(dirpath, "fundamentals.csv"), parse_dates=["date"])
    fund = fund.set_index(["date", "ticker"]).sort_index()
    for col in FUNDAMENTAL_FIELDS:
        if col not in fund.columns:
            fund[col] = float("nan")
    fund = fund[FUNDAMENTAL_FIELDS]

    meta_df = pd.read_csv(os.path.join(dirpath, "meta.csv"))
    meta: Dict[str, SecurityMeta] = {}
    for _, r in meta_df.iterrows():
        themes: List[str] = str(r.get("themes", "")).split("|") if str(r.get("themes", "")) else []
        themes = [t for t in themes if t and t != "nan"]
        meta[str(r["ticker"])] = SecurityMeta(
            ticker=str(r["ticker"]),
            name=str(r.get("name", r["ticker"])),
            sector=str(r.get("sector", "Unknown")),
            themes=themes,
            pays_dividend=bool(int(r.get("pays_dividend", 0))),
        )
    # Keep only tickers present in prices, in price-column order.
    keep = [t for t in prices.columns if t in meta]
    meta = {t: meta[t] for t in keep}
    return DataBundle(prices=prices[keep], benchmarks=benchmarks,
                      fundamentals=fund, meta=meta)


class CSVDataSource(DataSource):
    """Load a :class:`DataBundle` from a dataset directory of CSVs."""

    def __init__(self, dirpath: str) -> None:
        self.dirpath = dirpath

    def load(self) -> DataBundle:
        return load_bundle(self.dirpath)
