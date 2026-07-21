#!/usr/bin/env python3
"""Fetch a real US-equity dataset from financialdatasets.ai and cache it.

Downloads daily prices, historical financial metrics and company facts for a
chosen universe, assembles a DataBundle, and caches it as CSVs so training and
backtests can run offline afterwards.

Requires an API key::

    export FINANCIAL_DATASETS_API_KEY=sk-...
    python fetch_dataset.py --tickers AAPL,MSFT,NVDA,JPM,XOM --years 6 --out datasets/us_sample

Then train on it::

    python train_model.py --source datasets/us_sample
"""

from __future__ import annotations

import argparse
import datetime as dt

from wharton_ml_engine.data import FinancialDatasetSource, save_bundle

# A reasonable, liquid, sector-diverse default universe.
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",      # tech / comm
    "JPM", "BAC", "V",                                     # financials
    "UNH", "JNJ", "PFE",                                   # health care
    "XOM", "CVX",                                          # energy
    "PG", "KO", "WMT",                                     # staples
    "HD", "CAT", "BA",                                     # discretionary / industrials
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", default=",".join(DEFAULT_UNIVERSE),
                    help="comma-separated tickers")
    ap.add_argument("--years", type=float, default=6.0)
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--out", default="datasets/us_sample")
    args = ap.parse_args()

    end = dt.date.fromisoformat(args.end)
    start = end - dt.timedelta(days=int(args.years * 365.25))
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    print(f"Fetching {len(tickers)} tickers {start} -> {end} from financialdatasets.ai ...")
    src = FinancialDatasetSource(tickers=tickers, start=start.isoformat(),
                                 end=end.isoformat())
    bundle = src.load()
    print(f"  prices:       {bundle.prices.shape[0]} days x {bundle.prices.shape[1]} tickers")
    print(f"  benchmarks:   {list(bundle.benchmarks.columns)}")
    print(f"  fundamentals: {bundle.fundamentals.shape[0]} snapshots")
    print(f"  sectors:      {sorted(set(bundle.sectors.values()))}")

    paths = save_bundle(bundle, args.out)
    print(f"\nCached dataset to '{args.out}/':")
    for name, path in paths.items():
        print(f"  - {path}")
    print("\nNext:  python train_model.py --source", args.out)


if __name__ == "__main__":
    main()
