#!/usr/bin/env python3
"""Paper-trade a simulated portfolio with the engine's recommendations.

Maintains a persistent virtual book (JSON) and advances it one "trading day" at
a time — the same workflow as the Wharton simulator. Mark to market, ask the
engine, act on its decision, record the mark.

Typical use::

    # 1) start a $100k paper book
    python paper_trade.py --source datasets/us_sample --init --capital 100000

    # 2) run a step at a given date (repeat as new data arrives; omit --date for latest)
    python paper_trade.py --source datasets/us_sample --date 2025-06-30
    python paper_trade.py --source datasets/us_sample                 # latest date

    # 3) see the track record
    python paper_trade.py --source datasets/us_sample --report

Backfill a whole track record from cached data with --walk (weekly/monthly steps).
"""

from __future__ import annotations

import argparse
import json
import warnings

from wharton_ml_engine.client import sample_profile
from wharton_ml_engine.data import CSVDataSource, SyntheticDataSource
from wharton_ml_engine.engine import (
    PaperPortfolio,
    paper_trade_step,
    performance_summary,
)


def _load_bundle(args):
    if args.source:
        return CSVDataSource(args.source).load()
    return SyntheticDataSource(n_tickers=args.tickers, years=args.years, seed=args.seed).load()


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=None, help="cached dataset dir (else synthetic)")
    ap.add_argument("--tickers", type=int, default=60)
    ap.add_argument("--years", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--state", default="paper_state.json", help="portfolio state file")
    ap.add_argument("--init", action="store_true", help="create a fresh paper book")
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--date", default=None, help="run one step as of this date (YYYY-MM-DD)")
    ap.add_argument("--walk", choices=["weekly", "monthly"], default=None,
                    help="backfill a track record by stepping through history")
    ap.add_argument("--report", action="store_true", help="print the track record and exit")
    ap.add_argument("--force", action="store_true", help="rebalance even if the engine says hold")
    args = ap.parse_args()

    bundle = _load_bundle(args)
    profile = sample_profile()

    import os
    if args.init or not os.path.exists(args.state):
        pf = PaperPortfolio.new(capital=args.capital)
        pf.save(args.state)
        print(f"Initialised paper book: ${args.capital:,.0f} cash -> {args.state}")
        if args.init and not (args.date or args.walk):
            return
    else:
        pf = PaperPortfolio.load(args.state)

    if args.report:
        _print_report(pf)
        return

    if args.walk:
        idx = bundle.dates()
        freq = "W-FRI" if args.walk == "weekly" else "BME"
        import pandas as pd
        grid = pd.bdate_range(idx[252], idx[-1], freq=freq)
        steps = [d for d in grid if d in set(idx)] or [bundle.dates()[-1]]
        print(f"Walking {len(steps)} {args.walk} steps ...")
        for d in steps:
            r = paper_trade_step(bundle, profile, pf, as_of=d, force_rebalance=args.force)
            print("  " + r.summary())
        pf.save(args.state)
    else:
        r = paper_trade_step(bundle, profile, pf, as_of=args.date, force_rebalance=args.force)
        pf.save(args.state)
        print(r.summary())
        for reason in r.decision_reasons:
            print("    · " + reason)

    print()
    _print_report(pf)


def _print_report(pf: PaperPortfolio) -> None:
    s = performance_summary(pf)
    if s.get("status"):
        print(s["status"])
        return
    print("=" * 56)
    print(" PAPER PORTFOLIO — track record")
    print("=" * 56)
    print(f"  As of {s['asof']}  ({s['marks']} marks since {s['start']})")
    print(f"  NAV               ${s['nav']:,.0f}")
    print(f"  Total return      {s['total_return']:+.2%}")
    if "benchmark_return" in s:
        print(f"  Benchmark return  {s['benchmark_return']:+.2%}")
        print(f"  Excess vs bench   {s['excess_return']:+.2%}")
    if "max_drawdown" in s:
        print(f"  Max drawdown      {s['max_drawdown']:+.2%}")
    if pf.positions:
        print("  Top holdings (shares):")
        for t, sh in sorted(pf.positions.items(), key=lambda kv: -kv[1] * 1)[:8]:
            print(f"    {t:<6} {sh:,.1f}")


if __name__ == "__main__":
    main()
