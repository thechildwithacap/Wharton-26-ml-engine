#!/usr/bin/env python3
"""Backtest the ML-driven portfolio vs the rule-only engine (and benchmark).

Trains the ML alpha model on the first part of the history, then runs BOTH
engines — rule-only and ML-augmented — forward on the held-out window and
compares net-of-cost performance.  The ML model never sees test-window returns,
so any difference is an honest out-of-sample read on whether the learned alpha
adds value.

Usage::

    python backtest_compare.py                         # synthetic, 60/40 split
    python backtest_compare.py --source datasets/us_sample --split 0.6
    python backtest_compare.py --tickers 80 --years 10 --out engine_output
"""

from __future__ import annotations

import argparse
import warnings

from wharton_ml_engine import SyntheticDataSource, sample_profile
from wharton_ml_engine.engine import backtest_ml_vs_rules, format_comparison


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=None,
                    help="cached real dataset dir (from fetch_dataset.py); "
                         "omit to use the synthetic universe")
    ap.add_argument("--tickers", type=int, default=70)
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--split", type=float, default=0.6,
                    help="fraction of history used to train the ML model (rest is tested)")
    ap.add_argument("--horizon", type=int, default=21, help="ML forward-return horizon")
    ap.add_argument("--retrain-every", type=int, default=None,
                    help="walk-forward: retrain the ML model every N rebalances "
                         "(months); omit for a single fit at the split")
    ap.add_argument("--train-lookback-days", type=int, default=None,
                    help="walk-forward: use a rolling training window of this many "
                         "days (default: expanding)")
    ap.add_argument("--out", default=None, help="optional dir to write equity-curve CSV")
    args = ap.parse_args()

    if args.source:
        from wharton_ml_engine.data import CSVDataSource
        print(f"Loading cached dataset from '{args.source}' ...")
        bundle = CSVDataSource(args.source).load()
    else:
        print(f"Loading synthetic universe ({args.tickers} names, {args.years:g}y, "
              f"seed {args.seed}) ...")
        bundle = SyntheticDataSource(n_tickers=args.tickers, years=args.years,
                                     seed=args.seed).load()

    dates = bundle.dates()
    train_end = dates[int(len(dates) * args.split)]
    print(f"Training ML on data through {train_end.date()}, "
          f"testing on {train_end.date()} .. {dates[-1].date()} ...\n")

    if args.retrain_every:
        print(f"Walk-forward: retraining every {args.retrain_every} rebalances "
              f"({'rolling ' + str(args.train_lookback_days) + 'd' if args.train_lookback_days else 'expanding'} window)\n")

    result = backtest_ml_vs_rules(bundle, sample_profile(), train_end=train_end,
                                  ml_horizon=args.horizon,
                                  retrain_every=args.retrain_every,
                                  train_lookback_days=args.train_lookback_days)
    print(format_comparison(result))

    if args.out:
        import os
        os.makedirs(args.out, exist_ok=True)
        curves_path = os.path.join(args.out, "ml_vs_rules_equity.csv")
        result.equity_curves().to_csv(curves_path)
        summary_path = os.path.join(args.out, "ml_vs_rules_summary.csv")
        result.summary.to_csv(summary_path)
        print(f"\nWrote {curves_path}\n      {summary_path}")


if __name__ == "__main__":
    main()
