#!/usr/bin/env python3
"""End-to-end demo of the Wharton ML Investment Engine.

Runs the full pipeline on deterministic synthetic data — no API keys required —
and walks through:

1. Building the fictional client profile from the questionnaire.
2. An initial portfolio construction (regime, style weights, risk, mandate).
3. A later decision date where the engine compares the held book to a fresh
   candidate and applies the trading-decision logic (rebalance vs. hold).

Outputs a console summary and writes CSVs to ``./engine_output`` for the
Excel / Google Sheets front-end.

Usage::

    python run_demo.py                # default 60-name, 8-year synthetic universe
    python run_demo.py --tickers 90 --years 9 --outdir engine_output
"""

from __future__ import annotations

import argparse

from wharton_ml_engine import SyntheticDataSource, run_engine
from wharton_ml_engine.client import profile_from_answers, sample_answers
from wharton_ml_engine.reporting import export_report, print_summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", type=int, default=60, help="synthetic universe size")
    ap.add_argument("--years", type=float, default=8.0, help="years of history")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="engine_output", help="CSV export directory")
    ap.add_argument("--no-backtest", action="store_true",
                    help="skip the template backtests (faster)")
    args = ap.parse_args()

    print("Loading synthetic universe "
          f"({args.tickers} names, {args.years:g}y history, seed {args.seed}) ...")
    bundle = SyntheticDataSource(n_tickers=args.tickers, years=args.years,
                                 seed=args.seed).load()

    # --- 1. Client profile from the questionnaire ---------------------------
    profile = profile_from_answers(sample_answers())
    print(f"Client: {profile.name}  |  objective={profile.objective}  |  "
          f"risk={profile.risk_label}  |  turnover tol={profile.turnover_tolerance}\n")

    dates = bundle.dates()
    # Decision date 1: ~6 months before the end of history.
    d1 = dates[-126]
    # Decision date 2: end of history.
    d2 = dates[-1]

    # --- 2. Initial construction -------------------------------------------
    print("### DECISION 1 — initial construction ###")
    report1 = run_engine(bundle, profile, as_of=d1,
                         run_backtest=not args.no_backtest)
    print_summary(report1)
    initial_book = report1.candidate_weights

    # --- 3. Later rebalance decision, holding the initial book -------------
    print("\n### DECISION 2 — six months later, holding the initial book ###")
    report2 = run_engine(bundle, profile, as_of=d2,
                         current_weights=initial_book,
                         run_backtest=not args.no_backtest)
    print_summary(report2)

    # --- 4. Export the latest decision for the spreadsheet front-end -------
    paths = export_report(report2, args.outdir)
    print(f"\nExported {len(paths)} files to '{args.outdir}/':")
    for name, path in sorted(paths.items()):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
