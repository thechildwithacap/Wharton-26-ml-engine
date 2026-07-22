#!/usr/bin/env python3
"""Train the ML alpha model and save it for the engine to use.

The model learns, from history, how to combine the engine's signal scores into
an expected-return ranking.  Evaluation is walk-forward and no-look-ahead; the
headline metric is the Information Coefficient (rank correlation between the
model's score and realised forward returns), averaged over out-of-sample dates.

Usage::

    python train_model.py                       # ridge, 21-day horizon
    python train_model.py --task classification --horizon 42
    python train_model.py --tickers 90 --years 10 --out models/alpha_model.json
"""

from __future__ import annotations

import argparse
import warnings

from wharton_ml_engine import SyntheticDataSource
from wharton_ml_engine.ml import AlphaModel, build_training_panel, train_alpha_model


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["regression", "classification"],
                    default="regression")
    ap.add_argument("--horizon", type=int, default=21, help="forward-return horizon (days)")
    ap.add_argument("--source", default=None,
                    help="path to a cached real dataset dir (from fetch_dataset.py); "
                         "if omitted, uses the synthetic universe")
    ap.add_argument("--tickers", type=int, default=80)
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--splits", type=int, default=6, help="walk-forward folds")
    ap.add_argument("--alpha", type=float, default=10.0, help="ridge L2 strength")
    ap.add_argument("--extended", action="store_true",
                    help="train on the full SEC fundamental feature set (raw ranked "
                         "line items) plus the composite style scores")
    ap.add_argument("--out", default="models/alpha_model.json")
    args = ap.parse_args()

    if args.source:
        from wharton_ml_engine.data import CSVDataSource
        print(f"Loading cached dataset from '{args.source}' ...")
        bundle = CSVDataSource(args.source).load()
        print(f"  {bundle.prices.shape[1]} tickers, "
              f"{bundle.prices.index[0].date()} -> {bundle.prices.index[-1].date()}")
    else:
        print(f"Loading synthetic universe ({args.tickers} names, {args.years:g}y, "
              f"seed {args.seed}) ...")
        bundle = SyntheticDataSource(n_tickers=args.tickers, years=args.years,
                                     seed=args.seed).load()

    print(f"Building training panel (horizon {args.horizon}d"
          f"{', extended SEC features' if args.extended else ''}) ...")
    panel = build_training_panel(bundle, horizon_days=args.horizon, extended=args.extended)
    print(f"  {len(panel):,} samples across {panel['date'].nunique()} rebalance dates")

    print(f"Training {args.task} model with {args.splits}-fold walk-forward CV ...")
    hp = {"alpha": args.alpha} if args.task == "regression" else \
        {"l2": 1.0, "lr": 0.3, "epochs": 600}
    trained = train_alpha_model(bundle, task=args.task, horizon_days=args.horizon,
                                n_splits=args.splits, panel=panel,
                                extended=args.extended, **hp)

    m = trained.metrics
    print("\n" + "=" * 60)
    print(" OUT-OF-SAMPLE EVALUATION (walk-forward, no look-ahead)")
    print("=" * 60)
    print(f"  Mean IC (rank)      : {m.get('mean_ic', float('nan')):.4f}")
    print(f"  IC info ratio       : {m.get('ic_ir', float('nan')):.3f}")
    print(f"  IC t-stat           : {m.get('ic_t_stat', float('nan')):.2f}")
    print(f"  Hit rate (IC>0)     : {m.get('hit_rate', float('nan')):.1%}")
    print(f"  OOS periods         : {m.get('n_periods', 0)}")
    if args.task == "classification":
        print(f"  OOS accuracy        : {m.get('oos_accuracy', float('nan')):.1%}")
    else:
        print(f"  OOS R^2 (excess)    : {m.get('oos_r2', float('nan')):.4f}")

    print("\n Learned signal weights (largest magnitude first):")
    for feat, coef in trained.coefficients().items():
        print(f"    {feat:<12} {coef:+.4f}")

    path = trained.save(args.out)
    print(f"\nSaved model -> {path}")

    # Show the live ml_alpha ranking at the latest date.
    alpha = AlphaModel(trained)
    top = alpha.score(bundle).sort_values(ascending=False).head(8)
    print("\n Top ml_alpha names at the latest date:")
    for t, v in top.items():
        print(f"    {t:<5} {v:5.1f}")

    if m.get("ic_t_stat", 0) is not None and abs(m.get("ic_t_stat", 0)) < 2:
        print("\n  Note: |IC t-stat| < 2 — the signal is weak on this data; treat the "
              "ML alpha as one input among many, not a standalone strategy.")


if __name__ == "__main__":
    main()
