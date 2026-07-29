"""Merge cached :class:`DataBundle` datasets into one wider universe.

Fetching is rate-limited by free price providers, so a broad universe is built
in batches (large caps first, then mid, then small) and cached separately.  This
joins those batches into a single bundle: prices are aligned on the union of
trading dates, fundamentals panels are concatenated, and metadata is merged.

Names present in more than one batch keep the first bundle's data (batches are
passed most-authoritative first).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

from .source import DataBundle, SecurityMeta


def merge_bundles(bundles: Sequence[DataBundle],
                  how: str = "outer") -> DataBundle:
    """Combine bundles into one.  Earlier bundles win on duplicate tickers.

    ``how`` controls the date index join: ``"outer"`` keeps every date any
    bundle saw (missing values forward-filled per column), ``"inner"`` keeps
    only dates common to all — use ``inner`` when strict alignment matters more
    than history length.
    """
    bundles = [b for b in bundles if b is not None and not b.prices.empty]
    if not bundles:
        raise ValueError("no non-empty bundles to merge")
    if len(bundles) == 1:
        return bundles[0]

    # --- prices: union of dates, first-bundle-wins on duplicate tickers ---
    seen: set = set()
    price_parts: List[pd.DataFrame] = []
    for b in bundles:
        keep = [t for t in b.prices.columns if t not in seen]
        seen.update(keep)
        if keep:
            price_parts.append(b.prices[keep])

    index = price_parts[0].index
    for p in price_parts[1:]:
        index = index.union(p.index) if how == "outer" else index.intersection(p.index)
    index = index.sort_values()

    prices = pd.concat([p.reindex(index) for p in price_parts], axis=1)
    # Forward-fill gaps introduced by differing calendars, but never invent a
    # price before a name's first real observation.
    prices = prices.ffill().where(prices.ffill().notna() & _after_first_valid(prices))

    # --- benchmarks: first bundle that has them wins, reindexed ---
    benchmarks = pd.DataFrame(index=index)
    for b in bundles:
        if not b.benchmarks.empty:
            benchmarks = b.benchmarks.reindex(index).ffill()
            break

    # --- fundamentals: concatenate, drop duplicate (date, ticker) rows ---
    fund_parts = [b.fundamentals for b in bundles if not b.fundamentals.empty]
    if fund_parts:
        fundamentals = pd.concat(fund_parts)
        fundamentals = fundamentals[~fundamentals.index.duplicated(keep="first")]
        fundamentals = fundamentals.sort_index()
    else:
        fundamentals = bundles[0].fundamentals

    # --- meta: first bundle wins ---
    meta: Dict[str, SecurityMeta] = {}
    for b in bundles:
        for t, m in b.meta.items():
            if t not in meta and t in prices.columns:
                meta[t] = m

    keep = [t for t in prices.columns if t in meta]
    return DataBundle(prices=prices[keep], benchmarks=benchmarks,
                      fundamentals=fundamentals, meta=meta)


def _after_first_valid(prices: pd.DataFrame) -> pd.DataFrame:
    """Boolean mask that is True only at/after each column's first real price."""
    mask = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    for c in prices.columns:
        first = prices[c].first_valid_index()
        if first is None:
            mask[c] = False
        else:
            mask.loc[mask.index < first, c] = False
    return mask
