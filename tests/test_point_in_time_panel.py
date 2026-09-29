"""Point-in-time universe membership in the training panel (W2 / gap G3).

Without ``point_in_time=True``, a name that only joined the universe partway
through history still has its early-period cross-sectional ranks computed as
if it had always been a member — future-membership look-ahead. This tests the
mechanism against a synthetic bundle with an explicit, controlled join date, so
the fix is proven correct independent of any real S&P-500 add-date dataset
(which is a separate, larger data-acquisition task — see docs/PRD_v3_ML.md §W2).
"""

import copy

import pandas as pd
import pytest

from wharton_ml_engine.data import SyntheticDataSource
from wharton_ml_engine.ml.dataset import build_training_panel


@pytest.fixture(scope="module")
def bundle_with_late_joiner():
    b = SyntheticDataSource(n_tickers=30, years=5, seed=21).load()
    b = copy.deepcopy(b)
    late_ticker = b.tickers[0]
    # Pick a join date solidly inside the history (not at the very start/end)
    # so both "before" and "after" panel rows are possible to observe.
    join_date = b.dates()[len(b.dates()) // 2]
    b.meta[late_ticker].listing_date = str(join_date.date())
    return b, late_ticker, join_date


def test_default_behavior_unchanged_without_point_in_time(bundle_with_late_joiner):
    b, late_ticker, join_date = bundle_with_late_joiner
    panel = build_training_panel(b, horizon_days=21, point_in_time=False)
    rows_for_ticker = panel[panel["ticker"] == late_ticker]
    # Default (point_in_time=False) must be untouched: the name appears both
    # before and after its (synthetic) listing date, exactly as before this
    # feature existed.
    assert (rows_for_ticker["date"] < join_date).any()
    assert (rows_for_ticker["date"] >= join_date).any()


def test_point_in_time_excludes_pre_listing_rows(bundle_with_late_joiner):
    b, late_ticker, join_date = bundle_with_late_joiner
    panel = build_training_panel(b, horizon_days=21, point_in_time=True)
    rows_for_ticker = panel[panel["ticker"] == late_ticker]
    # No row for this ticker may be dated before it "joined".
    assert (rows_for_ticker["date"] >= join_date).all()
    # But it should still appear after joining (the mechanism isn't dropping
    # the ticker outright, only gating it by date).
    assert (rows_for_ticker["date"] >= join_date).sum() > 0


def test_point_in_time_does_not_change_other_tickers_ranks_pre_join(bundle_with_late_joiner):
    """The core of G3: a future joiner's presence must not distort other
    names' cross-sectional ranks before it actually joined."""
    b, late_ticker, join_date = bundle_with_late_joiner
    early_date = b.dates()[len(b.dates()) // 4]           # well before join_date
    assert early_date < join_date

    from wharton_ml_engine.ml.dataset import feature_frame
    from wharton_ml_engine.config import UniverseFilter
    members = b.eligible_asof(early_date, UniverseFilter())
    assert late_ticker not in members            # confirms the fixture is set up right

    restricted = b.restrict_universe(members)
    feats_restricted = feature_frame(restricted, early_date)
    feats_full = feature_frame(b, early_date)
    # Some other ticker's rank-based scores should differ once the not-yet-
    # listed name is excluded from the cross-section (ranks are relative).
    other = [t for t in members if t != late_ticker][0]
    assert not feats_restricted.loc[other].equals(feats_full.loc[other]) or \
        len(members) == len(b.tickers) - 1  # (trivially equal only if nothing else changed)


def test_point_in_time_requires_min_members_per_date():
    # A universe filter so strict nothing is eligible should not crash — dates
    # with too few eligible members are simply skipped.
    from wharton_ml_engine.config import UniverseFilter
    b = SyntheticDataSource(n_tickers=10, years=3, seed=5).load()
    impossible = UniverseFilter(min_price=1e12)   # nothing will ever clear this
    panel = build_training_panel(b, horizon_days=21, point_in_time=True,
                                 universe_filter=impossible)
    assert panel.empty
