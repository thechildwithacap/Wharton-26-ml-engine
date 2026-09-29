"""Feature registry: every training-panel column has a spec, with a sensible
family/sign/rationale, and the registry stays exhaustive as features are added.
"""

import pytest

from wharton_ml_engine.ml.dataset import EXTENDED_FEATURES, ML_FEATURES
from wharton_ml_engine.ml.features import (
    FAMILIES,
    REGISTRY,
    FeatureSpec,
    family_of,
    proxy_features,
    spec,
)


def test_every_ml_feature_is_registered():
    missing = [f for f in ML_FEATURES if f not in REGISTRY]
    assert missing == [], f"unregistered ML_FEATURES: {missing}"


def test_every_extended_feature_is_registered():
    missing = [f for f in EXTENDED_FEATURES if f not in REGISTRY]
    assert missing == [], f"unregistered EXTENDED_FEATURES: {missing}"


def test_registry_has_no_orphans():
    # Every registered spec should correspond to a real feature column, so the
    # registry can't silently drift from the code that actually produces panels.
    orphans = [n for n in REGISTRY if n not in EXTENDED_FEATURES]
    assert orphans == [], f"registry entries with no matching feature column: {orphans}"


def test_specs_have_content():
    for name, s in REGISTRY.items():
        assert s.family in FAMILIES
        assert s.expected_sign in (-1, 0, 1)
        assert len(s.description) > 10
        assert len(s.rationale) > 10
        assert s.source in ("sec", "prices", "macro", "derived")


def test_spec_lookup_raises_helpfully():
    with pytest.raises(KeyError, match="not_a_real_feature"):
        spec("not_a_real_feature")


def test_family_of_and_proxy_features():
    fam = family_of(ML_FEATURES)
    assert fam["value"] == "value"
    assert fam["analyst"] == "analyst_proxy"
    proxies = proxy_features(ML_FEATURES)
    assert "analyst" in proxies
    assert "value" not in proxies


def test_invalid_family_rejected():
    with pytest.raises(ValueError, match="unknown family"):
        FeatureSpec(name="x", family="not_a_family", description="d" * 20,
                   expected_sign=1, rationale="r" * 20, source="derived",
                   lag_days=0)


def test_invalid_sign_rejected():
    with pytest.raises(ValueError, match="expected_sign"):
        FeatureSpec(name="x", family="value", description="d" * 20,
                   expected_sign=2, rationale="r" * 20, source="derived",
                   lag_days=0)


def test_size_and_issuance_carry_institutional_memory():
    # These two features have specific, hard-won findings from this project's
    # audit history; the registry must not let that context silently vanish.
    assert "t=5.14" in REGISTRY["size"].rationale or "split" in REGISTRY["size"].rationale
    assert "issuance" in REGISTRY["f_net_issuance"].rationale.lower()


def test_feature_columns_accepts_named_sets():
    from wharton_ml_engine.ml.dataset import feature_columns
    assert feature_columns("base") == feature_columns(False) == ML_FEATURES
    assert feature_columns("extended") == feature_columns(True) == EXTENDED_FEATURES
    with pytest.raises(ValueError):
        feature_columns("bogus")
