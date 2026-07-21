import pytest

from wharton_ml_engine import SyntheticDataSource, sample_profile


@pytest.fixture(scope="session")
def bundle():
    # Small, fast, deterministic universe for tests.
    return SyntheticDataSource(n_tickers=40, years=5, seed=7).load()


@pytest.fixture(scope="session")
def profile():
    return sample_profile()


@pytest.fixture(scope="session")
def as_of(bundle):
    return bundle.dates()[-1]
