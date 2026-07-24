"""Income is dropped from the alpha blend but retained for the client mandate."""

from wharton_ml_engine import SyntheticDataSource, sample_profile
from wharton_ml_engine.client import compute_client_fit
from wharton_ml_engine.client.profile import ClientProfile
from wharton_ml_engine.config import STYLES
from wharton_ml_engine.ml import EXTENDED_FEATURES, ML_FEATURES
from wharton_ml_engine.signals import compute_signals


def test_income_not_in_alpha_paths():
    assert "income" not in STYLES                       # style blend
    assert "income" not in ML_FEATURES                  # ML alpha
    assert "f_dividend_yield" not in EXTENDED_FEATURES   # extended ML
    assert "f_payout_ratio" not in EXTENDED_FEATURES


def test_income_still_computed_for_reporting():
    b = SyntheticDataSource(n_tickers=30, years=5, seed=2).load()
    sig = compute_signals(b, sample_profile())
    assert "income" in sig.columns                      # still visible


def test_income_mandate_lives_in_client_fit():
    # An income-objective client still rewards dividend payers via Client Fit,
    # independent of the (removed) income alpha style.
    b = SyntheticDataSource(n_tickers=40, years=5, seed=5).load()
    income_client = ClientProfile(objective="income", risk_tolerance=3)
    fit = compute_client_fit(b, income_client)
    div = b.fundamentals_asof(b.dates()[-1])["dividend_yield"].reindex(fit.index)
    payers = fit.loc[div > div.median(), "client_fit"].mean()
    nonpayers = fit.loc[div <= div.median(), "client_fit"].mean()
    assert payers > nonpayers                            # dividends still rewarded
