"""Data ingestion layer (PRD section 4)."""

from .csv_source import CSVDataSource, load_bundle, save_bundle
from .financial_dataset import FinancialDatasetSource
from .source import (
    FUNDAMENTAL_FIELDS,
    DataBundle,
    DataSource,
    SecurityMeta,
)
from .synthetic import SyntheticDataSource

__all__ = [
    "FUNDAMENTAL_FIELDS",
    "DataBundle",
    "DataSource",
    "SecurityMeta",
    "SyntheticDataSource",
    "FinancialDatasetSource",
    "CSVDataSource",
    "save_bundle",
    "load_bundle",
]
