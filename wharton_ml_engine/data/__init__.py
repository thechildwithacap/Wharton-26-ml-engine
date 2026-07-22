"""Data ingestion layer (PRD section 4)."""

from .csv_source import CSVDataSource, load_bundle, save_bundle
from .financial_dataset import FinancialDatasetSource
from .sec_edgar import build_sec_fundamentals
from .source import (
    FUNDAMENTAL_FIELDS,
    DataBundle,
    DataSource,
    SecurityMeta,
)
from .synthetic import SyntheticDataSource
from .web_source import WebDataSource

__all__ = [
    "FUNDAMENTAL_FIELDS",
    "DataBundle",
    "DataSource",
    "SecurityMeta",
    "SyntheticDataSource",
    "FinancialDatasetSource",
    "WebDataSource",
    "build_sec_fundamentals",
    "CSVDataSource",
    "save_bundle",
    "load_bundle",
]
