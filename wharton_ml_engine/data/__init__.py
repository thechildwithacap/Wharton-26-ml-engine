"""Data ingestion layer (PRD section 4)."""

from .crsp import CRSPClient, CRSPDataSource, WRDSClient
from .csv_source import CSVDataSource, load_bundle, save_bundle
from .merge import merge_bundles
from .indices import (
    INDEX_REGISTRY,
    IndexInfo,
    available_universe,
    benchmark_for,
    list_indices,
)
from .financial_dataset import FinancialDatasetSource
from .sec_edgar import build_sec_fundamentals
from .sp500 import (
    SP500_REMOVALS,
    SurvivorshipEstimate,
    estimate_survivorship_bias,
    sp500_pit_metadata,
)
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
    "CRSPDataSource",
    "CRSPClient",
    "WRDSClient",
    "WebDataSource",
    "build_sec_fundamentals",
    "SP500_REMOVALS",
    "SurvivorshipEstimate",
    "estimate_survivorship_bias",
    "sp500_pit_metadata",
    "CSVDataSource",
    "save_bundle",
    "load_bundle",
    "INDEX_REGISTRY",
    "IndexInfo",
    "available_universe",
    "benchmark_for",
    "list_indices",
    "merge_bundles",
]
