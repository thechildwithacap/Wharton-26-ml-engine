"""Data ingestion layer (PRD section 4)."""

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
]
