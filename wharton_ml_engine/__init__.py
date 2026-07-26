"""Multi-Model ML Investment Engine for the Wharton Global High School
Investment Competition.

An internal *decision-support* system (not a trading bot).  It analyses the
allowed US equity universe across multiple investment styles, applies strict
risk / mandate controls, adapts style weights to the market regime via daily
backtests, and produces transparent, auditable portfolio recommendations.

Layer map (see the PRD):

* ``data``      — ingestion & the pluggable data-source seam (section 4)
* ``client``    — questionnaire, profile, Client Fit (section 5)
* ``signals``   — fundamental / quant / hybrid pods (section 6.1)
* ``risk``      — stock & portfolio risk, mandate checks (section 6.2)
* ``backtest``  — templates, regime, style weights, stability (section 6.3)
* ``engine``    — integration, construction, trade logic (section 7)
* ``reporting`` — CSV exports & summaries (sections 7.3 & 10)

Quick start::

    from wharton_ml_engine import SyntheticDataSource, sample_profile, run_engine
    from wharton_ml_engine.reporting import print_summary

    bundle = SyntheticDataSource().load()
    report = run_engine(bundle, sample_profile())
    print_summary(report)
"""

from __future__ import annotations

from .client import (
    ClientProfile,
    compute_client_fit,
    profile_from_answers,
    sample_answers,
    sample_profile,
)
from .competition import SCORING_METRICS, CompetitionRules
from .config import CostModel, EngineConfig, PortfolioConstraints
from .data import (
    CRSPDataSource,
    CSVDataSource,
    DataBundle,
    DataSource,
    FinancialDatasetSource,
    SyntheticDataSource,
    WRDSClient,
)
from .engine import (
    ComparisonResult,
    EngineReport,
    backtest_ml_vs_rules,
    format_comparison,
    run_engine,
)
from .ml import AlphaModel, TrainedModel, train_alpha_model
from .signals import compute_signals

__version__ = "1.20.0"

__all__ = [
    "__version__",
    "SyntheticDataSource",
    "FinancialDatasetSource",
    "CRSPDataSource",
    "WRDSClient",
    "CSVDataSource",
    "DataSource",
    "DataBundle",
    "ClientProfile",
    "sample_profile",
    "sample_answers",
    "profile_from_answers",
    "compute_client_fit",
    "compute_signals",
    "EngineConfig",
    "PortfolioConstraints",
    "CostModel",
    "CompetitionRules",
    "SCORING_METRICS",
    "run_engine",
    "EngineReport",
    "train_alpha_model",
    "TrainedModel",
    "AlphaModel",
    "backtest_ml_vs_rules",
    "ComparisonResult",
    "format_comparison",
]
