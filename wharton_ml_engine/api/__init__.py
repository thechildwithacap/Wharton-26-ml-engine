"""HTTP API layer for the engine UI (stdlib only, no web-framework dependency)."""

from .service import EngineService, report_to_dict

__all__ = ["EngineService", "report_to_dict"]
