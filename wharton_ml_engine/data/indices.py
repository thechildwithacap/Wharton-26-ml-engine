"""Named index universes so the engine can run on more than the S&P 500.

The engine scores whatever universe it is handed.  This module supplies the
*membership* for the major US indices as curated constituent lists, plus a
registry and helpers that intersect a list with whatever names are actually
present in the loaded :class:`DataBundle` — so picking "Nasdaq-100" restricts
the run to the NDX names available in the data, "Dow 30" to the DJIA names, and
so on.  The UI's index selector is backed directly by this registry.

These are constituent *lists*, not point-in-time membership histories (that is
what :mod:`wharton_ml_engine.data.sp500` adds for survivorship work).  For live
decision-support on the current universe they are exactly what's needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# --- Dow Jones Industrial Average (30 names) ------------------------------
DJIA: List[str] = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GS",
    "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT",
    "NKE", "PG", "TRV", "UNH", "V", "VZ", "WMT", "DOW", "INTC", "AMZN",
]

# --- Nasdaq-100 (growth / tech heavy; representative constituent list) ----
NDX: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "TSLA",
    "COST", "NFLX", "ADBE", "PEP", "AMD", "CSCO", "TMUS", "INTC", "CMCSA",
    "QCOM", "INTU", "TXN", "AMGN", "HON", "AMAT", "ISRG", "BKNG", "ADP",
    "VRTX", "GILD", "ADI", "REGN", "MU", "LRCX", "PANW", "SBUX", "MDLZ",
    "KLAC", "SNPS", "CDNS", "MAR", "ORLY", "CSX", "ASML", "ABNB", "FTNT",
    "NXPI", "PYPL", "MNST", "PCAR", "MRVL", "WDAY", "CRWD", "CTAS", "ROP",
]

# --- Philadelphia Semiconductor (SOX) ------------------------------------
SOX: List[str] = [
    "NVDA", "AVGO", "AMD", "QCOM", "TXN", "INTC", "AMAT", "MU", "LRCX",
    "KLAC", "NXPI", "ADI", "MRVL", "MCHP", "ON", "MPWR", "TER", "SWKS",
    "ASML", "TSM",
]

# --- S&P 400 MidCap (representative constituent list) ---------------------
# Mid caps matter for value work: the large-cap universe is where factor
# premia are most arbitraged away, and Graham-style screens rarely trigger.
MID: List[str] = [
    "AAON", "ATR", "BCO", "CBT", "CHE", "CW", "DCI", "EXP", "EXPO", "FIX",
    "HLI", "ICUI", "IDA", "JBT", "KBH", "LANC", "LSTR", "MAN", "MSA", "MTX",
    "NDSN", "NJR", "OGE", "OLED", "PBH", "POR", "RGLD", "RLI", "RS", "SEIC",
    "SIGI", "SKX", "SLGN", "SON", "SR", "TCBI", "THG", "THO", "TREX", "TTC",
    "UFPI", "VMI", "WERN", "WOR", "WSO", "WTS", "CROX", "GEF", "MATX", "HNI",
]

# --- S&P 600 SmallCap (representative constituent list) -------------------
SMALL: List[str] = [
    "ABCB", "ABM", "ACIW", "AEIS", "AGYS", "AIN", "AIR", "ALG", "AMN", "ANDE",
    "APAM", "APOG", "ARCB", "ASB", "ASGN", "ATGE", "AWR", "AZZ", "BANF", "BANR",
    "BCPC", "BGS", "BHE", "BJRI", "BKE", "BLMN", "BMI", "BOH", "BRC", "CAKE",
    "CALM", "CATY", "CBRL", "CBU", "CENT", "CHCO", "CNK", "CNMD", "CNS", "COHU",
    "COLB", "CPK", "CRS", "CSGS", "CTS", "CVBF", "CWT", "DIOD", "DLX", "DORM",
    "EBF", "EGBN", "ENR", "ENS", "EPAC", "ESE", "FELE", "FFBC", "FIZZ", "FORM",
    "FSS", "FUL", "GBX", "GFF", "GIII", "GMS", "GPI", "GRBK", "GVA", "HAE",
    "HCSG", "HELE", "HI", "HMN", "HTLD", "HUBG", "HWKN", "IBP", "ICFI", "INDB",
    "IOSP", "IPAR", "ITRI", "JBSS", "JJSF", "KAI", "KELYA", "KFY", "KLIC", "KMT",
    "KWR", "LCII", "LGND", "LKFN", "LNN", "LRN", "LZB", "MATW", "MCRI", "MCY",
    "MGEE", "MLI", "MMSI", "MNRO", "MRTN", "MTRN", "MYE", "NBTB", "NHC", "NPO",
    "NSIT", "NWBI", "NWE", "NWN", "NX", "OII", "OMCL", "OSIS", "OTTR", "PATK",
    "PDCO", "PFS", "PLAB", "PLUS", "PLXS", "POWI", "PRDO", "PRGS", "PRK", "PRLB",
    "PSMT", "PTEN", "PZZA", "RDN", "RGR", "ROCK", "ROG", "SANM", "SBCF", "SCL",
    "SCSC", "SCVL", "SFNC", "SHOO", "SJW", "SKT", "SKYW", "SMTC", "SPSC", "SSD",
    "STBA", "STC", "STRA", "SXI", "SXT", "TCBK", "TILE", "TMP", "TNC", "TPH",
    "TRMK", "TRN", "TTMI", "UNF", "USNA", "USPH", "UVV", "VECO", "VICR", "VRTS",
    "WABC", "WDFC", "WGO", "WMK", "WNC", "WSFS", "WTS", "YELP", "ZD",
]

# --- Sector composites (GICS-style) --------------------------------------
# Built at runtime from the bundle's sector metadata; listed here so the
# registry can advertise them.  Empty ticker list => "use sector filter".
_SECTOR_INDICES: Dict[str, str] = {
    "SEC_TECH": "Technology",
    "SEC_FIN": "Financials",
    "SEC_HEALTH": "Health Care",
    "SEC_ENERGY": "Energy",
    "SEC_STAPLES": "Consumer Staples",
    "SEC_DISC": "Consumer Discretionary",
}


@dataclass
class IndexInfo:
    id: str
    name: str
    n_members: int          # curated list size (or sector, runtime)
    n_available: int        # how many are present in the loaded data
    benchmark: str          # benchmark column to compare against


# id -> (display name, curated tickers or None for "whole bundle", benchmark)
INDEX_REGISTRY: Dict[str, Dict[str, object]] = {
    "SPX":  {"name": "S&P 500 (large-cap broad)", "tickers": None, "benchmark": "SPX"},
    "NDX":  {"name": "Nasdaq-100 (growth / tech)", "tickers": NDX, "benchmark": "NDX"},
    "DJIA": {"name": "Dow Jones Industrial Average (30)", "tickers": DJIA, "benchmark": "SPX"},
    "SOX":  {"name": "PHLX Semiconductor", "tickers": SOX, "benchmark": "NDX"},
    "MID":  {"name": "S&P 400 MidCap", "tickers": MID, "benchmark": "SPX"},
    "SMALL": {"name": "S&P 600 SmallCap", "tickers": SMALL, "benchmark": "SPX"},
    "SMID": {"name": "Small + Mid cap combined", "tickers": SMALL + MID,
             "benchmark": "SPX"},
}


def _sector_members(sector: str, bundle_sectors: Dict[str, str]) -> List[str]:
    return [t for t, s in bundle_sectors.items() if s == sector]


def index_members(index_id: str, bundle_sectors: Dict[str, str]) -> Optional[List[str]]:
    """Curated membership for ``index_id``.

    Returns the ticker list, a sector-derived list, or ``None`` for SPX which
    means "the whole loaded bundle" (already a large-cap universe).
    """
    if index_id in INDEX_REGISTRY:
        return INDEX_REGISTRY[index_id]["tickers"]        # type: ignore[return-value]
    if index_id in _SECTOR_INDICES:
        return _sector_members(_SECTOR_INDICES[index_id], bundle_sectors)
    raise KeyError(f"unknown index: {index_id!r}")


def available_universe(index_id: str, bundle_tickers: List[str],
                       bundle_sectors: Dict[str, str]) -> List[str]:
    """The index's constituents that are actually present in the loaded data,
    in the bundle's order.  SPX (None membership) => every name in the bundle."""
    members = index_members(index_id, bundle_sectors)
    if members is None:
        return list(bundle_tickers)
    want = set(members)
    return [t for t in bundle_tickers if t in want]


def benchmark_for(index_id: str) -> str:
    if index_id in INDEX_REGISTRY:
        return str(INDEX_REGISTRY[index_id]["benchmark"])
    return "SPX"


def list_indices(bundle_tickers: List[str],
                 bundle_sectors: Dict[str, str]) -> List[IndexInfo]:
    """Registry + sector indices, each annotated with how many names are
    available in the loaded data (so the UI can grey out empty ones)."""
    out: List[IndexInfo] = []
    for iid, meta in INDEX_REGISTRY.items():
        avail = available_universe(iid, bundle_tickers, bundle_sectors)
        members = meta["tickers"]
        n_members = len(bundle_tickers) if members is None else len(members)  # type: ignore[arg-type]
        out.append(IndexInfo(iid, str(meta["name"]), n_members, len(avail),
                             str(meta["benchmark"])))
    for iid, sector in _SECTOR_INDICES.items():
        members = _sector_members(sector, bundle_sectors)
        if members:
            out.append(IndexInfo(iid, f"{sector} sector", len(members),
                                 len(members), "SPX"))
    return out
