"""Download market data and convert it into LEAN's on-disk format.

Three layers, deliberately kept apart:

``leandata.model``
    The canonical intermediate representation. Knows nothing about vendors
    or about LEAN.
``leandata.sources``
    One adapter per vendor, each turning that vendor's download into the
    canonical form. Adding a source is a new module plus one ``register``
    call.
``leandata.lean``
    Turns the canonical form into the bytes LEAN reads. Never imports a
    source.
"""

from .errors import (
    EmptyHistoryError,
    LeanDataError,
    SourceError,
    UnsupportedResolutionError,
    ValidationError,
)
from .model import (
    BAR_COLUMNS,
    EXCHANGE_TIMEZONE,
    Provenance,
    Resolution,
    SecurityHistory,
    SymbolSpec,
)
from .sources import DataSource, FetchRequest, available, get_source, register

__all__ = [
    "BAR_COLUMNS",
    "EXCHANGE_TIMEZONE",
    "DataSource",
    "EmptyHistoryError",
    "FetchRequest",
    "LeanDataError",
    "Provenance",
    "Resolution",
    "SecurityHistory",
    "SourceError",
    "SymbolSpec",
    "UnsupportedResolutionError",
    "ValidationError",
    "available",
    "get_source",
    "register",
]
