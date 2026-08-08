"""Registry of data source adapters.

Adding a source is a new module here plus one ``register`` call at the bottom
of this file. Nothing in ``leandata.lean``, the CLI or the comparison script
changes.

Factories are lazy so that importing ``leandata`` never imports ``yfinance``
and its compiled dependency stack -- the LEAN runs share this virtualenv and
have no business loading any of it.
"""

import importlib
from typing import Callable

from ..errors import SourceError
from .base import BaseSource, DataSource, FetchRequest

__all__ = ["BaseSource", "DataSource", "FetchRequest", "register", "get_source", "available"]

SourceFactory = Callable[..., DataSource]

_REGISTRY: dict[str, SourceFactory] = {}


def register(name: str, factory: SourceFactory) -> None:
    _REGISTRY[name] = factory


def available() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_source(name: str, **options) -> DataSource:
    """Instantiate a registered source. Options are source-specific."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        known = ", ".join(available()) or "none registered"
        raise SourceError(f"unknown data source {name!r}; available: {known}") from None
    try:
        return factory(**options)
    except TypeError as exc:
        raise SourceError(f"bad options for source {name!r}: {exc}") from exc


def _module(name: str):
    return importlib.import_module(f"{__name__}.{name}")


register("yfinance", lambda **options: _module("yfinance_source").YFinanceSource(**options))
register("csv", lambda **options: _module("csv_source").CsvSource(**options))
