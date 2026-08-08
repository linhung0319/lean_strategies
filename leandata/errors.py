"""Exceptions raised by the data conversion pipeline."""


class LeanDataError(Exception):
    """Base class for every error this package raises."""


class ValidationError(LeanDataError):
    """A SecurityHistory violated the canonical contract."""


class SourceError(LeanDataError):
    """A data source could not be resolved, configured or reached."""


class EmptyHistoryError(SourceError):
    """A source returned no bars for the requested symbol and window."""


class UnsupportedResolutionError(LeanDataError):
    """The requested resolution has no implementation yet."""
