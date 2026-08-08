"""The contract every data source adapter implements."""

from dataclasses import dataclass
from datetime import date
from typing import ClassVar, Protocol, runtime_checkable

from ..errors import EmptyHistoryError, UnsupportedResolutionError
from ..model import Resolution, SecurityHistory


@dataclass(frozen=True)
class FetchRequest:
    """What to download. Deliberately vendor-neutral."""

    ticker: str
    resolution: Resolution = Resolution.DAILY
    start: date | None = None  # inclusive
    end: date | None = None  # inclusive
    market: str = "usa"
    security_type: str = "equity"
    exchange_code: str = "P"


@runtime_checkable
class DataSource(Protocol):
    """Turn a FetchRequest into the canonical SecurityHistory.

    A Protocol rather than an ABC so a new adapter does not have to import
    anything from here -- it just has to have the right shape.
    """

    name: ClassVar[str]

    def supported_resolutions(self) -> frozenset[Resolution]:
        ...

    def fetch(self, request: FetchRequest) -> SecurityHistory:
        ...


class BaseSource:
    """Optional mixin with the two checks every adapter ends up writing."""

    name: ClassVar[str] = "base"

    def supported_resolutions(self) -> frozenset[Resolution]:
        raise NotImplementedError

    def _check_resolution(self, resolution: Resolution) -> None:
        supported = self.supported_resolutions()
        if resolution not in supported:
            available = ", ".join(sorted(item.value for item in supported))
            raise UnsupportedResolutionError(
                f"source {self.name!r} does not support resolution {resolution.value!r}; "
                f"supported: {available}"
            )

    def _check_not_empty(self, frame, request: FetchRequest) -> None:
        if frame is None or len(frame) == 0:
            window = f"{request.start or 'start'} -> {request.end or 'today'}"
            raise EmptyHistoryError(
                f"source {self.name!r} returned no {request.resolution.value} bars for "
                f"{request.ticker} over {window}"
            )
