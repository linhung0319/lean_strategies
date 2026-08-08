"""The canonical intermediate representation every data source converts into.

This module is the contract between the ``sources`` package (which knows how
each vendor formats its download) and the ``lean`` package (which knows how
LEAN wants bytes on disk). Neither side imports the other; both import this.

Nothing here knows about LEAN's file format, and nothing here knows about any
particular vendor. Keeping that true is what makes adding a new source cheap.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Final, Mapping

import numpy as np
import pandas as pd

from .errors import ValidationError

BAR_COLUMNS: Final = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS: Final = ("open", "high", "low", "close")
EXCHANGE_TIMEZONE: Final = "America/New_York"


class Resolution(str, Enum):
    """The bar sizes LEAN understands. Only DAILY is implemented today."""

    DAILY = "daily"
    HOUR = "hour"
    MINUTE = "minute"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class SymbolSpec:
    """Identity of the instrument, independent of any data source."""

    ticker: str
    market: str = "usa"
    security_type: str = "equity"
    # Third column of a LEAN map file: P=NYSE Arca, Q=NASDAQ, N=NYSE.
    exchange_code: str = "P"
    # First date of the map file. Defaults to the first bar when None.
    listed_from: date | None = None

    def __post_init__(self) -> None:
        if not self.ticker or not self.ticker.strip():
            raise ValidationError("SymbolSpec.ticker must be a non-empty string")
        object.__setattr__(self, "ticker", self.ticker.strip().upper())

    @property
    def key(self) -> str:
        """The lowercase form LEAN uses for file and directory names."""
        return self.ticker.lower()


@dataclass(frozen=True)
class Provenance:
    """Where a history came from, so a surprising backtest can be traced back."""

    source: str
    fetched_at: datetime
    source_version: str = "unknown"
    options: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat(),
            "source_version": self.source_version,
            "options": dict(self.options),
        }


def empty_bars() -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(BAR_COLUMNS), dtype="float64")
    frame.index = pd.DatetimeIndex([], name="time")
    return frame


def empty_dividends() -> pd.DataFrame:
    frame = pd.DataFrame(columns=["amount"], dtype="float64")
    frame.index = pd.DatetimeIndex([], name="ex_date")
    return frame


def empty_splits() -> pd.DataFrame:
    frame = pd.DataFrame(columns=["ratio"], dtype="float64")
    frame.index = pd.DatetimeIndex([], name="ex_date")
    return frame


@dataclass(frozen=True)
class SecurityHistory:
    """RAW, as-traded market data in naive exchange-local time.

    Every adapter must satisfy this contract:

    ``bars.index``
        ``DatetimeIndex``, tz-naive, ``America/New_York``, strictly increasing
        and unique. Daily and hour rows carry the bar *start* time; daily is
        midnight.
    ``bars.columns``
        Exactly ``BAR_COLUMNS``, float64, no NaN or inf, OHLC strictly
        positive, ``low <= min(open, close)``, ``high >= max(open, close)``,
        ``volume >= 0``.
    Prices are RAW
        Neither split- nor dividend-adjusted. A source whose feed is
        split-adjusted -- Yahoo's is -- must un-adjust before constructing
        this. LEAN reproduces the adjustments from the factor file, so raw
        prices in plus a factor file out is the only combination that matches
        how the bundled data behaves.
    ``dividends``
        Index is the ex-date (tz-naive, midnight-normalised), column
        ``amount`` is the RAW cash per share as of that ex-date, > 0.
    ``splits``
        Index is the ex-date, column ``ratio``; 2.0 is a 2-for-1 forward
        split, 0.5 a 1-for-2 reverse split.

    Dividends and splits may fall outside the bar range or on non-trading
    days; clipping and calendar alignment are the writer's job, not the
    adapter's.
    """

    symbol: SymbolSpec
    resolution: Resolution
    bars: pd.DataFrame
    dividends: pd.DataFrame = field(default_factory=empty_dividends)
    splits: pd.DataFrame = field(default_factory=empty_splits)
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        validate(self)

    @property
    def first_date(self) -> date:
        return self.bars.index[0].date()

    @property
    def last_date(self) -> date:
        return self.bars.index[-1].date()

    def clip(self, start: date | None = None, end: date | None = None) -> "SecurityHistory":
        """Restrict the bars to an inclusive date window.

        Dividends and splits are deliberately *not* clipped here: the factor
        builder needs to see actions that sit just outside the bar range to
        decide whether to drop them, and it owns that decision.
        """
        mask = pd.Series(True, index=self.bars.index)
        if start is not None:
            mask &= self.bars.index >= pd.Timestamp(start)
        if end is not None:
            # end is a date, so include everything up to the end of that day.
            mask &= self.bars.index <= pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        return SecurityHistory(
            symbol=self.symbol,
            resolution=self.resolution,
            bars=self.bars.loc[mask.to_numpy()],
            dividends=self.dividends,
            splits=self.splits,
            provenance=self.provenance,
        )


def validate(history: SecurityHistory) -> None:
    """Raise ValidationError if the contract in SecurityHistory is broken."""
    _validate_bars(history.bars)
    _validate_events(history.dividends, "dividends", "amount")
    _validate_events(history.splits, "splits", "ratio")


def _validate_bars(bars: pd.DataFrame) -> None:
    if not isinstance(bars, pd.DataFrame):
        raise ValidationError(f"bars must be a DataFrame, got {type(bars).__name__}")
    if tuple(bars.columns) != BAR_COLUMNS:
        raise ValidationError(f"bars.columns must be exactly {BAR_COLUMNS}, got {tuple(bars.columns)}")
    if len(bars) == 0:
        raise ValidationError("bars must not be empty")

    index = bars.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValidationError(f"bars.index must be a DatetimeIndex, got {type(index).__name__}")
    if index.tz is not None:
        raise ValidationError(
            f"bars.index must be tz-naive exchange-local time ({EXCHANGE_TIMEZONE}), got tz={index.tz}"
        )
    if not index.is_monotonic_increasing:
        raise ValidationError("bars.index must be sorted oldest to newest")
    if index.has_duplicates:
        duplicates = index[index.duplicated()][:3].tolist()
        raise ValidationError(f"bars.index must be unique; duplicates include {duplicates}")

    for column in BAR_COLUMNS:
        values = bars[column].to_numpy(dtype="float64", copy=False)
        if not np.isfinite(values).all():
            bad = _first_bad(index, ~np.isfinite(values))
            raise ValidationError(f"bars.{column} contains NaN or inf, first at {bad}")

    for column in PRICE_COLUMNS:
        values = bars[column].to_numpy(dtype="float64", copy=False)
        if (values <= 0).any():
            bad = _first_bad(index, values <= 0)
            raise ValidationError(f"bars.{column} must be strictly positive, first violation at {bad}")

    volume = bars["volume"].to_numpy(dtype="float64", copy=False)
    if (volume < 0).any():
        raise ValidationError(f"bars.volume must be >= 0, first violation at {_first_bad(index, volume < 0)}")

    open_, high = bars["open"].to_numpy(), bars["high"].to_numpy()
    low, close = bars["low"].to_numpy(), bars["close"].to_numpy()
    too_low = high < np.maximum(open_, close)
    if too_low.any():
        raise ValidationError(f"bars.high must be >= max(open, close), first violation at {_first_bad(index, too_low)}")
    too_high = low > np.minimum(open_, close)
    if too_high.any():
        raise ValidationError(f"bars.low must be <= min(open, close), first violation at {_first_bad(index, too_high)}")


def _validate_events(frame: pd.DataFrame, name: str, column: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValidationError(f"{name} must be a DataFrame, got {type(frame).__name__}")
    if tuple(frame.columns) != (column,):
        raise ValidationError(f"{name}.columns must be exactly ('{column}',), got {tuple(frame.columns)}")
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValidationError(f"{name}.index must be a DatetimeIndex, got {type(index).__name__}")
    if len(frame) == 0:
        return
    if index.tz is not None:
        raise ValidationError(f"{name}.index must be tz-naive, got tz={index.tz}")
    if not index.is_monotonic_increasing:
        raise ValidationError(f"{name}.index must be sorted oldest to newest")
    if index.has_duplicates:
        raise ValidationError(f"{name}.index must be unique; combine same-day events before constructing")
    values = frame[column].to_numpy(dtype="float64", copy=False)
    if not np.isfinite(values).all():
        raise ValidationError(f"{name}.{column} contains NaN or inf")
    if (values <= 0).any():
        raise ValidationError(f"{name}.{column} must be strictly positive, first violation at {_first_bad(index, values <= 0)}")


def _first_bad(index: pd.DatetimeIndex, mask) -> str:
    position = int(np.argmax(mask))
    return f"row {position} ({index[position]})"
