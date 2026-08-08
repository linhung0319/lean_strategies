"""Yahoo Finance adapter.

Split into a client that talks to the network and a source that only
transforms frames. Everything interesting lives in the pure half, so the
tests drive it with a hand-written fake client and never touch Yahoo.

The one thing worth understanding before reading further: **Yahoo's OHLCV is
already split-adjusted**, even with ``auto_adjust=False`` (that flag only
controls dividend adjustment and the extra ``Adj Close`` column). LEAN wants
raw, as-traded prices with the adjustments expressed in the factor file
instead, so this adapter un-adjusts the splits back out. SPY happens to have
no splits in the bundled window, which makes it a poor test of that path --
AAPL is the one that exercises it.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import ClassVar, Literal

import numpy as np
import pandas as pd

from ..errors import SourceError
from ..model import (
    BAR_COLUMNS,
    EXCHANGE_TIMEZONE,
    PRICE_COLUMNS,
    Provenance,
    Resolution,
    SecurityHistory,
    SymbolSpec,
    empty_dividends,
    empty_splits,
)
from .base import BaseSource, FetchRequest

InvalidBarPolicy = Literal["clamp", "drop", "error"]

_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}
_INTERVALS = {Resolution.DAILY: "1d", Resolution.HOUR: "1h", Resolution.MINUTE: "1m"}


class YFinanceClient:
    """The only part of this package that makes a network request."""

    def __init__(self, *, repair: bool = False) -> None:
        try:
            import yfinance
        except ImportError as exc:  # pragma: no cover - exercised by hand
            raise SourceError(
                "yfinance is not installed. Run 'uv sync --group data' to add it."
            ) from exc
        self._yfinance = yfinance
        self.repair = repair
        self.version = getattr(yfinance, "__version__", "unknown")

    def history(
        self, ticker: str, start: date | None, end: date | None, interval: str
    ) -> pd.DataFrame:
        # yfinance treats `end` as exclusive; callers pass an inclusive date.
        exclusive_end = end + timedelta(days=1) if end else None
        return self._yfinance.Ticker(ticker).history(
            start=start,
            end=exclusive_end,
            interval=interval,
            auto_adjust=False,
            actions=True,
            repair=self.repair,
        )


@dataclass
class SanitizeReport:
    dropped_missing: int = 0
    dropped_invalid: int = 0
    clamped: int = 0
    filled_volume: int = 0
    warnings: list[str] = field(default_factory=list)

    def messages(self) -> list[str]:
        notes = list(self.warnings)
        if self.dropped_missing:
            notes.append(f"dropped {self.dropped_missing} bar(s) with missing or non-positive prices")
        if self.dropped_invalid:
            notes.append(f"dropped {self.dropped_invalid} bar(s) with an impossible high/low range")
        if self.clamped:
            notes.append(f"widened high/low on {self.clamped} bar(s) to enclose open/close")
        if self.filled_volume:
            notes.append(f"filled missing volume with 0 on {self.filled_volume} bar(s)")
        return notes


class YFinanceSource(BaseSource):
    name: ClassVar[str] = "yfinance"

    def __init__(
        self,
        client: YFinanceClient | None = None,
        *,
        on_invalid_bar: InvalidBarPolicy = "clamp",
        repair: bool = False,
    ) -> None:
        self._client = client
        self._repair = repair
        self.on_invalid_bar = on_invalid_bar
        self.warnings: list[str] = []

    @property
    def client(self) -> YFinanceClient:
        if self._client is None:
            self._client = YFinanceClient(repair=self._repair)
        return self._client

    def supported_resolutions(self) -> frozenset[Resolution]:
        # Only daily is wired into the LEAN writer today. Yahoo's intraday
        # history is also capped at a few weeks, so it would not help the
        # long backtests this project runs.
        return frozenset({Resolution.DAILY})

    def fetch(self, request: FetchRequest) -> SecurityHistory:
        self._check_resolution(request.resolution)
        raw = self.client.history(
            request.ticker, request.start, request.end, _INTERVALS[request.resolution]
        )
        self._check_not_empty(raw, request)

        frame = _flatten_columns(raw)
        frame = frame.set_axis(_naive_exchange_index(frame.index, request.resolution), axis=0)
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()

        dividends, splits = _extract_actions(frame)
        bars, report = _sanitize_bars(_select_bar_columns(frame), self.on_invalid_bar)
        self._check_not_empty(bars, request)
        # Actions outside the surviving bar range cannot be anchored to a close.
        dividends = dividends.loc[(dividends.index >= bars.index[0]) & (dividends.index <= bars.index[-1])]
        splits = splits.loc[splits.index <= bars.index[-1]]

        bars, dividends = _unadjust_splits(bars, dividends, splits)
        self.warnings = report.messages()

        history = SecurityHistory(
            symbol=SymbolSpec(
                ticker=request.ticker,
                market=request.market,
                security_type=request.security_type,
                exchange_code=request.exchange_code,
            ),
            resolution=request.resolution,
            bars=bars,
            dividends=dividends,
            splits=splits,
            provenance=Provenance(
                source=self.name,
                fetched_at=datetime.now(timezone.utc),
                source_version=self.client.version,
                options={
                    "auto_adjust": "False",
                    "repair": str(self.client.repair),
                    "on_invalid_bar": self.on_invalid_bar,
                },
            ),
        )
        return history.clip(request.start, request.end)


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse the ('Close', 'SPY') MultiIndex some yfinance paths return."""
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    return frame


def _naive_exchange_index(index, resolution: Resolution) -> pd.DatetimeIndex:
    """Convert to naive exchange-local time, which is what LEAN files carry.

    Converting *before* dropping the zone is what stops a UTC-stamped feed
    from shifting pre-open timestamps onto the wrong calendar day.
    """
    index = pd.DatetimeIndex(index)
    if index.tz is not None:
        index = index.tz_convert(EXCHANGE_TIMEZONE).tz_localize(None)
    if resolution is Resolution.DAILY:
        index = index.normalize()
    return index


def _extract_actions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dividends = empty_dividends()
    splits = empty_splits()

    if "Dividends" in frame.columns:
        amounts = pd.to_numeric(frame["Dividends"], errors="coerce").fillna(0.0)
        paid = amounts > 0
        if paid.any():
            dividends = pd.DataFrame({"amount": amounts[paid].astype("float64")})
            dividends.index = pd.DatetimeIndex(dividends.index, name="ex_date")

    if "Stock Splits" in frame.columns:
        # Yahoo writes 0.0 for "no split", not NaN, and 0.1 for a 1-for-10
        # reverse split -- so filter on != 0 rather than on notna().
        ratios = pd.to_numeric(frame["Stock Splits"], errors="coerce").fillna(0.0)
        occurred = ratios > 0
        if occurred.any():
            splits = pd.DataFrame({"ratio": ratios[occurred].astype("float64")})
            splits.index = pd.DatetimeIndex(splits.index, name="ex_date")

    return dividends, splits


def _select_bar_columns(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in _COLUMN_MAP if column not in frame.columns]
    if missing:
        raise SourceError(f"yfinance response is missing column(s): {', '.join(missing)}")
    # Drops 'Adj Close', 'Dividends', 'Stock Splits' and any 'Capital Gains'.
    bars = frame[list(_COLUMN_MAP)].rename(columns=_COLUMN_MAP)
    return bars[list(BAR_COLUMNS)].astype("float64")


def _sanitize_bars(bars: pd.DataFrame, policy: InvalidBarPolicy) -> tuple[pd.DataFrame, SanitizeReport]:
    report = SanitizeReport()

    volume = bars["volume"]
    missing_volume = ~np.isfinite(volume.to_numpy())
    if missing_volume.any():
        report.filled_volume = int(missing_volume.sum())
        bars = bars.copy()
        bars.loc[missing_volume, "volume"] = 0.0

    prices = bars[list(PRICE_COLUMNS)].to_numpy(dtype="float64")
    # Yahoo occasionally emits a phantom bar on a mistaken holiday. A gap is
    # always safer than a synthetic bar, which would feed the indicators.
    usable = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    if not usable.all():
        report.dropped_missing = int((~usable).sum())
        bars = bars.loc[usable]

    if bars.empty:
        return bars, report

    open_ = bars["open"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    close = bars["close"].to_numpy()
    wanted_high = np.maximum(open_, close)
    wanted_low = np.minimum(open_, close)
    broken = (high < wanted_high) | (low > wanted_low)

    if broken.any():
        if policy == "error":
            first = bars.index[int(np.argmax(broken))]
            raise SourceError(f"bar at {first} has an impossible high/low range")
        if policy == "drop":
            report.dropped_invalid = int(broken.sum())
            bars = bars.loc[~broken]
        else:
            report.clamped = int(broken.sum())
            bars = bars.copy()
            bars["high"] = np.maximum(high, wanted_high)
            bars["low"] = np.minimum(low, wanted_low)

    return bars, report


def _unadjust_splits(
    bars: pd.DataFrame, dividends: pd.DataFrame, splits: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recover raw prices from Yahoo's split-adjusted feed.

    With ``cum(t)`` the product of every split ratio whose ex-date is strictly
    after ``t`` -- how many of today's shares one share held at ``t`` became::

        raw_price(t)    = yahoo_price(t)  * cum(t)
        raw_volume(t)   = yahoo_volume(t) / cum(t)
        raw_dividend(t) = yahoo_dividend(t) * cum(t)

    Strictly after, because a bar or dividend dated on the ex-date itself is
    already quoted in post-split terms.
    """
    if splits.empty:
        return bars, dividends

    ratios = splits["ratio"].sort_index()
    split_days = ratios.index.to_numpy()
    # suffix[i] is the product of ratios[i:], so suffix[k] with k = number of
    # split days at or before t is exactly cum(t).
    suffix = np.append(np.cumprod(ratios.to_numpy()[::-1])[::-1], 1.0)

    bar_cum = suffix[np.searchsorted(split_days, bars.index.to_numpy(), side="right")]
    bars = bars.copy()
    for column in PRICE_COLUMNS:
        bars[column] = bars[column].to_numpy() * bar_cum
    bars["volume"] = bars["volume"].to_numpy() / bar_cum

    if not dividends.empty:
        dividend_cum = suffix[np.searchsorted(split_days, dividends.index.to_numpy(), side="right")]
        dividends = dividends.copy()
        dividends["amount"] = dividends["amount"].to_numpy() * dividend_cum

    return bars, dividends
