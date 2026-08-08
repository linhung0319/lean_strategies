"""LEAN factor files: ``<date>,<price factor>,<split factor>,<reference price>``.

LEAN stores raw, as-traded prices and reconstructs split- and
dividend-adjusted series at runtime from this file. Since equities default to
``DataNormalizationMode.Adjusted``, the factor file is not optional metadata
-- it directly moves every number a backtest reports.

The construction walks corporate actions newest to oldest, starting from a
terminal row whose factors are 1. Each action writes a row dated the
*previous trading day*, carrying that day's raw close as the reference price.
That mirrors ``Lean/ToolBox/FactorFileGenerator.cs``, with one deliberate
departure: the dividend factor here follows the runtime's
``CorporateFactorRow.Apply`` (``pf * (C - D) / C``) rather than the
generator's ``pf * (1 - D * splitFactor / C)``. The two disagree whenever a
split factor is not 1, and the runtime is the one that has to agree with
itself when it replays the file.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Final, Literal

import pandas as pd

from ..errors import ValidationError
from .formatting import net_round, strip_zeros, to_decimal, to_str

# Time.EndOfTime in Lean/Common/Time.cs
TERMINAL_DATE: Final = date(2050, 12, 31)
DATE_FORMAT: Final = "%Y%m%d"
# Factor files are CRLF separated with no trailing newline, matching the
# bundled spy.csv byte for byte.
LINE_SEPARATOR: Final = b"\r\n"

PRICE_FACTOR_PLACES: Final = 7
SPLIT_FACTOR_PLACES: Final = 8
REFERENCE_PRICE_PLACES: Final = 4

MissingDayPolicy = Literal["snap", "skip", "error"]


@dataclass(frozen=True)
class FactorRow:
    date: date
    price_factor: Decimal
    split_factor: Decimal
    reference_price: Decimal

    def render(self) -> str:
        return ",".join(
            (
                self.date.strftime(DATE_FORMAT),
                to_str(net_round(self.price_factor, PRICE_FACTOR_PLACES)),
                to_str(net_round(self.split_factor, SPLIT_FACTOR_PLACES)),
                strip_zeros(net_round(self.reference_price, REFERENCE_PRICE_PLACES)),
            )
        )


def build_factor_rows(
    bars: pd.DataFrame,
    dividends: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    on_missing_event_day: MissingDayPolicy = "snap",
) -> tuple[list[FactorRow], list[str]]:
    """Build ascending factor rows from raw daily bars and corporate actions.

    ``bars`` must carry RAW prices -- the same prices that go into the bar
    file. Dividend amounts must be raw cash per share as of their ex-date.
    Returns the rows plus any warnings worth surfacing to the operator.
    """
    warnings: list[str] = []
    calendar, closes = _trading_calendar(bars)
    events, event_warnings = _resolve_events(
        dividends, splits, calendar, on_missing_event_day
    )
    warnings.extend(event_warnings)

    rows = [FactorRow(TERMINAL_DATE, Decimal(1), Decimal(1), Decimal(0))]
    for ex_date in sorted(events, reverse=True):
        position = int(calendar.searchsorted(ex_date, side="left"))
        if position == 0:
            # No trading day precedes the action, so there is no close to
            # anchor it to. Everything older is in the same position.
            warnings.append(
                f"corporate action on {ex_date.date()} is at or before the first bar; "
                f"stopping the factor walk there"
            )
            break
        previous_day = calendar[position - 1]
        close = to_decimal(closes.iloc[position - 1])
        price_factor = rows[-1].price_factor
        split_factor = rows[-1].split_factor

        dividend = events[ex_date].get("dividend")
        if dividend:
            if close <= dividend:
                warnings.append(
                    f"dividend {dividend} on {ex_date.date()} is >= the prior close {close}; "
                    f"ignoring it"
                )
            else:
                price_factor = price_factor * (close - dividend) / close

        ratio = events[ex_date].get("ratio")
        if ratio:
            split_factor = split_factor / ratio

        rows.append(FactorRow(previous_day.date(), price_factor, split_factor, close))

    first_day = calendar[0].date()
    if rows[-1].date != first_day:
        # The oldest row anchors the chain at the start of the data. LEAN's
        # bundled files all carry a reference price of 1 here (spy, aapl, fb),
        # so it is a marker rather than a real close.
        rows.append(FactorRow(first_day, rows[-1].price_factor, rows[-1].split_factor, Decimal(1)))

    rows.reverse()
    _assert_ascending(rows)
    return rows, warnings


def render_factor_file(rows: list[FactorRow]) -> bytes:
    return LINE_SEPARATOR.join(row.render().encode("ascii") for row in rows)


def _trading_calendar(bars: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.Series]:
    """The bars *are* the calendar -- never infer trading days from weekdays."""
    normalized = pd.DatetimeIndex(bars.index).normalize()
    closes = pd.Series(bars["close"].to_numpy(), index=normalized)
    # Intraday resolutions collapse to one close per day; the day's last bar wins.
    closes = closes[~closes.index.duplicated(keep="last")]
    return pd.DatetimeIndex(closes.index), closes


def _resolve_events(
    dividends: pd.DataFrame,
    splits: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    policy: MissingDayPolicy,
) -> tuple[dict[pd.Timestamp, dict[str, Decimal]], list[str]]:
    """Group actions by ex-date and snap them onto the trading calendar."""
    warnings: list[str] = []
    raw: dict[pd.Timestamp, dict[str, Decimal]] = defaultdict(dict)

    for timestamp, amount in dividends["amount"].items():
        day = pd.Timestamp(timestamp).normalize()
        value = to_decimal(amount)
        # Multiple distributions on one ex-date are a single cash event.
        raw[day]["dividend"] = raw[day].get("dividend", Decimal(0)) + value

    for timestamp, ratio in splits["ratio"].items():
        value = to_decimal(ratio)
        if value == 1:
            continue  # a 1:1 split is a no-op, not a factor row
        day = pd.Timestamp(timestamp).normalize()
        raw[day]["ratio"] = raw[day].get("ratio", Decimal(1)) * value

    resolved: dict[pd.Timestamp, dict[str, Decimal]] = defaultdict(dict)
    last_day = calendar[-1]
    for day in sorted(raw):
        if day > last_day:
            warnings.append(f"corporate action on {day.date()} is after the last bar; dropping it")
            continue
        position = int(calendar.searchsorted(day, side="left"))
        aligned = calendar[position]
        if aligned != day:
            if policy == "error":
                raise ValidationError(
                    f"corporate action on {day.date()} does not fall on a trading day"
                )
            if policy == "skip":
                warnings.append(f"corporate action on {day.date()} is not a trading day; dropping it")
                continue
            warnings.append(
                f"corporate action on {day.date()} is not a trading day; "
                f"snapping it to {aligned.date()}"
            )
        target = resolved[aligned]
        for leg, value in raw[day].items():
            if leg == "dividend":
                target["dividend"] = target.get("dividend", Decimal(0)) + value
            else:
                target["ratio"] = target.get("ratio", Decimal(1)) * value
    return dict(resolved), warnings


def _assert_ascending(rows: list[FactorRow]) -> None:
    for previous, current in zip(rows, rows[1:]):
        if previous.date >= current.date:
            raise ValidationError(
                f"factor rows must be strictly ascending, got {previous.date} then {current.date}"
            )
