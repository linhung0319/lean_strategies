"""Bar file layouts -- how a resolution maps onto LEAN's zip/CSV tree.

Each resolution differs in three ways: where the zip lives, what the entry
inside it is called, and how the timestamp column is written. A layout owns
exactly those three decisions, so adding hour or minute support means adding
a class here and one line to ``BAR_LAYOUTS`` -- nothing else in the package
changes.

Naming rules come from ``LeanData.GenerateZipFilePath`` /
``GenerateZipEntryName`` in ``Lean/Common/Util/LeanData.cs``; the row format
from ``LeanData.GenerateLine`` and ``TradeBar.LineParseScale``.
"""

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import ClassVar, Final, Iterator, Protocol

import pandas as pd

from ..errors import UnsupportedResolutionError
from ..model import BAR_COLUMNS, Resolution, SymbolSpec
from .formatting import format_volume, scale_price

# Bar CSVs are LF separated with no trailing newline, matching the bundled
# spy.csv byte for byte.
LINE_SEPARATOR: Final = b"\n"


@dataclass(frozen=True)
class BarFile:
    """One zip archive and the single CSV entry inside it."""

    zip_path: PurePosixPath  # relative to the data root, e.g. equity/usa/daily/spy.zip
    entry_name: str  # e.g. spy.csv
    payload: bytes


class BarLayout(Protocol):
    resolution: ClassVar[Resolution]

    def entries(self, symbol: SymbolSpec, bars: pd.DataFrame) -> Iterator[BarFile]:
        ...


def _render_rows(bars: pd.DataFrame, time_format: str) -> bytes:
    """Render OHLCV rows. Prices are scaled to deci-cents; volume is not."""
    lines = []
    for timestamp, row in zip(bars.index, bars[list(BAR_COLUMNS)].itertuples(index=False)):
        lines.append(
            ",".join(
                (
                    timestamp.strftime(time_format),
                    scale_price(row.open),
                    scale_price(row.high),
                    scale_price(row.low),
                    scale_price(row.close),
                    format_volume(row.volume),
                )
            ).encode("ascii")
        )
    return LINE_SEPARATOR.join(lines)


class DailyBarLayout:
    """One zip for the whole history: ``equity/<market>/daily/<key>.zip``."""

    resolution: ClassVar[Resolution] = Resolution.DAILY
    time_format: ClassVar[str] = "%Y%m%d 00:00"

    def entries(self, symbol: SymbolSpec, bars: pd.DataFrame) -> Iterator[BarFile]:
        yield BarFile(
            zip_path=PurePosixPath(symbol.security_type, symbol.market, "daily", f"{symbol.key}.zip"),
            entry_name=f"{symbol.key}.csv",
            payload=_render_rows(bars, self.time_format),
        )


BAR_LAYOUTS: dict[Resolution, BarLayout] = {
    Resolution.DAILY: DailyBarLayout(),
    # Hour is a one-line variant (time_format "%Y%m%d %H:%M", directory
    # "hour"); minute needs per-day grouping and millisecond timestamps.
    # Both slot in here without touching the writer.
}


def layout_for(resolution: Resolution) -> BarLayout:
    try:
        return BAR_LAYOUTS[resolution]
    except KeyError:
        supported = ", ".join(sorted(r.value for r in BAR_LAYOUTS))
        raise UnsupportedResolutionError(
            f"no bar layout for resolution {resolution.value!r}; implemented: {supported}"
        ) from None
