"""LEAN map files: ``<date>,<mapped ticker>,<exchange code>``.

A map file records the ticker's history of renames plus its listing and
delisting dates. Nothing here renames anything -- a freshly downloaded symbol
has no rename history -- so the file is always two rows: the first trading
date and LEAN's far-future sentinel.

Format checked against ``Lean/Data/equity/usa/map_files/spy.csv``, which is
CRLF terminated *with* a trailing newline (unlike the factor and bar files).
"""

from dataclasses import dataclass
from datetime import date
from typing import Final

from ..model import SymbolSpec

# Time.EndOfTime in Lean/Common/Time.cs
TERMINAL_DATE: Final = date(2050, 12, 31)
DATE_FORMAT: Final = "%Y%m%d"
LINE_SEPARATOR: Final = b"\r\n"


@dataclass(frozen=True)
class MapRow:
    date: date
    ticker: str
    exchange_code: str

    def render(self) -> str:
        return f"{self.date.strftime(DATE_FORMAT)},{self.ticker},{self.exchange_code}"


def build_map_rows(symbol: SymbolSpec, first_date: date) -> list[MapRow]:
    listed = symbol.listed_from or first_date
    return [
        MapRow(listed, symbol.key, symbol.exchange_code),
        MapRow(TERMINAL_DATE, symbol.key, symbol.exchange_code),
    ]


def render_map_file(rows: list[MapRow]) -> bytes:
    body = LINE_SEPARATOR.join(row.render().encode("ascii") for row in rows)
    return body + LINE_SEPARATOR
