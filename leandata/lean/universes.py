"""LEAN ETF constituent universe files.

One CSV per day under ``equity/<market>/universes/etf/<etf>/<yyyyMMdd>.csv``,
which is where ``ETFConstituentUniverse.GetSource``
(``Lean/Common/Data/UniverseSelection/ETFConstituentUniverse.cs:80``) looks. The
path is built from ``config.Symbol.Underlying.Value``, so the folder is named
after the ETF the universe was created for and nothing needs registering.

Each line is six fields, matching ``ETFConstituentUniverse.Reader``::

    <ticker>,<security id>,<yyyyMMdd last update>,<weight>,<shares held>,<market value>

The last three are ``decimal?``: an empty field reads back as ``None``, which
is how a number this pipeline cannot supply is spelled. Weight is a fraction,
not a percentage -- LEAN's own SPY file gives AAPL ``0.06213795``.

Two details taken from that bundled file rather than from the reader, which
would accept either:

* Rows are sorted by ticker.
* Every line ends in a trailing comma, because ``MarketValue`` is empty and
  the reader indexes ``split[5]`` unconditionally -- a line without it throws.

Timing: ``Time`` is the file's date and ``EndTime`` is a day later, so the
file dated D is what selection sees on D+1. A file per trading day is
therefore what makes selection point-in-time rather than a day's lookahead.
"""

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Final, Iterable, Mapping

import numpy as np
import pandas as pd

from ..errors import ValidationError
from ..model import ConstituentPanel

DATE_FORMAT: Final = "%Y%m%d"
LINE_SEPARATOR: Final = b"\r\n"
# Lean's own file carries 8 decimal places on the weight.
WEIGHT_DECIMALS: Final = 8


@dataclass(frozen=True)
class UniverseReport:
    root: Path
    etf: str
    files: int
    first_date: date
    last_date: date
    rows: int
    unpriced_rows: int

    def describe(self) -> str:
        share = self.unpriced_rows / self.rows if self.rows else 0.0
        return "\n".join(
            [
                f"universe  {self.files} files  {self.first_date} -> {self.last_date}",
                f"          {self.rows} constituent rows, "
                f"{self.unpriced_rows} with no weight ({share:.2%})",
                f"          {self.root / 'equity'}",
            ]
        )


def render_number(value: float) -> str:
    """A decimal LEAN can parse, or an empty field when the value is unknown."""
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.{WEIGHT_DECIMALS}f}"


def render_universe_file(
    rows: Iterable[tuple[str, str, float, float, float]], last_update: date
) -> bytes:
    """``rows`` are ``(ticker, security id, weight, shares held, market value)``."""
    stamp = last_update.strftime(DATE_FORMAT)
    lines = [
        ",".join(
            [
                ticker,
                security_id,
                stamp,
                render_number(weight),
                render_number(shares_held),
                render_number(market_value),
            ]
        )
        for ticker, security_id, weight, shares_held, market_value in sorted(rows)
    ]
    if not lines:
        raise ValidationError("refusing to write an empty universe file")
    return LINE_SEPARATOR.join(line.encode("ascii") for line in lines) + LINE_SEPARATOR


class UniverseWriter:
    """Writes a ConstituentPanel into a LEAN data folder."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def directory(self, panel: ConstituentPanel) -> Path:
        relative = PurePosixPath(
            panel.etf.security_type, panel.etf.market, "universes", "etf", panel.etf.key
        )
        return self.root / relative

    def write(self, panel: ConstituentPanel, security_ids: Mapping[str, str]) -> UniverseReport:
        """``security_ids`` maps each ticker in the panel to its SID string.

        A SID encodes the security's first date, which is the first row of the
        map file the bar writer produced -- so the two have to be built from
        the same download or LEAN will fail to resolve the symbol.
        """
        missing = sorted(set(panel.tickers) - set(security_ids))
        if missing:
            raise ValidationError(
                f"no security id for {len(missing)} ticker(s) in the panel: "
                f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}"
            )

        target = self.directory(panel)
        target.mkdir(parents=True, exist_ok=True)

        written = 0
        total_rows = 0
        unpriced = 0
        days = []
        for timestamp, group in panel.rows.groupby(level="date", sort=True):
            day = pd.Timestamp(timestamp).date()
            rows = [
                (
                    ticker,
                    security_ids[ticker],
                    float(record.weight),
                    float(record.shares_held),
                    float(record.market_value),
                )
                for (_, ticker), record in zip(group.index, group.itertuples(index=False))
            ]
            total_rows += len(rows)
            unpriced += sum(1 for row in rows if not np.isfinite(row[2]))
            self._write_bytes(target / f"{day.strftime(DATE_FORMAT)}.csv",
                              render_universe_file(rows, day))
            written += 1
            days.append(day)

        return UniverseReport(
            root=self.root,
            etf=panel.etf.ticker,
            files=written,
            first_date=min(days),
            last_date=max(days),
            rows=total_rows,
            unpriced_rows=unpriced,
        )

    def _write_bytes(self, target: Path, payload: bytes) -> None:
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, target)
