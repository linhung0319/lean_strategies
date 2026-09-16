"""Point-in-time S&P 500 membership from the ``fja05680/sp500`` dataset.

QuantConnect's own ETF constituents dataset is not part of the free LEAN
distribution, so the membership half of a constituents universe has to come
from somewhere else. This is the cheapest credible source: one CSV,
``date,tickers``, one row per membership change back to 1996, each row listing
the *whole* roster as of that date rather than a delta.

It is a volunteer-maintained reconstruction, not an index-provider feed. Two
consequences worth carrying into any result built on it:

* Additions and removals are dated by announcement-adjacent effective dates
  that can be a day or two off the real rebalance.
* It carries membership only. Weights are not in the file and have to be
  reconstructed -- see :mod:`leandata.weights`.

``path`` replays a previously downloaded copy, which keeps a rebuild offline
and pins the roster: the upstream file is rewritten in place as the index
changes, so re-downloading a year from now would silently move a backtest.
"""

import csv
import io
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import ClassVar, Final

from ..errors import SourceError
from ..model import IndexMembership, Provenance

# The "(Updated)" file is the one upstream keeps current; the unsuffixed one
# is a frozen historical snapshot.
DEFAULT_URL: Final = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
INDEX_NAME: Final = "S&P 500"


class Sp500GithubSource:
    """Reads the dataset from ``path`` if given, otherwise downloads ``url``."""

    name: ClassVar[str] = "sp500-github"

    def __init__(
        self,
        path: str | Path | None = None,
        url: str = DEFAULT_URL,
        timeout: float = 60.0,
    ) -> None:
        self.path = Path(path) if path else None
        self.url = url
        self.timeout = float(timeout)

    def fetch_membership(self) -> IndexMembership:
        text, origin = self._read()
        changes = parse_membership_csv(text)
        return IndexMembership(
            name=INDEX_NAME,
            changes=changes,
            provenance=Provenance(
                source=self.name,
                fetched_at=datetime.now(timezone.utc),
                source_version=f"{changes[-1][0]} ({len(changes)} rows)",
                options={"origin": origin},
            ),
        )

    def save(self, target: Path) -> Path:
        """Write the raw CSV to ``target`` so a later run can replay it."""
        text, _ = self._read()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="")
        return target

    def _read(self) -> tuple[str, str]:
        if self.path is not None:
            if not self.path.exists():
                raise FileNotFoundError(f"membership snapshot not found at {self.path}")
            return self.path.read_text(encoding="utf-8"), str(self.path)
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout) as response:
                payload = response.read()
        except OSError as exc:
            raise SourceError(f"could not download {self.url}: {exc}") from exc
        return payload.decode("utf-8"), self.url


def parse_membership_csv(text: str) -> tuple[tuple[date, tuple[str, ...]], ...]:
    """``date,tickers`` rows into the sorted change list IndexMembership wants."""
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "date" not in reader.fieldnames or "tickers" not in reader.fieldnames:
        raise SourceError(
            f"expected a CSV with 'date' and 'tickers' columns, got {reader.fieldnames}"
        )

    rows: dict[date, tuple[str, ...]] = {}
    for number, row in enumerate(reader, start=2):
        raw_date = (row.get("date") or "").strip()
        if not raw_date:
            continue
        try:
            day = datetime.strptime(raw_date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise SourceError(f"line {number}: unparsable date {raw_date!r}") from exc
        members = tuple(
            sorted({item.strip().upper() for item in (row.get("tickers") or "").split(",") if item.strip()})
        )
        if not members:
            raise SourceError(f"line {number}: {day} has an empty roster")
        # Later rows win, so a re-stated date corrects rather than duplicates.
        rows[day] = members

    if not rows:
        raise SourceError("the membership CSV has no usable rows")
    return tuple((day, rows[day]) for day in sorted(rows))
