"""Building and inspecting a standalone LEAN data folder.

Pointing LEAN's ``data-folder`` at a directory *replaces* ``Lean/Data``
wholesale -- it is not layered on top. So a folder holding converted data
also has to carry the reference data LEAN reads unconditionally, or the run
fails deep into startup with an unhelpful stack trace.

Three reference paths matter for a US equity backtest:

``market-hours``
    Trading sessions and time zones. The ``Equity-usa-[*]`` wildcard covers
    any new ticker, so nothing needs registering -- but the file must exist.
``symbol-properties``
    Tick size, lot size, quote currency. Same wildcard story.
``alternative/interest-rate/usa``
    The risk-free curve. Easy to miss because nothing errors without it:
    LEAN just computes a different Sharpe Ratio, which would quietly poison
    any comparison against the bundled data.
"""

import shutil
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Final

MIRRORED: Final = (
    PurePosixPath("market-hours"),
    PurePosixPath("symbol-properties"),
    PurePosixPath("alternative/interest-rate/usa"),
)

INTEREST_RATE_FILE: Final = PurePosixPath("alternative/interest-rate/usa/interest-rate.csv")
# Version-controlled top-up for the file above; see leandata/reference/README.md.
INTEREST_RATE_PATCH: Final = Path(__file__).resolve().parent.parent / "reference" / "interest-rate-usa.csv"


@dataclass(frozen=True)
class SeriesInfo:
    ticker: str
    rows: int
    first_date: date
    last_date: date


@dataclass(frozen=True)
class OverlayInfo:
    root: Path
    missing_reference: tuple[PurePosixPath, ...]
    series: tuple[SeriesInfo, ...]

    def describe(self) -> str:
        lines = [f"data folder {self.root}"]
        if self.missing_reference:
            missing = ", ".join(str(path) for path in self.missing_reference)
            lines.append(f"  MISSING reference data: {missing}")
        for entry in self.series:
            lines.append(
                f"  {entry.ticker:<8} {entry.rows:>6} rows  {entry.first_date} -> {entry.last_date}"
            )
        if not self.series:
            lines.append("  no daily equity data")
        return "\n".join(lines)


def ensure_overlay(root: Path, lean_data: Path, *, refresh: bool = False) -> list[Path]:
    """Copy LEAN's reference data into ``root``. Never writes to ``lean_data``.

    Copies rather than symlinks: junctions on Windows 10 Home need either
    elevation or Developer Mode.
    """
    root = Path(root)
    lean_data = Path(lean_data)
    if not lean_data.is_dir():
        raise FileNotFoundError(f"LEAN data folder not found at {lean_data}")

    copied: list[Path] = []
    for relative in MIRRORED:
        source = lean_data / relative
        if not source.exists():
            raise FileNotFoundError(f"reference data missing from the LEAN clone: {source}")
        destination = root / relative
        if destination.exists() and not refresh and not _is_stale(source, destination):
            continue
        if destination.exists() and refresh:
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)
        copied.append(destination)
    return copied


def extend_interest_rates(root: Path, patch: Path = INTEREST_RATE_PATCH) -> list[str]:
    """Top up the mirrored risk-free rate curve with rows LEAN does not ship.

    LEAN's bundled file stops at 2023-07-27 and upstream has not updated it
    since; ``InterestRateProvider`` then carries that last rate forward
    forever, quietly skewing Sharpe, Sortino, Alpha and Treynor for anything
    backtested past mid-2023.

    Only rows dated strictly after the existing last entry are appended, so
    this is idempotent and never contradicts what LEAN shipped. Returns the
    rows added.
    """
    target = Path(root) / INTEREST_RATE_FILE
    patch = Path(patch)
    if not target.exists() or not patch.exists():
        return []

    existing = [line for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(existing) < 2:
        return []
    last_date = existing[-1].split(",")[0]

    added = []
    for line in patch.read_text(encoding="utf-8").splitlines()[1:]:  # skip the header
        if not line.strip():
            continue
        if line.split(",")[0] > last_date:
            added.append(line.strip())

    if added:
        target.write_text("\n".join(existing + added) + "\n", encoding="utf-8")
    return added


# The FOMC meets eight times a year. A curve whose last change is this old
# relative to the data has probably missed one; a shorter gap is just a hold.
RATE_HOLD_TOLERANCE_DAYS: Final = 400


def stale_rate_warning(last_rate_date: date, last_bar_date: date) -> str | None:
    """Flag a risk-free curve that has probably missed an FOMC change.

    The curve is a step function, so holding the last rate past its date is
    correct behaviour, not staleness -- only an implausibly long hold is
    worth reporting.
    """
    gap = (last_bar_date - last_rate_date).days
    if gap <= RATE_HOLD_TOLERANCE_DAYS:
        return None
    return (
        f"WARNING   the risk-free rate has been unchanged since {last_rate_date} while the data "
        f"runs to {last_bar_date} ({gap} days). The FOMC meets eight times a year, so a change "
        f"has probably been missed; Sharpe, Sortino, Alpha and Treynor would be off. "
        f"Add the missing rows to leandata/reference/interest-rate-usa.csv.\n"
    )


def interest_rate_range(root: Path) -> tuple[date, date] | None:
    """First and last dated rate in a data folder, for reporting."""
    target = Path(root) / INTEREST_RATE_FILE
    if not target.exists():
        return None
    rows = [line for line in target.read_text(encoding="utf-8").splitlines() if line.strip()][1:]
    if not rows:
        return None
    parse = lambda line: datetime.strptime(line.split(",")[0], "%Y-%m-%d").date()
    return parse(rows[0]), parse(rows[-1])


def missing_reference_data(root: Path) -> list[PurePosixPath]:
    """Reference paths a LEAN run needs that this folder does not have."""
    root = Path(root)
    return [relative for relative in MIRRORED if not (root / relative).exists()]


def describe(root: Path, *, market: str = "usa", security_type: str = "equity") -> OverlayInfo:
    root = Path(root)
    daily = root / security_type / market / "daily"
    series: list[SeriesInfo] = []
    if daily.is_dir():
        for archive in sorted(daily.glob("*.zip")):
            frame = read_daily_csv(archive)
            if not frame:
                continue
            series.append(
                SeriesInfo(
                    ticker=archive.stem.upper(),
                    rows=len(frame),
                    first_date=frame[0][0],
                    last_date=frame[-1][0],
                )
            )
    return OverlayInfo(
        root=root,
        missing_reference=tuple(missing_reference_data(root)),
        series=tuple(series),
    )


def read_daily_csv(archive: Path) -> list[tuple[date, float, float, float, float, float]]:
    """Read a LEAN daily bar zip back into unscaled (date, o, h, l, c, volume).

    Used by the verify command and the dataset comparison, so both read the
    same bytes LEAN reads rather than trusting what the writer thought it
    wrote.
    """
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        if not names:
            return []
        payload = zipped.read(names[0]).decode("ascii")

    rows = []
    for line in payload.splitlines():
        if not line.strip():
            continue
        stamp, open_, high, low, close, volume = line.split(",")
        rows.append(
            (
                datetime.strptime(stamp, "%Y%m%d %H:%M").date(),
                float(open_) / 10_000,
                float(high) / 10_000,
                float(low) / 10_000,
                float(close) / 10_000,
                float(volume),
            )
        )
    return rows


def _is_stale(source: Path, destination: Path) -> bool:
    """Has the LEAN clone moved ahead of this mirror?

    A local edit is not staleness. LEAN's bundled ``interest-rate.csv`` stops
    in 2023 and upstream has not touched it since, so anyone backtesting past
    that has to append rows to their copy -- and re-copying over that on the
    next convert would silently undo it. So a mirror that is *newer* than its
    source is left alone whatever its size, and only a missing file or a
    genuinely newer source triggers a re-copy.
    """
    for path in source.rglob("*"):
        if path.is_dir():
            continue
        mirror = destination / path.relative_to(source)
        if not mirror.exists():
            return True
        source_stat, mirror_stat = path.stat(), mirror.stat()
        if source_stat.st_mtime > mirror_stat.st_mtime + 1:
            return True
        if source_stat.st_size != mirror_stat.st_size and source_stat.st_mtime >= mirror_stat.st_mtime:
            return True  # a half-finished copy, not an edit
    return False
