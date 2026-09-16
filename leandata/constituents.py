"""Building a LEAN data folder for a whole index, not a single symbol.

``convert`` handles one ticker. A constituents-universe backtest needs three
things at once, and they have to agree with each other:

1. bars, factor and map files for **every** ticker that was in the index at
   any point in the window -- including the ones that left, and including the
   ones the vendor now knows by a different name, or the universe is only as
   honest as today's membership;
2. one universe file per trading day saying who was in on that day and how
   heavily;
3. a security identifier per ticker that encodes the *same* first date the
   map file carries, because that is how LEAN resolves one back to the other.

This module is the orchestration. The pieces it drives each stay ignorant of
the others: ``sources`` downloads, ``weights`` does the arithmetic,
``lean.writer`` and ``lean.universes`` write bytes.

Everything downloaded is written to a snapshot cache first and read back from
it on the next run. Over six hundred symbols that matters for more than
speed: a run that dies two thirds of the way through a vendor's rate limit
resumes instead of restarting, and a rebuild months later reproduces the same
backtest rather than quietly picking up restated history.
"""

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from .errors import LeanDataError
from .lean.identifiers import equity_security_id
from .lean.universes import UniverseReport, UniverseWriter
from .lean.writer import LeanDataWriter
from .model import (
    IndexMembership,
    Provenance,
    Resolution,
    SecurityHistory,
    SymbolSpec,
)
from .snapshot import read_snapshot, snapshot_paths, write_snapshot
from .sources.base import FetchRequest
from .tickers import lean_ticker, load_renames, resolve, yahoo_ticker
from .weights import cap_weighted_panel, restated_market_caps, trading_calendar

SHARES_HEADER = ["date", "shares"]

# Yahoo tolerates a steady stream of requests but not a flood of them, and a
# full build is well over a thousand. Cached tickers do not sleep.
DEFAULT_PAUSE_SECONDS = 0.15


@dataclass(frozen=True)
class TickerOutcome:
    """What happened to one member of the index."""

    index_ticker: str
    lean_ticker: str
    status: str  # written | unavailable
    bars: int = 0
    first_date: date | None = None
    last_date: date | None = None
    shares_points: int = 0
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.status == "written"


@dataclass
class BuildReport:
    etf: str
    window: tuple[date, date]
    outcomes: list[TickerOutcome] = field(default_factory=list)
    universe: UniverseReport | None = None
    membership_rows: int = 0
    membership_file: Path | None = None

    @property
    def unavailable(self) -> list[TickerOutcome]:
        return [item for item in self.outcomes if not item.usable]

    @property
    def unpriced(self) -> list[TickerOutcome]:
        return [item for item in self.outcomes if item.usable and not item.shares_points]

    def describe(self) -> str:
        usable = [item for item in self.outcomes if item.usable]
        lines = [
            f"index     {self.membership_rows} membership rows",
            f"symbols   {len(usable)} of {len(self.outcomes)} written  "
            f"{self.window[0]} -> {self.window[1]}",
        ]
        if self.membership_file:
            lines.insert(1, f"          {self.membership_file}")
        if self.unavailable:
            names = ", ".join(item.index_ticker for item in self.unavailable)
            lines.append(f"missing   {len(self.unavailable)} with no vendor data: {names}")
        if self.unpriced:
            names = ", ".join(item.index_ticker for item in self.unpriced)
            lines.append(
                f"unweighed {len(self.unpriced)} priced but with no share count: {names}"
            )
        if self.universe is not None:
            lines.append(self.universe.describe())
        return "\n".join(lines)


def build_constituent_universe(
    *,
    etf_ticker: str,
    membership: IndexMembership,
    source,
    out: Path,
    start: date,
    end: date,
    cache: Path,
    market: str = "usa",
    etf_exchange_code: str = "P",
    constituent_exchange_code: str = "Q",
    refresh: bool = False,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    renames: dict[str, str] | None = None,
    log: Callable[[str], None] = print,
) -> BuildReport:
    """Download, convert and index an entire ETF constituents dataset.

    ``source`` is any :class:`~leandata.sources.base.DataSource` that also
    offers ``fetch_shares_outstanding``; today that means the yfinance
    adapter. ``constituent_exchange_code`` only sets the third column of each
    map file, which a US equity backtest never reads -- the market-hours
    wildcard covers every exchange -- so one value for all of them is not the
    simplification it looks like.
    """
    out = Path(out)
    cache = Path(cache)
    renames = load_renames() if renames is None else renames
    report = BuildReport(
        etf=etf_ticker.upper(),
        window=(start, end),
        membership_rows=len(membership.changes),
    )

    # The ETF first: it is the benchmark, it names the folder the universe
    # files land in, and its bars are the trading calendar everything else is
    # reindexed onto.
    log(f"[etf] {etf_ticker}")
    etf_history, _ = _load(
        source, etf_ticker, etf_ticker, start, end, cache, market, etf_exchange_code, refresh
    )
    if etf_history is None:
        raise LeanDataError(f"no data for the ETF {etf_ticker}; cannot build a calendar")
    LeanDataWriter(out).write(etf_history)
    calendar = trading_calendar(etf_history, start, end)
    log(f"[etf] {len(calendar)} trading days {calendar[0].date()} -> {calendar[-1].date()}")

    members = membership.union(start, end)
    renamed = sum(1 for name in members if resolve(name, renames) != name)
    log(
        f"[members] {len(members)} tickers were in {membership.name} during the window"
        + (f", {renamed} of them since renamed" if renamed else "")
    )

    histories: dict[str, SecurityHistory] = {}
    shares: dict[str, pd.Series] = {}
    security_ids: dict[str, str] = {}
    rename: dict[str, str] = {}

    for position, index_ticker in enumerate(members, start=1):
        # A renamed company is downloaded, stored and written under the name
        # it trades under now, which is the only one the vendor answers to.
        current = resolve(index_ticker, renames)
        key = lean_ticker(current)
        rename[index_ticker] = key
        history, share_series = _load(
            source,
            current,
            yahoo_ticker(current),
            start,
            end,
            cache,
            market,
            constituent_exchange_code,
            refresh,
            pause_seconds=pause_seconds,
        )
        if history is None:
            report.outcomes.append(
                TickerOutcome(index_ticker, key, "unavailable", reason="no bars from the source")
            )
            log(f"[{position}/{len(members)}] {index_ticker:<14} no data")
            continue

        written = LeanDataWriter(out).write(history)
        histories[index_ticker] = history
        shares[index_ticker] = share_series
        # The identifier has to encode the first row of the map file the
        # writer just produced, or LEAN cannot resolve one back to the other.
        security_ids[key] = equity_security_id(key, written.first_date, market)

        report.outcomes.append(
            TickerOutcome(
                index_ticker,
                key,
                "written",
                bars=written.bar_count,
                first_date=written.first_date,
                last_date=written.last_date,
                shares_points=len(share_series),
            )
        )
        label = index_ticker if current == index_ticker else f"{index_ticker}>{key}"
        log(
            f"[{position}/{len(members)}] {label:<14} {written.bar_count:>5} bars  "
            f"{written.first_date} -> {written.last_date}  shares {len(share_series):>4}"
        )

    log("[weights] reconstructing market caps")
    caps = restated_market_caps(histories, shares, calendar)
    panel = cap_weighted_panel(
        SymbolSpec(ticker=etf_ticker, market=market, exchange_code=etf_exchange_code),
        membership,
        caps,
        rename=rename,
        provenance=Provenance(
            source="cap-weight-proxy",
            fetched_at=datetime.now(timezone.utc),
            source_version="1",
            options={"membership": membership.name, "prices": getattr(source, "name", "unknown")},
        ),
    )
    # Members with no bars are still written, so the file keeps saying who was
    # in the index; the only first date available for one is the window's.
    for ticker in panel.tickers:
        security_ids.setdefault(ticker, equity_security_id(ticker, start, market))

    log(f"[universe] writing {len(panel.dates)} daily files")
    report.universe = UniverseWriter(out).write(panel, security_ids)
    return report


def _relabel(
    history: SecurityHistory, ticker: str, market: str, exchange_code: str
) -> SecurityHistory:
    """Restate a history under the ticker LEAN will look for on disk."""
    return SecurityHistory(
        symbol=SymbolSpec(
            ticker=ticker,
            market=market,
            security_type=history.symbol.security_type,
            exchange_code=exchange_code,
        ),
        resolution=history.resolution,
        bars=history.bars,
        dividends=history.dividends,
        splits=history.splits,
        provenance=history.provenance,
    )


def _load(
    source,
    index_ticker: str,
    vendor_ticker: str,
    start: date,
    end: date,
    cache: Path,
    market: str,
    exchange_code: str,
    refresh: bool,
    pause_seconds: float = 0.0,
) -> tuple[SecurityHistory | None, pd.Series]:
    """Snapshot cache first, vendor second. Returns ``(history, shares)``.

    A ticker the vendor cannot serve leaves a marker file behind rather than
    nothing, so a resumed run does not pay for the same failure again.
    """
    stem = lean_ticker(index_ticker)
    bars_path, actions_path = snapshot_paths(cache, stem)
    shares_path = cache / f"{stem}_shares.csv"
    missing_marker = cache / f"{stem}.missing"

    if not refresh and bars_path.exists():
        bars, dividends, splits = read_snapshot(bars_path, actions_path)
        history = SecurityHistory(
            symbol=SymbolSpec(ticker=stem, market=market, exchange_code=exchange_code),
            resolution=Resolution.DAILY,
            bars=bars,
            dividends=dividends,
            splits=splits,
            provenance=Provenance(
                source="cache",
                fetched_at=datetime.now(timezone.utc),
                options={"path": str(bars_path)},
            ),
        )
        return history.clip(start, end), _read_shares(shares_path)
    if not refresh and missing_marker.exists():
        return None, pd.Series(dtype="float64")

    request = FetchRequest(
        ticker=vendor_ticker,
        resolution=Resolution.DAILY,
        start=start,
        end=end,
        market=market,
        exchange_code=exchange_code,
    )
    if pause_seconds:
        time.sleep(pause_seconds)
    try:
        history = source.fetch(request)
    except Exception:  # noqa: BLE001 - a delisted ticker fails vendor-specific ways
        cache.mkdir(parents=True, exist_ok=True)
        missing_marker.write_text("", encoding="ascii")
        return None, pd.Series(dtype="float64")

    try:
        share_series = source.fetch_shares_outstanding(request)
    except Exception:  # noqa: BLE001 - shares are optional, bars are not
        share_series = pd.Series(dtype="float64")

    history = _relabel(history, stem, market, exchange_code)
    write_snapshot(history, cache, stem)
    _write_shares(shares_path, share_series)
    return history, share_series


def _read_shares(path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], name="time"))
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], name="time"))
    series = pd.Series(
        frame["shares"].astype("float64").to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(frame["date"]), name="time"),
    )
    return series.sort_index()


def _write_shares(path: Path, series: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": [pd.Timestamp(value).strftime("%Y-%m-%d") for value in series.index],
            "shares": series.to_numpy(dtype="float64"),
        },
        columns=SHARES_HEADER,
    ).to_csv(path, index=False)
