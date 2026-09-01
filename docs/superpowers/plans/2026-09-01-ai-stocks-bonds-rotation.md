# AI Stocks/Bonds Rotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run QuantConnect's `AIStocksBondsRotationAlgorithm` on the local LEAN
engine against SPY/GLD/BND from yfinance, BTCUSD from Bitfinex's public API and
three FRED macro series, so its statistics can be set beside the published
cloud figures.

**Architecture:** `leandata` gains two new paths without disturbing the equity
one. Crypto reuses `SecurityHistory` and every on-disk difference is confined
to a new `AssetProfile` looked up from `SymbolSpec.security_type`. FRED is not
OHLCV, so it gets a narrow `ValueSeries` type and its own writer. The algorithm
file stays byte-for-byte pasteable into QuantConnect by defining a local `Fred`
type only when the name is absent.

**Tech Stack:** Python 3.11.11 x64, pandas 2.2.3, scikit-learn, stdlib
`urllib.request`, LEAN (C#) via `run_local.py`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-ai-stocks-bonds-rotation-design.md`

## Global Constraints

- **Every existing test must keep passing, untouched.** Equity output must stay
  byte-identical. Run `uv run pytest` before and after every task.
- **No new runtime dependency except `scikit-learn`.** The two new sources use
  stdlib `urllib.request`. `run_local.py` points LEAN's embedded interpreter at
  this same `.venv`, so anything added here is loaded by every backtest.
- **`scikit-learn` goes in `[project.dependencies]`, not a group.** The
  algorithm imports it at backtest time. `yfinance` stays in the `data` group.
- **`leandata/lean/` must never import `leandata/sources/`, and vice versa.**
  Both import `leandata/model.py`. This is the property that makes adding a
  source cheap; do not break it.
- **Bar CSVs are LF separated with no trailing newline.** Factor and map files
  are CRLF terminated and written in binary mode.
- **Backtest window: 2021-09-01 to 2026-08-31. Fetch window: 2016-01-01 to
  2026-08-31.**
- **Never write to `../Lean/Data`.** Everything lands in `lean_strategies/data/`.
- Commit style follows the repo: `feat:`, `fix:`, `docs:`, `chore:`, `test:`.
- Do not touch the pre-existing uncommitted changes in `variants.py`,
  `algorithms/stock_monthly_rebalance.py` or `rebal.tmp`. They belong to
  separate work. Task 10 modifies `variants.py`; add to it, do not revert it.

## File Structure

| File | Responsibility |
|---|---|
| `leandata/lean/profiles.py` (new) | The only place asset-class differences live: scaling, volume format, zip suffix, whether auxiliary files exist, timezone |
| `leandata/lean/formatting.py` | Gains `unscaled()` for numbers LEAN stores as given |
| `leandata/lean/bars.py` | Derives the profile from the symbol; renders rows accordingly |
| `leandata/lean/writer.py` | Writes factor/map files only when the profile says they exist |
| `leandata/lean/overlay.py` | `read_daily_csv` learns not to unscale crypto; `describe` reports crypto and FRED |
| `leandata/lean/customdata.py` (new) | `ValueSeries` to `alternative/fred/<key>.csv` |
| `leandata/model.py` | Gains `ValueSeries`; `SecurityHistory` unchanged |
| `leandata/sources/bitfinex.py` (new) | Bitfinex public candles to a crypto `SecurityHistory` |
| `leandata/sources/fred.py` (new) | `fredgraph.csv` to a `ValueSeries` |
| `leandata/cli.py` | `--security-type`, new `convert-series` command, `check` reports both |
| `algorithms/ai_stocks_bonds_rotation.py` (new) | The algorithm, verbatim from QuantConnect but for the `Fred` fallback and fixed dates |
| `tests/lean_stubs.py` | Grows the API surface the new algorithm touches |

---

### Task 1: Asset profiles and the unscaled number format

**Files:**
- Create: `lean_strategies/leandata/lean/profiles.py`
- Modify: `lean_strategies/leandata/lean/formatting.py` (append at end)
- Test: `lean_strategies/tests/test_profiles.py`

**Interfaces:**
- Consumes: `leandata.errors.LeanDataError`, and from `formatting`:
  `to_decimal`, `strip_zeros`, `ROUND_HALF_EVEN`, `Decimal`.
- Produces:
  - `leandata.lean.profiles.AssetProfile` — frozen dataclass with fields
    `security_type: str`, `timezone: str`, `price_scaled: bool`,
    `volume_integral: bool`, `zip_suffix: str`, `writes_corporate_actions: bool`
  - `leandata.lean.profiles.EQUITY: AssetProfile`
  - `leandata.lean.profiles.CRYPTO: AssetProfile`
  - `leandata.lean.profiles.profile_for(security_type: str) -> AssetProfile`
  - `leandata.lean.formatting.unscaled(value, *, places: int = 8) -> str`

- [ ] **Step 1: Write the failing test**

Create `lean_strategies/tests/test_profiles.py`:

```python
"""Asset-class profiles and the unscaled number format.

The numbers asserted here come from LEAN's own crypto file,
Lean/Data/crypto/coinbase/daily/btcusd_trade.zip, whose first row is
    20141201 00:00,300,370,300,370,0.05655554
and from a real Bitfinex response row
    [1451606400000, 429.17, 433.98, 436.49, 426.26, 13077.668908090154]
"""

import pytest

from leandata.errors import LeanDataError
from leandata.lean.formatting import unscaled
from leandata.lean.profiles import CRYPTO, EQUITY, profile_for


def test_unscaled_leaves_a_price_alone():
    assert unscaled(429.17) == "429.17"
    assert unscaled(77683) == "77683"


def test_unscaled_strips_trailing_zeros_rather_than_padding():
    assert unscaled(370.0) == "370"
    assert unscaled(0.05655554) == "0.05655554"


def test_unscaled_never_writes_scientific_notation():
    # Decimal.normalize() would turn this into 1E+6 without the "f" format.
    assert unscaled(1000000) == "1000000"


def test_unscaled_rounds_to_eight_places_half_to_even():
    assert unscaled(13077.668908090154) == "13077.66890809"


def test_equity_profile_describes_todays_behaviour():
    assert profile_for("equity") is EQUITY
    assert EQUITY.price_scaled is True
    assert EQUITY.volume_integral is True
    assert EQUITY.zip_suffix == ""
    assert EQUITY.writes_corporate_actions is True
    assert EQUITY.timezone == "America/New_York"


def test_crypto_profile_describes_leans_crypto_files():
    assert profile_for("crypto") is CRYPTO
    assert CRYPTO.price_scaled is False
    assert CRYPTO.volume_integral is False
    assert CRYPTO.zip_suffix == "_trade"
    assert CRYPTO.writes_corporate_actions is False
    assert CRYPTO.timezone == "UTC"


def test_unknown_security_type_names_the_ones_that_exist():
    with pytest.raises(LeanDataError) as caught:
        profile_for("future")
    assert "future" in str(caught.value)
    assert "crypto" in str(caught.value)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd lean_strategies && uv run pytest tests/test_profiles.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'leandata.lean.profiles'`

- [ ] **Step 3: Create `leandata/lean/profiles.py`**

```python
"""Per-asset-class differences in how LEAN stores bars on disk.

Everything that varies between equity and crypto lives here, so adding
another asset class later is a new instance rather than a new branch in
bars.py, writer.py and overlay.py.

Each difference was confirmed against LEAN's source, not assumed:

``price_scaled``
    Equity prices are stored in deci-cents; crypto prices are stored as they
    trade. ``TradeBar.ParseEquity`` calls ``LineParseScale(..., useScaleFactor:
    true)`` while ``TradeBar.ParseCrypto`` calls ``LineParseNoScale``
    (Lean/Common/Data/Market/TradeBar.cs:381, :432).
``volume_integral``
    Equity volume is a share count; crypto volume is a coin amount and is
    fractional. Lean/Data/crypto/coinbase/daily/btcusd_trade.zip carries
    ``20141201 00:00,300,370,300,370,0.05655554``.
``zip_suffix``
    Crypto hour and daily archives carry the tick type -- ``btcusd_trade.zip``
    against equity's ``spy.zip`` (Lean/Common/Util/LeanData.cs:800-816).
``writes_corporate_actions``
    Only equities have factor and map files.
``timezone``
    Here rather than as ``model.EXCHANGE_TIMEZONE`` because Bitfinex keeps UTC
    while US equities keep New York (Lean/Data/market-hours/
    market-hours-database.json, entry ``Crypto-bitfinex-[*]``).
"""

from dataclasses import dataclass
from typing import Final

from ..errors import LeanDataError


@dataclass(frozen=True)
class AssetProfile:
    security_type: str
    timezone: str
    price_scaled: bool
    volume_integral: bool
    zip_suffix: str
    writes_corporate_actions: bool


EQUITY: Final = AssetProfile(
    security_type="equity",
    timezone="America/New_York",
    price_scaled=True,
    volume_integral=True,
    zip_suffix="",
    writes_corporate_actions=True,
)

CRYPTO: Final = AssetProfile(
    security_type="crypto",
    timezone="UTC",
    price_scaled=False,
    volume_integral=False,
    zip_suffix="_trade",
    writes_corporate_actions=False,
)

PROFILES: Final = {profile.security_type: profile for profile in (EQUITY, CRYPTO)}


def profile_for(security_type: str) -> AssetProfile:
    try:
        return PROFILES[security_type]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise LeanDataError(
            f"no asset profile for security type {security_type!r}; known: {known}"
        ) from None
```

- [ ] **Step 4: Append `unscaled` to `leandata/lean/formatting.py`**

Add at the end of the file, after `format_volume`:

```python
# Crypto prices and volumes are both stored exactly as they come, and both
# carry more decimal places than a share price does. Eight places is what
# LEAN's own crypto files use: btcusd_trade.zip has volumes like 0.05655554.
UNSCALED_PLACES: Final = 8


def unscaled(value, *, places: int = UNSCALED_PLACES) -> str:
    """Format a number LEAN stores verbatim, as crypto prices and volumes are.

    One function serves both because crypto makes no distinction: neither the
    price nor the volume is scaled, and both keep their decimals. Equity is
    the asset class that needs two, and it already has ``scale_price`` and
    ``format_volume``.
    """
    quantized = to_decimal(value).quantize(
        Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN
    )
    return strip_zeros(quantized)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd lean_strategies && uv run pytest tests/test_profiles.py -v`
Expected: PASS, 8 tests

- [ ] **Step 6: Run the whole suite to confirm nothing regressed**

Run: `cd lean_strategies && uv run pytest`
Expected: PASS, no failures

- [ ] **Step 7: Commit**

```bash
cd lean_strategies
git add leandata/lean/profiles.py leandata/lean/formatting.py tests/test_profiles.py
git commit -m "feat: add asset profiles so crypto and equity can share one writer"
```

---

### Task 2: Bars and writer dispatch on the profile

**Files:**
- Modify: `lean_strategies/leandata/lean/bars.py`
- Modify: `lean_strategies/leandata/lean/writer.py`
- Test: `lean_strategies/tests/test_crypto_lean_files.py`

**Interfaces:**
- Consumes: `profile_for`, `AssetProfile` (Task 1); `formatting.unscaled` (Task 1);
  existing `SecurityHistory`, `SymbolSpec`, `BarFile`, `LeanDataWriter`.
- Produces:
  - `DailyBarLayout.entries(symbol, bars)` keeps its **existing two-argument
    signature** and derives the profile from `symbol.security_type` internally.
    Do not add a parameter; `tests/test_lean_files.py:26` calls it with two
    arguments and must keep working.
  - `WriteReport.factor_file` and `WriteReport.map_file` become `Path | None`.

- [ ] **Step 1: Write the failing test**

Create `lean_strategies/tests/test_crypto_lean_files.py`:

```python
"""Crypto bars land in the shape LEAN's own crypto files use.

The reference is Lean/Data/crypto/coinbase/daily/btcusd_trade.zip, whose
first row is
    20141201 00:00,300,370,300,370,0.05655554
Three things there differ from an equity file: the "_trade" suffix on the
archive name, prices that are not multiplied by 10000, and a fractional
volume.
"""

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from leandata.lean.writer import LeanDataWriter
from leandata.model import BAR_COLUMNS, Resolution, SecurityHistory, SymbolSpec

BUNDLED = Path(__file__).resolve().parents[2] / "Lean" / "Data" / "crypto" / "coinbase" / "daily"


def crypto_history(ticker="BTCUSD"):
    index = pd.DatetimeIndex(
        pd.date_range("2016-01-01", periods=3, freq="D"), name="time"
    )
    bars = pd.DataFrame(
        {
            "open": [429.17, 433.89, 432.66],
            "high": [436.49, 435.80, 433.07],
            "low": [426.26, 430.00, 421.73],
            "close": [433.98, 432.70, 428.39],
            "volume": [13077.668908090154, 6347.95293087, 19201.83336317],
        },
        index=index,
    )
    return SecurityHistory(
        symbol=SymbolSpec(ticker=ticker, market="bitfinex", security_type="crypto"),
        resolution=Resolution.DAILY,
        bars=bars[list(BAR_COLUMNS)],
    )


def written_lines(tmp_path):
    LeanDataWriter(tmp_path).write(crypto_history())
    archive = tmp_path / "crypto" / "bitfinex" / "daily" / "btcusd_trade.zip"
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.namelist() == ["btcusd.csv"]
        return zipped.read("btcusd.csv").decode("ascii").split("\n")


def test_crypto_daily_archive_carries_the_trade_suffix(tmp_path):
    report = LeanDataWriter(tmp_path).write(crypto_history())
    assert report.bar_files == (
        tmp_path / "crypto" / "bitfinex" / "daily" / "btcusd_trade.zip",
    )


def test_crypto_prices_are_not_scaled_to_deci_cents(tmp_path):
    lines = written_lines(tmp_path)
    assert lines[0] == "20160101 00:00,429.17,436.49,426.26,433.98,13077.66890809"


def test_crypto_volume_keeps_its_decimals(tmp_path):
    lines = written_lines(tmp_path)
    assert lines[1].endswith(",6347.95293087")


def test_crypto_writes_no_factor_or_map_file(tmp_path):
    report = LeanDataWriter(tmp_path).write(crypto_history())
    assert report.factor_file is None
    assert report.map_file is None
    assert not (tmp_path / "crypto" / "bitfinex" / "factor_files").exists()
    assert not (tmp_path / "crypto" / "bitfinex" / "map_files").exists()


def test_crypto_report_describe_omits_the_absent_files(tmp_path):
    report = LeanDataWriter(tmp_path).write(crypto_history())
    text = report.describe()
    assert "factors" not in text
    assert "map" not in text
    assert "bars      3 rows" in text


@pytest.mark.skipif(not BUNDLED.is_dir(), reason="../Lean is not checked out")
def test_written_rows_have_the_same_shape_as_leans_own_crypto_file(tmp_path):
    with zipfile.ZipFile(BUNDLED / "btcusd_trade.zip") as zipped:
        reference = zipped.read(zipped.namelist()[0]).decode("ascii").splitlines()[0]
    ours = written_lines(tmp_path)[0]
    # Same field count, same timestamp shape, and a price that is plainly not
    # a deci-cent integer.
    assert len(ours.split(",")) == len(reference.split(",")) == 6
    assert ours.split(",")[0].endswith(" 00:00")
    assert reference.split(",")[0].endswith(" 00:00")
    assert "." in ours.split(",")[1]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd lean_strategies && uv run pytest tests/test_crypto_lean_files.py -v`
Expected: FAIL. The writer currently writes to `crypto/bitfinex/daily/btcusd.zip`
with deci-cent prices and unconditionally builds factor and map files.

- [ ] **Step 3: Modify `leandata/lean/bars.py`**

Change the import block to add the profile lookup and `unscaled`:

```python
from ..errors import UnsupportedResolutionError
from ..model import BAR_COLUMNS, Resolution, SymbolSpec
from .formatting import format_volume, scale_price, unscaled
from .profiles import AssetProfile, profile_for
```

Replace `_render_rows` with:

```python
def _render_rows(bars: pd.DataFrame, time_format: str, profile: AssetProfile) -> bytes:
    """Render OHLCV rows in the format the asset class calls for.

    Equity prices are deci-cent integers and volume is a share count; crypto
    keeps both verbatim. See leandata/lean/profiles.py for the citations.
    """
    price = scale_price if profile.price_scaled else unscaled
    volume = format_volume if profile.volume_integral else unscaled
    lines = []
    for timestamp, row in zip(bars.index, bars[list(BAR_COLUMNS)].itertuples(index=False)):
        lines.append(
            ",".join(
                (
                    timestamp.strftime(time_format),
                    price(row.open),
                    price(row.high),
                    price(row.low),
                    price(row.close),
                    volume(row.volume),
                )
            ).encode("ascii")
        )
    return LINE_SEPARATOR.join(lines)
```

Replace `DailyBarLayout` with:

```python
class DailyBarLayout:
    """One zip for the whole history: ``<type>/<market>/daily/<key><suffix>.zip``.

    The suffix is empty for equity and ``_trade`` for crypto
    (Lean/Common/Util/LeanData.cs:800-816). The profile is derived from the
    symbol rather than passed in, so existing callers need no change.
    """

    resolution: ClassVar[Resolution] = Resolution.DAILY
    time_format: ClassVar[str] = "%Y%m%d 00:00"

    def entries(self, symbol: SymbolSpec, bars: pd.DataFrame) -> Iterator[BarFile]:
        profile = profile_for(symbol.security_type)
        yield BarFile(
            zip_path=PurePosixPath(
                symbol.security_type,
                symbol.market,
                "daily",
                f"{symbol.key}{profile.zip_suffix}.zip",
            ),
            entry_name=f"{symbol.key}.csv",
            payload=_render_rows(bars, self.time_format, profile),
        )
```

- [ ] **Step 4: Modify `leandata/lean/writer.py`**

Add to the imports:

```python
from .profiles import profile_for
```

Change `WriteReport` so the two auxiliary paths are optional and `describe`
skips them:

```python
@dataclass(frozen=True)
class WriteReport:
    bar_files: tuple[Path, ...]
    factor_file: Path | None
    map_file: Path | None
    bar_count: int
    first_date: date
    last_date: date
    factor_rows: int
    warnings: tuple[str, ...]

    def describe(self) -> str:
        lines = [
            f"bars      {self.bar_count} rows  {self.first_date} -> {self.last_date}",
        ]
        # Crypto has neither file; see leandata/lean/profiles.py.
        if self.factor_file is not None:
            lines.append(f"factors   {self.factor_rows} rows  {self.factor_file}")
        if self.map_file is not None:
            lines.append(f"map       {self.map_file}")
        lines.extend(f"data      {path}" for path in self.bar_files)
        lines.extend(f"warning   {message}" for message in self.warnings)
        return "\n".join(lines)
```

Replace the body of `write` down to the `return` with:

```python
    def write(self, history: SecurityHistory) -> WriteReport:
        profile = profile_for(history.symbol.security_type)
        layout = layout_for(history.resolution)

        bar_paths = []
        for entry in layout.entries(history.symbol, history.bars):
            bar_paths.append(self._write_zip(entry))

        factor_path: Path | None = None
        map_path: Path | None = None
        factor_rows: tuple = ()
        warnings: tuple = ()
        if profile.writes_corporate_actions:
            factor_rows, warnings = build_factor_rows(
                history.bars, history.dividends, history.splits
            )
            factor_path = self._write_bytes(
                self._auxiliary_path(history.symbol, "factor_files"),
                render_factor_file(factor_rows),
            )
            map_path = self._write_bytes(
                self._auxiliary_path(history.symbol, "map_files"),
                render_map_file(build_map_rows(history.symbol, history.first_date)),
            )
        if history.provenance is not None:
            self._write_provenance(history)

        return WriteReport(
            bar_files=tuple(bar_paths),
            factor_file=factor_path,
            map_file=map_path,
            bar_count=len(history.bars),
            first_date=history.first_date,
            last_date=history.last_date,
            factor_rows=len(factor_rows),
            warnings=tuple(warnings),
        )
```

- [ ] **Step 5: Run the new test to verify it passes**

Run: `cd lean_strategies && uv run pytest tests/test_crypto_lean_files.py -v`
Expected: PASS, 6 tests

- [ ] **Step 6: Run the whole suite — the equity path must be unchanged**

Run: `cd lean_strategies && uv run pytest`
Expected: PASS, no failures. `tests/test_lean_files.py` and
`tests/test_factor_cross_validation.py` are the ones that would catch a
regression in equity output.

- [ ] **Step 7: Commit**

```bash
cd lean_strategies
git add leandata/lean/bars.py leandata/lean/writer.py tests/test_crypto_lean_files.py
git commit -m "feat: write crypto bars in LEAN's unscaled, suffixed format"
```

---

### Task 3: Reading a crypto data folder back

**Files:**
- Modify: `lean_strategies/leandata/lean/overlay.py`
- Test: `lean_strategies/tests/test_crypto_lean_files.py` (append)

**Interfaces:**
- Consumes: `profile_for` (Task 1), `LeanDataWriter` (Task 2).
- Produces:
  - `overlay.read_daily_csv(archive, *, price_scaled: bool = True)` — the
    default keeps every existing caller correct.
  - `overlay.describe(root, *, market="usa", security_type="equity")` gains a
    `SeriesInfo` entry per crypto archive and a `fred` tuple. Its signature
    keeps its defaults, but the returned `OverlayInfo` gains
    `crypto: tuple[SeriesInfo, ...]` and `fred: tuple[SeriesInfo, ...]`.

Note: `OverlayInfo.fred` is populated in Task 7 once the FRED writer exists.
Here it is added as an always-empty field so the dataclass does not change
shape twice.

- [ ] **Step 1: Write the failing test**

Append to `lean_strategies/tests/test_crypto_lean_files.py`:

```python
from leandata.lean import overlay


def test_read_daily_csv_does_not_unscale_crypto(tmp_path):
    LeanDataWriter(tmp_path).write(crypto_history())
    archive = tmp_path / "crypto" / "bitfinex" / "daily" / "btcusd_trade.zip"
    rows = overlay.read_daily_csv(archive, price_scaled=False)
    assert rows[0][4] == pytest.approx(433.98)
    assert rows[0][5] == pytest.approx(13077.66890809)


def test_read_daily_csv_still_unscales_equity_by_default(tmp_path):
    # Guards the default: every existing caller relies on it.
    LeanDataWriter(tmp_path).write(crypto_history())
    archive = tmp_path / "crypto" / "bitfinex" / "daily" / "btcusd_trade.zip"
    rows = overlay.read_daily_csv(archive)
    assert rows[0][4] == pytest.approx(433.98 / 10_000)


def test_describe_reports_crypto_alongside_equity(tmp_path):
    LeanDataWriter(tmp_path).write(crypto_history())
    info = overlay.describe(tmp_path)
    assert [entry.ticker for entry in info.crypto] == ["BTCUSD"]
    assert info.crypto[0].rows == 3
    assert "BTCUSD" in info.describe()
    assert "3 rows" in info.describe()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd lean_strategies && uv run pytest tests/test_crypto_lean_files.py -v`
Expected: FAIL with `TypeError: read_daily_csv() got an unexpected keyword
argument 'price_scaled'`

- [ ] **Step 3: Modify `leandata/lean/overlay.py`**

Add to the imports at the top:

```python
from .profiles import CRYPTO, EQUITY
```

Replace `read_daily_csv` with:

```python
def read_daily_csv(
    archive: Path, *, price_scaled: bool = True
) -> list[tuple[date, float, float, float, float, float]]:
    """Read a LEAN daily bar zip back into unscaled (date, o, h, l, c, volume).

    Used by the verify command and the dataset comparison, so both read the
    same bytes LEAN reads rather than trusting what the writer thought it
    wrote.

    ``price_scaled`` mirrors ``AssetProfile.price_scaled``. It defaults to True
    because every caller that predates crypto reads equity files, where prices
    are deci-cent integers. Passing False for a crypto archive is what stops
    a 433.98 BTC close being reported as 0.043398.
    """
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        if not names:
            return []
        payload = zipped.read(names[0]).decode("ascii")

    divisor = 10_000.0 if price_scaled else 1.0
    rows = []
    for line in payload.splitlines():
        if not line.strip():
            continue
        stamp, open_, high, low, close, volume = line.split(",")
        rows.append(
            (
                datetime.strptime(stamp, "%Y%m%d %H:%M").date(),
                float(open_) / divisor,
                float(high) / divisor,
                float(low) / divisor,
                float(close) / divisor,
                float(volume),
            )
        )
    return rows
```

Replace `OverlayInfo` and `describe` with:

```python
@dataclass(frozen=True)
class OverlayInfo:
    root: Path
    missing_reference: tuple[PurePosixPath, ...]
    series: tuple[SeriesInfo, ...]
    crypto: tuple[SeriesInfo, ...] = ()
    fred: tuple[SeriesInfo, ...] = ()

    def describe(self) -> str:
        lines = [f"data folder {self.root}"]
        if self.missing_reference:
            missing = ", ".join(str(path) for path in self.missing_reference)
            lines.append(f"  MISSING reference data: {missing}")
        for entry in (*self.series, *self.crypto, *self.fred):
            lines.append(
                f"  {entry.ticker:<8} {entry.rows:>6} rows  {entry.first_date} -> {entry.last_date}"
            )
        if not self.series and not self.crypto and not self.fred:
            lines.append("  no daily data")
        return "\n".join(lines)


def _daily_series(root: Path, security_type: str, market: str, price_scaled: bool,
                  suffix: str) -> list[SeriesInfo]:
    daily = root / security_type / market / "daily"
    found: list[SeriesInfo] = []
    if not daily.is_dir():
        return found
    for archive in sorted(daily.glob("*.zip")):
        frame = read_daily_csv(archive, price_scaled=price_scaled)
        if not frame:
            continue
        ticker = archive.stem
        if suffix and ticker.endswith(suffix):
            ticker = ticker[: -len(suffix)]
        found.append(
            SeriesInfo(
                ticker=ticker.upper(),
                rows=len(frame),
                first_date=frame[0][0],
                last_date=frame[-1][0],
            )
        )
    return found


def describe(root: Path, *, market: str = "usa", security_type: str = "equity") -> OverlayInfo:
    """Report what a data folder holds.

    ``market`` and ``security_type`` still select the equity folder to scan, so
    the existing callers behave as before. Crypto is scanned separately across
    every market directory, because a folder can carry more than one exchange.
    """
    root = Path(root)
    crypto_root = root / CRYPTO.security_type
    crypto: list[SeriesInfo] = []
    if crypto_root.is_dir():
        for exchange in sorted(path.name for path in crypto_root.iterdir() if path.is_dir()):
            crypto.extend(
                _daily_series(root, CRYPTO.security_type, exchange,
                              CRYPTO.price_scaled, CRYPTO.zip_suffix)
            )

    return OverlayInfo(
        root=root,
        missing_reference=tuple(missing_reference_data(root)),
        series=tuple(_daily_series(root, security_type, market,
                                   EQUITY.price_scaled, EQUITY.zip_suffix)),
        crypto=tuple(crypto),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd lean_strategies && uv run pytest tests/test_crypto_lean_files.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Run the whole suite**

Run: `cd lean_strategies && uv run pytest`
Expected: PASS. If `tests/test_lean_files.py::test_read_daily_csv_unscales_prices`
fails, the default on `price_scaled` was changed by mistake.

- [ ] **Step 6: Commit**

```bash
cd lean_strategies
git add leandata/lean/overlay.py tests/test_crypto_lean_files.py
git commit -m "feat: read and report crypto archives without unscaling them"
```

---

### Task 4: `ValueSeries` and the custom-data writer

**Files:**
- Modify: `lean_strategies/leandata/model.py` (append)
- Create: `lean_strategies/leandata/lean/customdata.py`
- Test: `lean_strategies/tests/test_value_series.py`

**Interfaces:**
- Consumes: `leandata.errors.ValidationError`, `leandata.model.Provenance`.
- Produces:
  - `leandata.model.ValueSeries` — frozen dataclass, fields `key: str`,
    `values: pd.Series`, `provenance: Provenance | None = None`, with
    properties `first_date -> date` and `last_date -> date`
  - `leandata.lean.customdata.CUSTOM_DATA_ROOT: PurePosixPath` = `alternative/fred`
  - `leandata.lean.customdata.render_value_series(series: ValueSeries) -> bytes`
  - `leandata.lean.customdata.write_value_series(root: Path, series: ValueSeries) -> Path`
  - `leandata.lean.customdata.read_value_series(path: Path) -> list[tuple[date, float]]`

- [ ] **Step 1: Write the failing test**

Create `lean_strategies/tests/test_value_series.py`:

```python
"""The canonical type for a single-valued time series, and its LEAN file.

FRED series are not OHLCV, so they do not go through SecurityHistory. The
constraint that matters most here is that values may be negative or zero:
measured over 2016-01-01 to 2026-08-31, T10Y3M ranges from -1.89 to 2.27, is
negative on 738 rows and exactly 0 on 11.
"""

from datetime import date

import pandas as pd
import pytest

from leandata.errors import ValidationError
from leandata.lean.customdata import (
    read_value_series,
    render_value_series,
    write_value_series,
)
from leandata.model import ValueSeries


def series(values, start="2016-01-04"):
    index = pd.DatetimeIndex(pd.bdate_range(start=start, periods=len(values)), name="time")
    return pd.Series([float(v) for v in values], index=index, dtype="float64")


def test_negative_and_zero_values_are_accepted():
    built = ValueSeries(key="t10y3m", values=series([-1.89, 0.0, 2.27]))
    assert built.values.iloc[0] == -1.89
    assert built.first_date == date(2016, 1, 4)
    assert built.last_date == date(2016, 1, 6)


def test_a_timezone_aware_index_is_rejected():
    aware = series([1.0, 2.0]).tz_localize("UTC")
    with pytest.raises(ValidationError, match="tz-naive"):
        ValueSeries(key="vixcls", values=aware)


def test_an_unsorted_index_is_rejected():
    unsorted = series([1.0, 2.0]).iloc[::-1]
    with pytest.raises(ValidationError, match="sorted"):
        ValueSeries(key="vixcls", values=unsorted)


def test_a_duplicated_index_is_rejected():
    doubled = pd.concat([series([1.0]), series([2.0])])
    with pytest.raises(ValidationError, match="unique"):
        ValueSeries(key="vixcls", values=doubled)


def test_a_nan_value_is_rejected():
    with pytest.raises(ValidationError, match="NaN"):
        ValueSeries(key="vixcls", values=series([1.0, float("nan")]))


def test_an_empty_series_is_rejected():
    empty = pd.Series([], index=pd.DatetimeIndex([], name="time"), dtype="float64")
    with pytest.raises(ValidationError, match="empty"):
        ValueSeries(key="vixcls", values=empty)


def test_rendered_rows_are_yyyymmdd_and_lf_separated():
    built = ValueSeries(key="t10y3m", values=series([-1.89, 0.0, 2.27]))
    assert render_value_series(built) == (
        b"20160104,-1.89\n20160105,0\n20160106,2.27"
    )


def test_write_lands_under_alternative_fred(tmp_path):
    built = ValueSeries(key="VIXCLS", values=series([20.7, 19.34]))
    written = write_value_series(tmp_path, built)
    assert written == tmp_path / "alternative" / "fred" / "vixcls.csv"
    assert written.read_bytes() == b"20160104,20.7\n20160105,19.34"


def test_round_trip_through_read(tmp_path):
    built = ValueSeries(key="t10y3m", values=series([-1.89, 0.0, 2.27]))
    written = write_value_series(tmp_path, built)
    assert read_value_series(written) == [
        (date(2016, 1, 4), -1.89),
        (date(2016, 1, 5), 0.0),
        (date(2016, 1, 6), 2.27),
    ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd lean_strategies && uv run pytest tests/test_value_series.py -v`
Expected: FAIL, `ImportError: cannot import name 'ValueSeries'`

- [ ] **Step 3: Append `ValueSeries` to `leandata/model.py`**

Add at the end of the file:

```python
@dataclass(frozen=True)
class ValueSeries:
    """One number per date -- a macro series, not a traded instrument.

    FRED's VIXCLS, T10Y3M and DFF are observations, not bars: no OHLC, no
    volume, no corporate actions, and no requirement that the number be
    positive. Forcing them into SecurityHistory would mean five identical
    price columns and a validator that rejects half the real data, so they get
    their own narrow type instead.

    ``values.index``
        ``DatetimeIndex``, tz-naive, midnight-normalised, strictly increasing
        and unique.
    ``values``
        float64, finite. **May be negative or zero.**
    """

    key: str
    values: pd.Series
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", self.key.strip().lower())
        validate_value_series(self)

    @property
    def first_date(self) -> date:
        return self.values.index[0].date()

    @property
    def last_date(self) -> date:
        return self.values.index[-1].date()


def validate_value_series(series: ValueSeries) -> None:
    """Raise ValidationError if the contract in ValueSeries is broken."""
    if not series.key:
        raise ValidationError("ValueSeries.key must be a non-empty string")

    values = series.values
    if not isinstance(values, pd.Series):
        raise ValidationError(f"values must be a Series, got {type(values).__name__}")
    if len(values) == 0:
        raise ValidationError("values must not be empty")

    index = values.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValidationError(f"values.index must be a DatetimeIndex, got {type(index).__name__}")
    if index.tz is not None:
        raise ValidationError(f"values.index must be tz-naive, got tz={index.tz}")
    if not index.is_monotonic_increasing:
        raise ValidationError("values.index must be sorted oldest to newest")
    if index.has_duplicates:
        duplicates = index[index.duplicated()][:3].tolist()
        raise ValidationError(f"values.index must be unique; duplicates include {duplicates}")

    numbers = values.to_numpy(dtype="float64", copy=False)
    if not np.isfinite(numbers).all():
        raise ValidationError(
            f"values contains NaN or inf, first at {_first_bad(index, ~np.isfinite(numbers))}"
        )
    # Deliberately no positivity check: T10Y3M is a spread and goes negative.
```

- [ ] **Step 4: Create `leandata/lean/customdata.py`**

```python
"""Writing a ValueSeries into a LEAN data folder as custom data.

LEAN has no built-in type for a FRED series in this repository -- the ``Fred``
class lives in the separate Lean.DataSource.FRED project -- so the algorithm
supplies its own ``PythonData`` subclass and reads these files directly.
That makes the path and the row format a contract between this module and
``algorithms/ai_stocks_bonds_rotation.py``; change one and you must change
the other.

The format is deliberately the same shape as a LEAN daily bar timestamp,
``YYYYMMDD``, so the algorithm's reader can parse it with the same
``strptime`` format string every other LEAN file uses.
"""

from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Final

from ..model import ValueSeries
from .formatting import strip_zeros, to_decimal

CUSTOM_DATA_ROOT: Final = PurePosixPath("alternative/fred")
DATE_FORMAT: Final = "%Y%m%d"
# Bar files are LF separated with no trailing newline; match them.
LINE_SEPARATOR: Final = b"\n"


def render_value_series(series: ValueSeries) -> bytes:
    """``YYYYMMDD,value`` rows, LF separated, no trailing newline."""
    lines = []
    for timestamp, value in series.values.items():
        lines.append(
            f"{timestamp.strftime(DATE_FORMAT)},{strip_zeros(to_decimal(value))}".encode("ascii")
        )
    return LINE_SEPARATOR.join(lines)


def value_series_path(root: Path, key: str) -> Path:
    return Path(root) / CUSTOM_DATA_ROOT / f"{key.lower()}.csv"


def write_value_series(root: Path, series: ValueSeries) -> Path:
    target = value_series_path(root, series.key)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the target and rename, so an interrupted run never leaves a
    # half-written file that the algorithm would silently read as short.
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(render_value_series(series))
    temporary.replace(target)
    return target


def read_value_series(path: Path) -> list[tuple[date, float]]:
    """Read a written file back, so ``check`` reports what LEAN will read."""
    rows = []
    for line in Path(path).read_bytes().decode("ascii").splitlines():
        if not line.strip():
            continue
        stamp, value = line.split(",")
        rows.append((datetime.strptime(stamp, DATE_FORMAT).date(), float(value)))
    return rows
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd lean_strategies && uv run pytest tests/test_value_series.py -v`
Expected: PASS, 9 tests

- [ ] **Step 6: Run the whole suite**

Run: `cd lean_strategies && uv run pytest`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
cd lean_strategies
git add leandata/model.py leandata/lean/customdata.py tests/test_value_series.py
git commit -m "feat: add ValueSeries for single-valued macro series"
```

---

### Task 5: The Bitfinex source

**Files:**
- Create: `lean_strategies/leandata/sources/bitfinex.py`
- Modify: `lean_strategies/leandata/sources/__init__.py` (one `register` line)
- Create: `lean_strategies/tests/bitfinex_fakes.py`
- Test: `lean_strategies/tests/test_bitfinex_source.py`

**Interfaces:**
- Consumes: `BaseSource`, `FetchRequest` from `leandata.sources.base`;
  `SecurityHistory`, `SymbolSpec`, `Provenance`, `Resolution`, `BAR_COLUMNS`
  from `leandata.model`; `profile_for` must **not** be imported here —
  `leandata/sources/` may not import `leandata/lean/`. Hardcode `"UTC"`.
- Produces:
  - `leandata.sources.bitfinex.BitfinexClient` with
    `candles(symbol: str, start_ms: int, end_ms: int, limit: int) -> list[list]`
    and attribute `version: str`
  - `leandata.sources.bitfinex.BitfinexSource(client=None)` implementing the
    `DataSource` protocol; `name = "bitfinex"`
  - Registered under the name `"bitfinex"`

- [ ] **Step 1: Write the fake and the failing test**

Create `lean_strategies/tests/bitfinex_fakes.py`:

```python
"""A stand-in for BitfinexClient.

Injected through the constructor, the way yfinance_fakes.py is, so the
network-marked smoke test still exercises the real client.

The rows here are verbatim from a real response for tBTCUSD, which is what
makes the field-order test meaningful:
    [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]
"""

REAL_ROWS = [
    [1451606400000, 429.17, 433.98, 436.49, 426.26, 13077.668908090154],
    [1451692800000, 433.89, 432.70, 435.80, 430.00, 6347.952930870056],
    [1451779200000, 432.66, 428.39, 433.07, 421.73, 19201.833363170073],
]


class FakeBitfinexClient:
    def __init__(self, pages=None):
        # pages: a list of response lists, returned one per call.
        self.pages = list(pages) if pages is not None else [REAL_ROWS]
        self.calls = []
        self.version = "fake"

    def candles(self, symbol, start_ms, end_ms, limit):
        self.calls.append((symbol, start_ms, end_ms, limit))
        if not self.pages:
            return []
        return self.pages.pop(0)
```

Create `lean_strategies/tests/test_bitfinex_source.py`:

```python
"""The Bitfinex adapter, driven by a fake client.

The field order is the whole point of this file. Bitfinex returns
    [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]
with close third, before high and low. Confirmed empirically over 3889 real
rows: read that way every row satisfies LEAN's OHLC invariants, and read as
[MTS, O, H, L, C, V] 3879 of them do not. SecurityHistory validation alone
would not catch the swap, because on a quiet day high and close are close
enough that high >= max(open, close) still holds.
"""

from datetime import date

import pandas as pd
import pytest
from bitfinex_fakes import REAL_ROWS, FakeBitfinexClient

from leandata.errors import EmptyHistoryError
from leandata.model import BAR_COLUMNS, Resolution
from leandata.sources import get_source
from leandata.sources.base import FetchRequest
from leandata.sources.bitfinex import BitfinexSource


def request(**overrides):
    defaults = dict(
        ticker="BTCUSD",
        resolution=Resolution.DAILY,
        start=date(2016, 1, 1),
        end=date(2016, 1, 3),
        market="bitfinex",
        security_type="crypto",
    )
    defaults.update(overrides)
    return FetchRequest(**defaults)


def test_columns_map_mts_open_close_high_low_volume():
    history = BitfinexSource(FakeBitfinexClient()).fetch(request())
    first = history.bars.iloc[0]
    assert first["open"] == pytest.approx(429.17)
    assert first["close"] == pytest.approx(433.98)
    assert first["high"] == pytest.approx(436.49)
    assert first["low"] == pytest.approx(426.26)
    assert first["volume"] == pytest.approx(13077.668908090154)


def test_bars_have_the_canonical_column_order():
    history = BitfinexSource(FakeBitfinexClient()).fetch(request())
    assert tuple(history.bars.columns) == BAR_COLUMNS


def test_timestamps_become_naive_utc_midnight():
    history = BitfinexSource(FakeBitfinexClient()).fetch(request())
    assert history.bars.index.tz is None
    assert list(history.bars.index) == [
        pd.Timestamp("2016-01-01"),
        pd.Timestamp("2016-01-02"),
        pd.Timestamp("2016-01-03"),
    ]


def test_symbol_is_a_bitfinex_crypto_symbol():
    history = BitfinexSource(FakeBitfinexClient()).fetch(request())
    assert history.symbol.ticker == "BTCUSD"
    assert history.symbol.market == "bitfinex"
    assert history.symbol.security_type == "crypto"
    assert history.symbol.key == "btcusd"


def test_no_dividends_or_splits():
    history = BitfinexSource(FakeBitfinexClient()).fetch(request())
    assert history.dividends.empty
    assert history.splits.empty


def test_the_trading_pair_is_prefixed_with_t():
    client = FakeBitfinexClient()
    BitfinexSource(client).fetch(request())
    assert client.calls[0][0] == "tBTCUSD"


def test_paging_stops_when_a_page_comes_back_short():
    client = FakeBitfinexClient(pages=[REAL_ROWS, []])
    BitfinexSource(client).fetch(request())
    # One full page is shorter than the limit, so no second call is needed.
    assert len(client.calls) == 1


def test_paging_continues_while_pages_come_back_full():
    later = [[1451865600000, 428.39, 433.09, 434.0, 427.0, 100.0]]
    client = FakeBitfinexClient(pages=[REAL_ROWS, later, []])
    source = BitfinexSource(client)
    source.page_limit = len(REAL_ROWS)  # force a full first page
    history = source.fetch(request(end=date(2016, 1, 4)))
    assert len(client.calls) == 2
    assert len(history.bars) == 4


def test_duplicate_timestamps_across_pages_are_dropped():
    client = FakeBitfinexClient(pages=[REAL_ROWS, REAL_ROWS[-1:], []])
    source = BitfinexSource(client)
    source.page_limit = len(REAL_ROWS)
    history = source.fetch(request())
    assert len(history.bars) == 3


def test_an_empty_response_raises_rather_than_writing_nothing():
    with pytest.raises(EmptyHistoryError):
        BitfinexSource(FakeBitfinexClient(pages=[[]])).fetch(request())


def test_only_daily_is_supported():
    from leandata.errors import UnsupportedResolutionError

    with pytest.raises(UnsupportedResolutionError):
        BitfinexSource(FakeBitfinexClient()).fetch(request(resolution=Resolution.HOUR))


def test_registered_under_its_name():
    assert isinstance(get_source("bitfinex"), BitfinexSource)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd lean_strategies && uv run pytest tests/test_bitfinex_source.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'leandata.sources.bitfinex'`

- [ ] **Step 3: Create `leandata/sources/bitfinex.py`**

```python
"""Bitfinex public candles adapter.

Split into a client that talks to the network and a source that only
transforms rows, so the tests drive it with a hand-written fake and never
touch the exchange.

**The field order is the thing to get right.** A candle row is

    [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]

with close third, before high and low. This was confirmed against 3889 real
rows: read that way every row satisfies LEAN's OHLC invariants, and read as
the more familiar [MTS, O, H, L, C, V] 3879 of them do not. The model's
validation would not reliably catch the swap on its own, so
tests/test_bitfinex_source.py asserts the mapping directly.

Timestamps are UTC midnight and stay UTC. LEAN's market-hours entry
``Crypto-bitfinex-[*]`` has both dataTimeZone and exchangeTimeZone set to UTC,
so converting to New York the way the equity adapter does would shift every
bar onto the wrong day.

There are no dividends or splits: crypto has no corporate actions, and the
writer skips factor and map files for this asset class anyway.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import ClassVar

import pandas as pd

from ..errors import SourceError
from ..model import (
    BAR_COLUMNS,
    Provenance,
    Resolution,
    SecurityHistory,
    SymbolSpec,
    empty_dividends,
    empty_splits,
)
from .base import BaseSource, FetchRequest

API_ROOT = "https://api-pub.bitfinex.com/v2"
# The documented ceiling for this endpoint. One request at this limit covers
# 2016-01-01 to 2026-08-31 (3889 rows), so paging is a guard, not a routine
# path -- but a longer window or a finer resolution would need it.
MAX_LIMIT = 10_000
TIMEFRAMES = {Resolution.DAILY: "1D"}


class BitfinexClient:
    """The only part of this module that makes a network request."""

    def __init__(self, *, timeout: float = 60.0) -> None:
        self.timeout = timeout
        self.version = "v2"

    def candles(self, symbol: str, start_ms: int, end_ms: int, limit: int) -> list:
        query = urllib.parse.urlencode(
            {"start": start_ms, "end": end_ms, "limit": limit, "sort": 1}
        )
        url = f"{API_ROOT}/candles/trade:1D:{symbol}/hist?{query}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise SourceError(f"bitfinex request failed for {symbol}: {exc}") from exc

        try:
            rows = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SourceError(f"bitfinex returned non-JSON for {symbol}: {payload[:120]!r}") from exc

        # An error comes back as ["error", code, message] rather than a status.
        if rows and isinstance(rows[0], str):
            raise SourceError(f"bitfinex error for {symbol}: {rows}")
        return rows


class BitfinexSource(BaseSource):
    name: ClassVar[str] = "bitfinex"

    def __init__(self, client: BitfinexClient | None = None) -> None:
        self._client = client
        self.page_limit = MAX_LIMIT
        self.warnings: list[str] = []

    @property
    def client(self) -> BitfinexClient:
        if self._client is None:
            self._client = BitfinexClient()
        return self._client

    def supported_resolutions(self) -> frozenset[Resolution]:
        return frozenset({Resolution.DAILY})

    def fetch(self, request: FetchRequest) -> SecurityHistory:
        self._check_resolution(request.resolution)
        rows = self._all_rows(request)
        self._check_not_empty(rows, request)

        bars = _to_bars(rows)
        self._check_not_empty(bars, request)
        self.warnings = []

        history = SecurityHistory(
            symbol=SymbolSpec(
                ticker=request.ticker,
                market=request.market or "bitfinex",
                security_type=request.security_type or "crypto",
            ),
            resolution=request.resolution,
            bars=bars,
            dividends=empty_dividends(),
            splits=empty_splits(),
            provenance=Provenance(
                source=self.name,
                fetched_at=datetime.now(timezone.utc),
                source_version=self.client.version,
                options={"pair": _pair(request.ticker), "timeframe": TIMEFRAMES[request.resolution]},
            ),
        )
        return history.clip(request.start, request.end)

    def _all_rows(self, request: FetchRequest) -> list:
        """Page forward until a short page arrives or the window is covered."""
        pair = _pair(request.ticker)
        start_ms = _to_ms(request.start or date(2010, 1, 1))
        end_ms = _to_ms((request.end or date.today()) + timedelta(days=1)) - 1

        collected: list = []
        seen: set[int] = set()
        while start_ms <= end_ms:
            page = self.client.candles(pair, start_ms, end_ms, self.page_limit)
            if not page:
                break
            fresh = [row for row in page if row[0] not in seen]
            seen.update(row[0] for row in fresh)
            collected.extend(fresh)
            if len(page) < self.page_limit:
                break
            start_ms = max(row[0] for row in page) + 1
        return collected


def _pair(ticker: str) -> str:
    """Bitfinex names trading pairs tBTCUSD; funding symbols use f."""
    return f"t{ticker.strip().upper()}"


def _to_ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)


def _to_bars(rows: list) -> pd.DataFrame:
    """[MTS, OPEN, CLOSE, HIGH, LOW, VOLUME] -> the canonical bar frame."""
    if not rows:
        return pd.DataFrame(columns=list(BAR_COLUMNS), dtype="float64")

    index = pd.DatetimeIndex(
        [datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc).replace(tzinfo=None) for row in rows],
        name="time",
    ).normalize()
    frame = pd.DataFrame(
        {
            "open": [float(row[1]) for row in rows],
            "high": [float(row[3]) for row in rows],
            "low": [float(row[4]) for row in rows],
            "close": [float(row[2]) for row in rows],
            "volume": [abs(float(row[5])) for row in rows],
        },
        index=index,
    )
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame[list(BAR_COLUMNS)].astype("float64")
```

- [ ] **Step 4: Register the source**

In `leandata/sources/__init__.py`, add after the two existing `register` calls:

```python
register("bitfinex", lambda **options: _module("bitfinex").BitfinexSource(**options))
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd lean_strategies && uv run pytest tests/test_bitfinex_source.py -v`
Expected: PASS, 12 tests

- [ ] **Step 6: Add the network smoke test**

Create `lean_strategies/tests/test_bitfinex_network.py`:

```python
"""One real call to Bitfinex. Deselected by default; see pyproject.toml."""

from datetime import date

import pytest

from leandata.model import Resolution
from leandata.sources.base import FetchRequest
from leandata.sources.bitfinex import BitfinexSource


@pytest.mark.network
def test_a_real_window_comes_back_valid():
    history = BitfinexSource().fetch(
        FetchRequest(
            ticker="BTCUSD",
            resolution=Resolution.DAILY,
            start=date(2016, 1, 1),
            end=date(2016, 1, 31),
            market="bitfinex",
            security_type="crypto",
        )
    )
    # SecurityHistory validates OHLC invariants on construction, so reaching
    # here at all means the field mapping held against live data.
    assert len(history.bars) >= 28
    assert history.first_date == date(2016, 1, 1)
    assert history.bars["close"].iloc[0] == pytest.approx(433.98, rel=1e-6)
```

- [ ] **Step 7: Run the network test once**

Run: `cd lean_strategies && uv run pytest tests/test_bitfinex_network.py -m network -v`
Expected: PASS

- [ ] **Step 8: Run the whole offline suite**

Run: `cd lean_strategies && uv run pytest`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
cd lean_strategies
git add leandata/sources/bitfinex.py leandata/sources/__init__.py \
        tests/bitfinex_fakes.py tests/test_bitfinex_source.py tests/test_bitfinex_network.py
git commit -m "feat: add a Bitfinex candles source for BTCUSD"
```

---

### Task 6: The FRED source

**Files:**
- Create: `lean_strategies/leandata/sources/fred.py`
- Modify: `lean_strategies/leandata/sources/__init__.py` (one `register` line)
- Create: `lean_strategies/tests/fred_fakes.py`
- Test: `lean_strategies/tests/test_fred_source.py`, `tests/test_fred_network.py`

**Interfaces:**
- Consumes: `ValueSeries`, `Provenance` (Task 4); `BaseSource`, `FetchRequest`.
- Produces:
  - `leandata.sources.fred.FredClient.series_csv(series_id, start, end) -> str`
  - `leandata.sources.fred.FredSource(client=None)` with
    `fetch_series(request: FetchRequest) -> ValueSeries`; `name = "fred"`
  - Registered under `"fred"`

`FredSource` deliberately does **not** implement `fetch` returning a
`SecurityHistory`; it returns a `ValueSeries` from `fetch_series`. The CLI's
new `convert-series` command calls that method, so it never flows through
`LeanDataWriter`.

- [ ] **Step 1: Write the fake and the failing test**

Create `lean_strategies/tests/fred_fakes.py`:

```python
"""A stand-in for FredClient.

The CSV shape here is verbatim from fredgraph.csv. Note the missing
observation is an EMPTY FIELD, not the "." that FRED's older text format
uses: over 2016-01-01 to 2026-08-31, VIXCLS has 69 empty rows and T10Y3M 115.
"""

VIXCLS_CSV = "\n".join(
    [
        "observation_date,VIXCLS",
        "2016-01-04,20.70",
        "2016-01-05,19.34",
        "2016-01-06,",
        "2016-01-07,24.99",
    ]
)

T10Y3M_CSV = "\n".join(
    [
        "observation_date,T10Y3M",
        "2016-01-04,2.02",
        "2016-01-05,0.00",
        "2016-01-06,-1.89",
    ]
)


class FakeFredClient:
    def __init__(self, payloads=None):
        self.payloads = dict(payloads or {"VIXCLS": VIXCLS_CSV, "T10Y3M": T10Y3M_CSV})
        self.calls = []
        self.version = "fake"

    def series_csv(self, series_id, start, end):
        self.calls.append((series_id, start, end))
        return self.payloads[series_id]
```

Create `lean_strategies/tests/test_fred_source.py`:

```python
"""The FRED adapter, driven by a fake client."""

from datetime import date

import pytest
from fred_fakes import FakeFredClient

from leandata.errors import EmptyHistoryError
from leandata.model import Resolution, ValueSeries
from leandata.sources import get_source
from leandata.sources.base import FetchRequest
from leandata.sources.fred import FredSource


def request(ticker="VIXCLS", **overrides):
    defaults = dict(
        ticker=ticker,
        resolution=Resolution.DAILY,
        start=date(2016, 1, 1),
        end=date(2016, 1, 31),
    )
    defaults.update(overrides)
    return FetchRequest(**defaults)


def test_returns_a_value_series_keyed_by_lowercase_series_id():
    built = FredSource(FakeFredClient()).fetch_series(request())
    assert isinstance(built, ValueSeries)
    assert built.key == "vixcls"


def test_empty_observations_are_dropped_not_filled():
    built = FredSource(FakeFredClient()).fetch_series(request())
    assert len(built.values) == 3
    assert date(2016, 1, 6) not in [stamp.date() for stamp in built.values.index]


def test_negative_and_zero_values_survive():
    built = FredSource(FakeFredClient()).fetch_series(request("T10Y3M"))
    assert list(built.values) == [2.02, 0.0, -1.89]


def test_the_window_is_passed_through_as_cosd_and_coed():
    client = FakeFredClient()
    FredSource(client).fetch_series(request())
    assert client.calls[0] == ("VIXCLS", date(2016, 1, 1), date(2016, 1, 31))


def test_a_series_with_no_usable_rows_raises():
    payload = "observation_date,VIXCLS\n2016-01-04,\n2016-01-05,"
    with pytest.raises(EmptyHistoryError):
        FredSource(FakeFredClient({"VIXCLS": payload})).fetch_series(request())


def test_a_dot_marker_is_also_treated_as_missing():
    # FRED's older text format uses "."; tolerate it so a future endpoint
    # change does not turn a gap into a crash.
    payload = "observation_date,VIXCLS\n2016-01-04,20.70\n2016-01-05,.\n"
    built = FredSource(FakeFredClient({"VIXCLS": payload})).fetch_series(request())
    assert len(built.values) == 1


def test_registered_under_its_name():
    assert isinstance(get_source("fred"), FredSource)
```

Create `lean_strategies/tests/test_fred_network.py`:

```python
"""One real call to FRED. Deselected by default; see pyproject.toml."""

from datetime import date

import pytest

from leandata.model import Resolution
from leandata.sources.base import FetchRequest
from leandata.sources.fred import FredSource


@pytest.mark.network
def test_t10y3m_really_does_go_negative():
    built = FredSource().fetch_series(
        FetchRequest(
            ticker="T10Y3M",
            resolution=Resolution.DAILY,
            start=date(2022, 1, 1),
            end=date(2023, 12, 31),
        )
    )
    assert (built.values < 0).any(), "the 2022-2023 inversion should be present"
    assert built.first_date >= date(2022, 1, 1)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd lean_strategies && uv run pytest tests/test_fred_source.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'leandata.sources.fred'`

- [ ] **Step 3: Create `leandata/sources/fred.py`**

```python
"""Federal Reserve Economic Data adapter.

Free and unauthenticated: fredgraph.csv needs no API key. This is the same
origin QuantConnect's own FRED dataset draws from, which is why the local run
and the cloud run should agree on the factor inputs even though they disagree
about everything else.

Two details that are easy to get wrong:

**A missing observation is an empty field, not a dot.** ``fredgraph.csv``
emits ``2016-01-18,`` for a market holiday. The ``.`` marker belongs to
FRED's older text format; this module tolerates both, but the empty field is
what actually arrives. Measured over 2016-01-01 to 2026-08-31: VIXCLS has 69
empty rows out of 2780, T10Y3M has 115 out of 2781, DFF has none out of 3893
because it publishes on weekends too.

**Values may be negative or zero.** T10Y3M is a yield spread; over that same
window it ranges from -1.89 to 2.27, is negative on 738 rows and exactly 0 on
11. That is why these become a ValueSeries rather than anything that reuses
the price validators.

Gaps are dropped rather than filled. The algorithm calls ``.dropna()`` on the
factor frame and intersects it with the label index, so a gap costs one
training row; a fabricated observation would instead be trained on.
"""

import csv
import io
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from typing import ClassVar

import pandas as pd

from ..errors import EmptyHistoryError, SourceError, UnsupportedResolutionError
from ..model import Provenance, Resolution, ValueSeries
from .base import BaseSource, FetchRequest

FREDGRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
MISSING_MARKERS = frozenset({"", ".", "NA", "nan"})


class FredClient:
    """The only part of this module that makes a network request."""

    def __init__(self, *, timeout: float = 60.0) -> None:
        self.timeout = timeout
        self.version = "fredgraph"

    def series_csv(self, series_id: str, start: date | None, end: date | None) -> str:
        parameters = {"id": series_id}
        if start is not None:
            parameters["cosd"] = start.isoformat()
        if end is not None:
            parameters["coed"] = end.isoformat()
        url = f"{FREDGRAPH_URL}?{urllib.parse.urlencode(parameters)}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                return response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise SourceError(f"FRED request failed for {series_id}: {exc}") from exc


class FredSource(BaseSource):
    name: ClassVar[str] = "fred"

    def __init__(self, client: FredClient | None = None) -> None:
        self._client = client
        self.warnings: list[str] = []

    @property
    def client(self) -> FredClient:
        if self._client is None:
            self._client = FredClient()
        return self._client

    def supported_resolutions(self) -> frozenset[Resolution]:
        return frozenset({Resolution.DAILY})

    def fetch(self, request: FetchRequest):
        """FRED series are not securities. Use ``fetch_series``."""
        raise UnsupportedResolutionError(
            "FRED returns a ValueSeries, not a SecurityHistory; call fetch_series"
        )

    def fetch_series(self, request: FetchRequest) -> ValueSeries:
        self._check_resolution(request.resolution)
        series_id = request.ticker.strip().upper()
        payload = self.client.series_csv(series_id, request.start, request.end)

        stamps, values, skipped = [], [], 0
        reader = csv.reader(io.StringIO(payload))
        header = next(reader, None)
        if header is None or len(header) < 2:
            raise SourceError(f"FRED returned no header for {series_id}: {payload[:120]!r}")
        for row in reader:
            if len(row) < 2:
                continue
            raw = row[1].strip()
            if raw in MISSING_MARKERS:
                skipped += 1
                continue
            stamps.append(datetime.strptime(row[0].strip(), "%Y-%m-%d"))
            values.append(float(raw))

        if not values:
            window = f"{request.start or 'start'} -> {request.end or 'today'}"
            raise EmptyHistoryError(
                f"source 'fred' returned no usable observations for {series_id} over {window}"
            )

        self.warnings = (
            [f"dropped {skipped} observation(s) FRED reported as missing"] if skipped else []
        )
        index = pd.DatetimeIndex(stamps, name="time").normalize()
        frame = pd.Series(values, index=index, dtype="float64")
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()

        return ValueSeries(
            key=series_id,
            values=frame,
            provenance=Provenance(
                source=self.name,
                fetched_at=datetime.now(timezone.utc),
                source_version=self.client.version,
                options={"series_id": series_id},
            ),
        )
```

- [ ] **Step 4: Register the source**

In `leandata/sources/__init__.py`, add:

```python
register("fred", lambda **options: _module("fred").FredSource(**options))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd lean_strategies && uv run pytest tests/test_fred_source.py -v`
Expected: PASS, 7 tests

- [ ] **Step 6: Run the network test once**

Run: `cd lean_strategies && uv run pytest tests/test_fred_network.py -m network -v`
Expected: PASS

- [ ] **Step 7: Run the whole offline suite**

Run: `cd lean_strategies && uv run pytest`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
cd lean_strategies
git add leandata/sources/fred.py leandata/sources/__init__.py \
        tests/fred_fakes.py tests/test_fred_source.py tests/test_fred_network.py
git commit -m "feat: add a FRED source for macro factor series"
```

---

### Task 7: Wire both into the CLI

**Files:**
- Modify: `lean_strategies/leandata/cli.py`
- Modify: `lean_strategies/leandata/lean/overlay.py` (populate `OverlayInfo.fred`)
- Test: `lean_strategies/tests/test_cli_crypto_and_series.py`

**Interfaces:**
- Consumes: everything from Tasks 1 to 6.
- Produces:
  - `convert --security-type {equity,crypto}` (default `equity`)
  - `convert-series --source fred --ticker VIXCLS --start ... --end ...`
  - `check` listing equity, crypto and FRED entries
  - `overlay.describe` populating `OverlayInfo.fred`

- [ ] **Step 1: Write the failing test**

Create `lean_strategies/tests/test_cli_crypto_and_series.py`:

```python
"""The two new CLI paths, driven end to end with fake sources."""

import pandas as pd
import pytest
from bitfinex_fakes import FakeBitfinexClient
from fred_fakes import FakeFredClient

from leandata import cli, sources
from leandata.lean import overlay
from leandata.sources.bitfinex import BitfinexSource
from leandata.sources.fred import FredSource


@pytest.fixture
def fake_sources():
    """Swap the registry entries for fakes, then put the real ones back."""
    original = dict(sources._REGISTRY)
    sources.register("bitfinex", lambda **_: BitfinexSource(FakeBitfinexClient()))
    sources.register("fred", lambda **_: FredSource(FakeFredClient()))
    yield
    sources._REGISTRY.clear()
    sources._REGISTRY.update(original)


def test_convert_writes_a_crypto_archive(tmp_path, fake_sources, capsys):
    code = cli.main([
        "convert", "--source", "bitfinex", "--ticker", "BTCUSD",
        "--security-type", "crypto", "--market", "bitfinex",
        "--start", "2016-01-01", "--end", "2016-01-03",
        "--out", str(tmp_path), "--no-overlay",
    ])
    assert code == 0 or "missing reference data" in capsys.readouterr().err
    assert (tmp_path / "crypto" / "bitfinex" / "daily" / "btcusd_trade.zip").exists()
    assert not (tmp_path / "crypto" / "bitfinex" / "factor_files").exists()


def test_convert_series_writes_a_fred_file(tmp_path, fake_sources):
    code = cli.main([
        "convert-series", "--source", "fred", "--ticker", "VIXCLS",
        "--start", "2016-01-01", "--end", "2016-01-31",
        "--out", str(tmp_path),
    ])
    assert code == 0
    written = tmp_path / "alternative" / "fred" / "vixcls.csv"
    assert written.exists()
    assert written.read_bytes() == b"20160104,20.7\n20160105,19.34\n20160107,24.99"


def test_check_reports_crypto_and_fred(tmp_path, fake_sources, capsys):
    cli.main([
        "convert", "--source", "bitfinex", "--ticker", "BTCUSD",
        "--security-type", "crypto", "--market", "bitfinex",
        "--start", "2016-01-01", "--end", "2016-01-03",
        "--out", str(tmp_path), "--no-overlay",
    ])
    cli.main([
        "convert-series", "--source", "fred", "--ticker", "VIXCLS",
        "--start", "2016-01-01", "--end", "2016-01-31",
        "--out", str(tmp_path),
    ])
    capsys.readouterr()

    cli.main(["check", "--out", str(tmp_path)])
    printed = capsys.readouterr().out
    assert "BTCUSD" in printed
    assert "VIXCLS" in printed


def test_describe_lists_fred_series(tmp_path, fake_sources):
    cli.main([
        "convert-series", "--source", "fred", "--ticker", "VIXCLS",
        "--start", "2016-01-01", "--end", "2016-01-31",
        "--out", str(tmp_path),
    ])
    info = overlay.describe(tmp_path)
    assert [entry.ticker for entry in info.fred] == ["VIXCLS"]
    assert info.fred[0].rows == 3
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd lean_strategies && uv run pytest tests/test_cli_crypto_and_series.py -v`
Expected: FAIL, `argparse` errors on the unrecognised `--security-type` and on
the unknown `convert-series` command.

- [ ] **Step 3: Populate `OverlayInfo.fred` in `leandata/lean/overlay.py`**

Add to the imports:

```python
from .customdata import CUSTOM_DATA_ROOT, read_value_series
```

Add this helper next to `_daily_series`:

```python
def _fred_series(root: Path) -> list[SeriesInfo]:
    """The FRED files the algorithm's custom data type will read."""
    folder = root / CUSTOM_DATA_ROOT
    found: list[SeriesInfo] = []
    if not folder.is_dir():
        return found
    for path in sorted(folder.glob("*.csv")):
        rows = read_value_series(path)
        if not rows:
            continue
        found.append(
            SeriesInfo(
                ticker=path.stem.upper(),
                rows=len(rows),
                first_date=rows[0][0],
                last_date=rows[-1][0],
            )
        )
    return found
```

And in `describe`, change the `return` to include it:

```python
    return OverlayInfo(
        root=root,
        missing_reference=tuple(missing_reference_data(root)),
        series=tuple(_daily_series(root, security_type, market,
                                   EQUITY.price_scaled, EQUITY.zip_suffix)),
        crypto=tuple(crypto),
        fred=tuple(_fred_series(root)),
    )
```

- [ ] **Step 4: Modify `leandata/cli.py`**

Update the module docstring's command list:

```python
"""Command line for the data conversion pipeline.

    convert         download a symbol and write it into a LEAN data folder
    convert-series  download a macro series (FRED) into a LEAN data folder
    verify          diff a generated symbol against another LEAN data folder
    check           report whether a data folder is complete enough for a run
    snapshot        save a source's raw output as a replayable CSV pair
"""
```

Add the import:

```python
from .lean.customdata import write_value_series
```

In `build_parser`, add to the `convert` subparser, next to `--market`:

```python
    convert.add_argument(
        "--security-type",
        default="equity",
        choices=["equity", "crypto"],
        help="crypto skips factor and map files and stores prices unscaled",
    )
```

And register the new subcommand, after the `convert` block:

```python
    convert_series = commands.add_parser(
        "convert-series", help="download a macro series into a LEAN data folder"
    )
    convert_series.add_argument("--source", default="fred", choices=available())
    convert_series.add_argument("--ticker", required=True, help="e.g. VIXCLS")
    convert_series.add_argument("--start", type=parse_date, help="inclusive, YYYY-MM-DD")
    convert_series.add_argument("--end", type=parse_date, help="inclusive, YYYY-MM-DD")
    convert_series.add_argument("--out", type=Path, default=DEFAULT_OUT)
    convert_series.add_argument(
        "--source-option", dest="source_options", action="append",
        type=parse_option, default=[], metavar="KEY=VALUE",
    )
```

Change `_fetch` so the request carries the security type:

```python
    request = FetchRequest(
        ticker=args.ticker,
        resolution=Resolution(args.resolution),
        start=args.start,
        end=args.end,
        market=getattr(args, "market", "usa"),
        security_type=getattr(args, "security_type", "equity"),
        exchange_code=getattr(args, "exchange_code", "P"),
    )
```

Add the new command function next to `command_convert`:

```python
def command_convert_series(args) -> int:
    options = dict(args.source_options)
    source = get_source(args.source, **options)
    if not hasattr(source, "fetch_series"):
        print(f"error: source {args.source!r} does not provide a value series",
              file=sys.stderr)
        return 1
    request = FetchRequest(
        ticker=args.ticker,
        resolution=Resolution.DAILY,
        start=args.start,
        end=args.end,
    )
    series = source.fetch_series(request)
    for warning in getattr(source, "warnings", []):
        print(f"warning   {warning}")
    written = write_value_series(args.out, series)
    print(f"series    {len(series.values)} rows  {series.first_date} -> {series.last_date}")
    print(f"wrote     {written}")
    return 0
```

Register it:

```python
COMMANDS = {
    "convert": command_convert,
    "convert-series": command_convert_series,
    "verify": command_verify,
    "check": command_check,
    "snapshot": command_snapshot,
}
```

Finally, make `command_check` tolerate a folder with no equity entries — change
its stale-rate guard from `if info.series:` to:

```python
        dated = [*info.series, *info.crypto, *info.fred]
        if dated:
            print(overlay.stale_rate_warning(rates[1], max(e.last_date for e in dated)) or "", end="")
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd lean_strategies && uv run pytest tests/test_cli_crypto_and_series.py -v`
Expected: PASS, 4 tests

- [ ] **Step 6: Run the whole suite**

Run: `cd lean_strategies && uv run pytest`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
cd lean_strategies
git add leandata/cli.py leandata/lean/overlay.py tests/test_cli_crypto_and_series.py
git commit -m "feat: convert crypto and macro series from the command line"
```

---

### Task 8: Grow the LEAN stubs for the new algorithm

**Files:**
- Modify: `lean_strategies/tests/lean_stubs.py`
- Test: `lean_strategies/tests/test_lean_stubs.py` (append)

**Interfaces:**
- Produces, all exported from the fake `AlgorithmImports` module:
  - `pd` (the real pandas) and `timedelta` — the QuantConnect file uses both
    without importing them, because `AlgorithmImports` re-exports them.
  - `Market` with `BITFINEX = "bitfinex"`
  - `DataNormalizationMode` with `TOTAL_RETURN`, `RAW`, `ADJUSTED`
  - `PortfolioTarget(symbol, weight)` with `.symbol` and `.quantity`
  - `PythonData`, `SubscriptionDataSource`, `SubscriptionTransportMedium`,
    `Globals` — needed at *import* time, because the algorithm's
    `except NameError:` branch defines `class Fred(PythonData)`.
  - On `QCAlgorithm`: `settings`, `schedule`, `date_rules`, `time_rules`,
    `add_crypto`, `add_data`, `history`, `set_warmup`, and a `set_holdings`
    that accepts a list of targets.

- [ ] **Step 1: Write the failing test**

Append to `lean_strategies/tests/test_lean_stubs.py`:

```python
def test_algorithm_imports_exports_pandas_and_timedelta():
    # The QuantConnect file writes pd.Series() and timedelta(...) with no
    # import of its own, because AlgorithmImports re-exports both.
    import AlgorithmImports

    assert AlgorithmImports.pd.__name__ == "pandas"
    assert AlgorithmImports.timedelta(days=1).days == 1


def test_python_data_is_available_for_custom_types():
    import AlgorithmImports

    class Example(AlgorithmImports.PythonData):
        pass

    point = Example()
    point.value = 1.0
    assert point.value == 1.0


def test_portfolio_target_records_symbol_and_weight():
    import AlgorithmImports

    target = AlgorithmImports.PortfolioTarget("SPY", 0.4)
    assert target.symbol == "SPY"
    assert target.quantity == 0.4


def test_set_holdings_accepts_a_list_of_targets():
    import lean_stubs

    algorithm = lean_stubs.QCAlgorithm()
    targets = [lean_stubs.PortfolioTarget("SPY", 0.4), lean_stubs.PortfolioTarget("BND", 0.6)]
    algorithm.set_holdings(targets, True)
    assert algorithm.targets == [("SPY", 0.4), ("BND", 0.6)]
    assert algorithm.liquidated is True


def test_history_returns_whatever_the_test_registered():
    import lean_stubs
    import pandas as pd

    algorithm = lean_stubs.QCAlgorithm()
    frame = pd.DataFrame({"value": [1.0]})
    algorithm.history_responses[("A", "B")] = frame
    assert algorithm.history(["A", "B"], None, None) is frame


def test_schedule_and_rules_are_inert_but_record_the_callback():
    import lean_stubs

    algorithm = lean_stubs.QCAlgorithm()
    called = []
    algorithm.schedule.on(
        algorithm.date_rules.month_start("SPY"),
        algorithm.time_rules.after_market_open("SPY", 1),
        lambda: called.append(1),
    )
    assert len(algorithm.scheduled) == 1
    algorithm.scheduled[0][2]()
    assert called == [1]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd lean_strategies && uv run pytest tests/test_lean_stubs.py -v`
Expected: FAIL, `AttributeError: module 'AlgorithmImports' has no attribute 'pd'`

- [ ] **Step 3: Extend `tests/lean_stubs.py`**

Add to the imports at the top:

```python
import pandas as pd
```

Add these classes above `QCAlgorithm`:

```python
class Market:
    USA = "usa"
    BITFINEX = "bitfinex"


class DataNormalizationMode:
    RAW = "Raw"
    ADJUSTED = "Adjusted"
    TOTAL_RETURN = "TotalReturn"


class PortfolioTarget:
    """LEAN's target carries a quantity; a percent target carries a weight."""

    def __init__(self, symbol, quantity):
        self.symbol = symbol
        self.quantity = quantity


class PythonData:
    """Base for custom data types. Only enough surface to be subclassed."""

    def __init__(self):
        self.symbol = None
        self.time = None
        self.value = 0.0

    def get_source(self, config, date, is_live_mode):
        raise NotImplementedError

    def reader(self, config, line, date, is_live_mode):
        raise NotImplementedError


class SubscriptionTransportMedium:
    LOCAL_FILE = 0
    REMOTE_FILE = 1


class SubscriptionDataSource:
    def __init__(self, source, transport_medium=SubscriptionTransportMedium.LOCAL_FILE):
        self.source = source
        self.transport_medium = transport_medium


class Globals:
    """Stands in for LEAN's static Globals.DataFolder."""

    data_folder = "data"


class _Settings:
    def __init__(self):
        self.daily_precise_end_time = True
        self.seed_initial_prices = False


class _Rule:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _DateRules:
    def month_start(self, symbol=None):
        return _Rule("month_start", (symbol,))

    def week_start(self, symbol=None):
        return _Rule("week_start", (symbol,))

    def every_day(self, symbol=None):
        return _Rule("every_day", (symbol,))


class _TimeRules:
    def after_market_open(self, symbol, minutes=0):
        return _Rule("after_market_open", (symbol, minutes))

    def at(self, hour, minute=0):
        return _Rule("at", (hour, minute))


class _Schedule:
    def __init__(self, algorithm):
        self._algorithm = algorithm

    def on(self, date_rule, time_rule, callback):
        self._algorithm.scheduled.append((date_rule, time_rule, callback))
```

In `QCAlgorithm.__init__`, add after the existing attributes:

```python
        self.settings = _Settings()
        self.schedule = _Schedule(self)
        self.date_rules = _DateRules()
        self.time_rules = _TimeRules()
        self.scheduled = []
        self.targets = []
        self.liquidated = None
        # Tests register frames keyed by tuple(symbols); history() looks the
        # request up rather than simulating a data feed.
        self.history_responses = {}
```

Add these methods to `QCAlgorithm`:

```python
    def add_crypto(self, ticker, resolution=None, market=None, leverage=None):
        return types.SimpleNamespace(symbol=Symbol(ticker), market=market, leverage=leverage)

    def add_data(self, data_type, ticker, resolution=None):
        return types.SimpleNamespace(symbol=Symbol(ticker), data_type=data_type)

    def set_warmup(self, period, resolution=None):
        self.warm_up_period = period

    def history(self, symbols, period=None, resolution=None, **options):
        key = tuple(symbols) if isinstance(symbols, (list, tuple)) else (symbols,)
        if key not in self.history_responses:
            raise AssertionError(
                f"no history registered for {key}; set algorithm.history_responses[{key}]"
            )
        return self.history_responses[key]
```

Replace `set_holdings` so it accepts either shape:

```python
    def set_holdings(self, symbol_or_targets, weight_or_liquidate=None):
        """LEAN overloads this: one symbol and a weight, or a list of targets."""
        if isinstance(symbol_or_targets, (list, tuple)):
            self.targets = [
                (target.symbol, target.quantity) for target in symbol_or_targets
            ]
            self.liquidated = bool(weight_or_liquidate)
            for symbol, weight in self.targets:
                self._apply_weight(symbol, weight)
            return
        self._apply_weight(symbol_or_targets, float(weight_or_liquidate))

    def _apply_weight(self, symbol, weight):
        self.orders.append((self.time, symbol, float(weight)))
        total = self.portfolio.total_portfolio_value
        target_value = total * float(weight)
        holding = self.portfolio[symbol]
        holding.quantity = target_value / self._price if self._price else 0.0
        holding.holdings_value = target_value
        self.portfolio.cash = total - target_value
```

Finally, extend the export list in `install()`:

```python
def install():
    """Register the fakes as the ``AlgorithmImports`` module."""
    module = types.ModuleType("AlgorithmImports")
    exported = [
        "QCAlgorithm",
        "Resolution",
        "RollingWindow",
        "Slice",
        "Symbol",
        "TradeBar",
        "Market",
        "DataNormalizationMode",
        "PortfolioTarget",
        "PythonData",
        "SubscriptionDataSource",
        "SubscriptionTransportMedium",
        "Globals",
    ]
    for name in exported:
        setattr(module, name, globals()[name])
    # AlgorithmImports re-exports these, and the QuantConnect sources rely on
    # it: the rotation algorithm writes pd.Series() and timedelta(...) with no
    # import of its own.
    module.pd = pd
    module.timedelta = timedelta
    module.datetime = datetime
    module.__all__ = exported + ["pd", "timedelta", "datetime"]
    sys.modules["AlgorithmImports"] = module
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd lean_strategies && uv run pytest tests/test_lean_stubs.py -v`
Expected: PASS

- [ ] **Step 5: Run the whole suite — the seven existing algorithm tests use these stubs**

Run: `cd lean_strategies && uv run pytest`
Expected: PASS. `test_spy_*.py` and `test_stock_algorithms.py` all drive
`set_holdings(symbol, weight)`, which the new overload must still handle.

- [ ] **Step 6: Commit**

```bash
cd lean_strategies
git add tests/lean_stubs.py tests/test_lean_stubs.py
git commit -m "test: grow the LEAN stubs for scheduled, multi-asset algorithms"
```

---

### Task 9: The algorithm

**Files:**
- Create: `lean_strategies/algorithms/ai_stocks_bonds_rotation.py`
- Test: `lean_strategies/tests/test_ai_rotation.py`

**Interfaces:**
- Consumes: the stubs from Task 8; the file format from
  `leandata/lean/customdata.py` (Task 4) — `alternative/fred/<key>.csv`
  carrying `YYYYMMDD,value`.
- Produces: `AIStocksBondsRotationAlgorithm` with `_rebalance()`,
  `_max_bitcoin_weight`, `_model`, `_scaler`, `_bitcoin`, `_equities`,
  `_symbols`, `_factors`.

**Deliberate differences from the QuantConnect source, and no others:**

1. A local `Fred` type, defined only when the name is absent.
2. `set_start_date(2021, 9, 1)` / `set_end_date(2026, 8, 31)` in place of the
   relative start.
3. Explicit `import os` and `from datetime import datetime, timedelta`. On
   QuantConnect `from AlgorithmImports import *` already provides all three and
   they resolve to the same objects, so this changes nothing there; it is
   listed because the `_rebalance` body must otherwise stay character for
   character identical, and a reviewer should be able to diff it.

**A stated assumption:** the local `Fred.reader` timestamps an observation for
day D at D+1 00:00. FRED publishes a day's value after that day's close, so
emitting it at D would let the model see it during D. One day of shift is the
conservative direction; it is recorded here because it is a choice, not a fact
read off the wire.

- [ ] **Step 1: Write the failing test**

Create `lean_strategies/tests/test_ai_rotation.py`:

```python
"""Allocation decisions of the AI stocks/bonds rotation algorithm.

Decision logic only. Fills, fees and real warm-up are LEAN's job and are
verified by running the engine (see README.md).

The regressor is replaced with a stub that returns a scripted prediction per
symbol. Fighting a real DecisionTreeRegressor into predicting a chosen sign
would test sklearn, not this file; what is worth pinning down here is the
weighting arithmetic and the Bitcoin cap.
"""

import pandas as pd
import pytest

from ai_stocks_bonds_rotation import AIStocksBondsRotationAlgorithm


class ScriptedModel:
    """Stands in for DecisionTreeRegressor: one prediction per fit/predict pair."""

    def __init__(self, predictions):
        self.predictions = list(predictions)
        self.fits = 0

    def fit(self, features, labels):
        self.fits += 1
        return self

    def predict(self, features):
        return [self.predictions.pop(0)]


class PassthroughScaler:
    def fit_transform(self, frame):
        return frame

    def transform(self, frame):
        return frame


def history_frame(column, symbols, index, values_for):
    """The shape LEAN's history() returns: MultiIndex (symbol, time), one column.

    The algorithm does ``history(...)[column].unstack(0)``, which moves level 0
    -- the symbol -- into the columns. Built row by row rather than with
    unstack/swaplevel gymnastics so the shape is obvious on inspection.
    """
    records = []
    for symbol in symbols:
        for stamp, value in zip(index, values_for(symbol)):
            records.append((symbol, stamp, float(value)))
    frame = pd.DataFrame(records, columns=["symbol", "time", column])
    return frame.set_index(["symbol", "time"])


def build(predictions, max_bitcoin_weight=0.1):
    algorithm = AIStocksBondsRotationAlgorithm()
    algorithm.initialize()
    algorithm._model = ScriptedModel(predictions)
    algorithm._scaler = PassthroughScaler()
    algorithm._max_bitcoin_weight = max_bitcoin_weight

    # Long enough that pct_change(21).shift(-21).dropna() leaves rows behind.
    index = pd.date_range("2020-01-01", periods=80, freq="D")
    algorithm.history_responses[tuple(algorithm._factors)] = history_frame(
        "value", algorithm._factors, index, lambda symbol: range(len(index))
    )
    algorithm.history_responses[tuple(algorithm._symbols)] = history_frame(
        "close", algorithm._symbols, index, lambda symbol: range(1, len(index) + 1)
    )
    return algorithm


def weights(algorithm):
    return dict(algorithm.targets)


def test_all_positive_predictions_are_weighted_to_one_and_a_half_gross():
    # SPY, GLD, BND, BTCUSD -> equal predictions -> equal weights.
    algorithm = build([1.0, 1.0, 1.0, 1.0], max_bitcoin_weight=1.0)
    algorithm._rebalance()
    assert sum(weights(algorithm).values()) == pytest.approx(1.5)
    assert set(weights(algorithm).values()) == {pytest.approx(0.375)}


def test_a_non_positive_prediction_drops_the_symbol_entirely():
    algorithm = build([1.0, -1.0, 1.0, 1.0], max_bitcoin_weight=1.0)
    algorithm._rebalance()
    assert algorithm._equities[1] not in weights(algorithm)
    assert len(weights(algorithm)) == 3
    assert sum(weights(algorithm).values()) == pytest.approx(1.5)


def test_the_bitcoin_cap_is_applied_and_equities_absorb_the_rest():
    # Bitcoin would otherwise take 0.375; capped at 0.10. The equities are then
    # renormalised to 1.5 among themselves, so gross exposure ends at 1.6 rather
    # than 1.5. That is what QuantConnect's code does; it is reproduced, not
    # corrected, because the point of the exercise is identical decision logic.
    algorithm = build([1.0, 1.0, 1.0, 1.0], max_bitcoin_weight=0.1)
    algorithm._rebalance()
    allocated = weights(algorithm)
    assert allocated[algorithm._bitcoin] == pytest.approx(0.1)
    equity_total = sum(allocated[symbol] for symbol in algorithm._equities)
    assert equity_total == pytest.approx(1.5)
    assert sum(allocated.values()) == pytest.approx(1.6)


def test_the_cap_does_nothing_when_bitcoin_is_already_below_it():
    algorithm = build([10.0, 10.0, 10.0, 0.1], max_bitcoin_weight=0.5)
    algorithm._rebalance()
    allocated = weights(algorithm)
    assert allocated[algorithm._bitcoin] < 0.5
    assert sum(allocated.values()) == pytest.approx(1.5)


def test_targets_are_liquidating_so_dropped_symbols_are_closed():
    algorithm = build([1.0, 1.0, 1.0, 1.0], max_bitcoin_weight=1.0)
    algorithm._rebalance()
    assert algorithm.liquidated is True


def test_warming_up_rebalances_nothing():
    algorithm = build([1.0, 1.0, 1.0, 1.0])
    algorithm.is_warming_up = True
    algorithm._rebalance()
    assert algorithm.targets == []


def test_the_four_securities_and_three_factors_are_subscribed():
    algorithm = AIStocksBondsRotationAlgorithm()
    algorithm.initialize()
    assert list(algorithm._equities) == ["SPY", "GLD", "BND"]
    assert algorithm._bitcoin == "BTCUSD"
    assert list(algorithm._factors) == ["VIXCLS", "T10Y3M", "DFF"]


def test_a_monthly_rebalance_is_scheduled():
    algorithm = AIStocksBondsRotationAlgorithm()
    algorithm.initialize()
    assert len(algorithm.scheduled) == 1
    date_rule, time_rule, callback = algorithm.scheduled[0]
    assert date_rule.name == "month_start"
    assert time_rule.name == "after_market_open"
    assert callback == algorithm._rebalance
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd lean_strategies && uv run pytest tests/test_ai_rotation.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'ai_stocks_bonds_rotation'`

- [ ] **Step 3: Create `algorithms/ai_stocks_bonds_rotation.py`**

```python
# region imports
from AlgorithmImports import *

import os
from datetime import datetime, timedelta

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
# endregion

# QuantConnect ships a `Fred` data type through its FRED dataset. The LEAN
# repository this project runs against does not -- that class lives in the
# separate Lean.DataSource.FRED project -- so define a stand-in, but only when
# the name is absent. Pasted into a QuantConnect project this block does
# nothing and the real type is used, which keeps the promise the README makes
# about every file in algorithms/.
try:
    Fred
except NameError:
    class Fred(PythonData):
        """Reads `alternative/fred/<series>.csv`, written by leandata.

        The row format is `YYYYMMDD,value` and is a contract with
        leandata/lean/customdata.py; change one and you must change the other.

        An observation for day D is timestamped D+1 00:00. FRED publishes a
        day's value after that day's close, so emitting it at D would let the
        model train on a number that did not exist yet. One day of shift is
        the conservative direction. This is a decision, not something read off
        the wire, and it is the most likely source of a small difference
        against a QuantConnect run.
        """

        def get_source(self, config, date, is_live_mode):
            path = os.path.join(
                Globals.data_folder, "alternative", "fred",
                f"{config.symbol.value.lower()}.csv",
            )
            return SubscriptionDataSource(path, SubscriptionTransportMedium.LOCAL_FILE)

        def reader(self, config, line, date, is_live_mode):
            if not line or not line[0].isdigit():
                return None
            parts = line.split(",")
            if len(parts) < 2 or not parts[1].strip():
                return None
            point = Fred()
            point.symbol = config.symbol
            point.time = datetime.strptime(parts[0], "%Y%m%d") + timedelta(days=1)
            point.value = float(parts[1])
            return point


class AIStocksBondsRotationAlgorithm(QCAlgorithm):

    def initialize(self):
        # QuantConnect's original reads set_start_date(self.end_date -
        # timedelta(5*365)). Fixed here the way every other algorithm in this
        # repo is; delete set_end_date before pasting into QuantConnect.
        self.set_start_date(2021, 9, 1)
        self.set_end_date(2026, 8, 31)
        self.settings.daily_precise_end_time = False
        self.settings.seed_initial_prices = True
        # Add securities
        self._bitcoin = self.add_crypto("BTCUSD", market=Market.BITFINEX, leverage=2).symbol
        self._equities = [self.add_equity(ticker).symbol for ticker in ['SPY', 'GLD', 'BND']]
        self._symbols = self._equities + [self._bitcoin]
        # Add FRED data
        self._factors = [
            self.add_data(Fred, ticker, Resolution.DAILY).symbol
            for ticker in ['VIXCLS', 'T10Y3M', 'DFF']
        ]
        # ML model setup
        self._model = DecisionTreeRegressor(max_depth=12, random_state=1)
        self._scaler = StandardScaler()
        # Parameters
        self._max_bitcoin_weight = self.get_parameter("max_bitcoin_weight", 0.1)
        lookback_years = self.get_parameter("lookback_years", 4)
        self._lookback = timedelta(lookback_years * 365)
        # Schedule monthly rebalancing
        self.schedule.on(
            self.date_rules.month_start(self._equities[0]),
            self.time_rules.after_market_open(self._equities[0], 1),
            self._rebalance
        )
        self.set_warm_up(timedelta(35))

    def _rebalance(self):
        if self.is_warming_up:
            return
        # Get historical factor data
        factors = self.history(
            self._factors,
            self._lookback,
            Resolution.DAILY
        )["value"].unstack(0).dropna()

        # Calculate 21-day forward returns as labels
        labels = self.history(
            self._symbols,
            self._lookback,
            Resolution.DAILY,
            data_normalization_mode=DataNormalizationMode.TOTAL_RETURN
        )["close"].unstack(0).dropna().pct_change(21).shift(-21).dropna()

        # Train model and make predictions
        prediction_by_symbol = pd.Series()
        for symbol in self._symbols:
            asset_labels = labels[symbol].dropna()
            idx = factors.index.intersection(asset_labels.index)
            # Fit model
            self._model.fit(
                self._scaler.fit_transform(factors.loc[idx]),
                asset_labels.loc[idx]
            )
            # Predict
            prediction = self._model.predict(self._scaler.transform([factors.iloc[-1]]))[0]
            if prediction > 0:
                prediction_by_symbol.loc[symbol] = prediction
        # Calculate weights
        weight_by_symbol = (
            1.5 * prediction_by_symbol / prediction_by_symbol.sum()
        )
        # Cap Bitcoin weight
        if (self._bitcoin in weight_by_symbol
            and weight_by_symbol.loc[self._bitcoin] > self._max_bitcoin_weight):
            weight_by_symbol.loc[self._bitcoin] = self._max_bitcoin_weight
            if len(weight_by_symbol) > 1:
                equities = [s for s in self._equities if s in weight_by_symbol]
                equity_weights = weight_by_symbol.loc[equities]
                weight_by_symbol.loc[equities] = (
                    1.5 * equity_weights
                    / equity_weights.sum()
                )
        # Execute trades
        targets = [
            PortfolioTarget(symbol, weight)
            for symbol, weight in weight_by_symbol.items()
        ]
        self.set_holdings(targets, True)

    def on_warmup_finished(self):
        self._rebalance()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd lean_strategies && uv run pytest tests/test_ai_rotation.py -v`
Expected: PASS, 8 tests

If `test_all_positive_predictions_are_weighted_to_one_and_a_half_gross` fails
on the shape of the history frames, fix the *test helper*, not the algorithm:
the algorithm body must stay byte-identical to QuantConnect's apart from the
two documented differences. The frames must be MultiIndexed `(symbol, time)`
so that `["value"].unstack(0)` yields one column per symbol.

- [ ] **Step 5: Run the whole suite**

Run: `cd lean_strategies && uv run pytest`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd lean_strategies
git add algorithms/ai_stocks_bonds_rotation.py tests/test_ai_rotation.py
git commit -m "feat: port the AI stocks/bonds rotation algorithm"
```

---

### Task 10: Dependencies and the variant entry

**Files:**
- Modify: `lean_strategies/pyproject.toml`
- Modify: `lean_strategies/variants.py`
- Test: `lean_strategies/tests/test_variants.py` (existing; confirm it still passes)

**Interfaces:**
- Produces: the variant name `"AI Stocks/Bonds Rotation"` mapping to
  `("ai_stocks_bonds_rotation", {})`, and `ALGORITHM_CLASSES` gaining
  `"ai_stocks_bonds_rotation": "AIStocksBondsRotationAlgorithm"`.

- [ ] **Step 1: Add scikit-learn to the default dependencies**

In `lean_strategies/pyproject.toml`, change:

```toml
dependencies = [
    "pandas==2.2.3",
    "wrapt==1.16.0",
    # Imported by algorithms/ai_stocks_bonds_rotation.py at backtest time, so
    # unlike yfinance it cannot live in an optional group: run_local.py points
    # LEAN's embedded interpreter at this same .venv.
    "scikit-learn>=1.5",
]
```

- [ ] **Step 2: Install and confirm the embedded interpreter can see it**

Run:
```bash
cd lean_strategies
uv sync
uv run python -c "import sklearn; print(sklearn.__version__)"
```
Expected: a version number, no ImportError.

- [ ] **Step 3: Register the variant**

In `lean_strategies/variants.py`, add to `ALGORITHM_CLASSES`:

```python
    "ai_stocks_bonds_rotation": "AIStocksBondsRotationAlgorithm",
```

And append after the existing `VARIANTS.update(...)` block:

```python
# The QuantConnect leaderboard replication. Needs a data folder holding SPY,
# GLD and BND from yfinance, BTCUSD from Bitfinex, and the three FRED series:
#   uv run --group data convert_data.py convert --source yfinance --ticker SPY \
#       --exchange-code P --start 2016-01-01 --end 2026-08-31
#   uv run convert_data.py convert --source bitfinex --ticker BTCUSD \
#       --security-type crypto --market bitfinex --start 2016-01-01 --end 2026-08-31
#   uv run convert_data.py convert-series --source fred --ticker VIXCLS \
#       --start 2016-01-01 --end 2026-08-31
# The algorithm carries its own dates, so the variant needs no parameters.
VARIANTS["AI Stocks/Bonds Rotation"] = ("ai_stocks_bonds_rotation", {})
```

- [ ] **Step 4: Run the variant test**

Run: `cd lean_strategies && uv run pytest tests/test_variants.py -v`
Expected: PASS. This test checks every variant names a real module and class.

- [ ] **Step 5: Confirm run_local.py lists it**

Run: `cd lean_strategies && uv run run_local.py --list`
Expected: `AI Stocks/Bonds Rotation` appears in the output.

- [ ] **Step 6: Run the whole suite**

Run: `cd lean_strategies && uv run pytest`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
cd lean_strategies
git add pyproject.toml uv.lock variants.py
git commit -m "chore: add scikit-learn and register the rotation variant"
```

---

### Task 11: Build the data folder, run the backtest, record the comparison

**Files:**
- Modify: `lean_strategies/README.md`
- Creates (not committed, `data/` is gitignored for prices):
  `data/equity/usa/**`, `data/crypto/bitfinex/daily/btcusd_trade.zip`,
  `data/alternative/fred/{vixcls,t10y3m,dff}.csv`

**Interfaces:**
- Consumes: every task above.
- Produces: a measured result table in `README.md`.

- [ ] **Step 1: Fetch the three equities**

Run:
```bash
cd lean_strategies
for TICKER in SPY GLD BND; do
  uv run --group data convert_data.py convert --source yfinance --ticker $TICKER \
      --exchange-code P --start 2016-01-01 --end 2026-08-31
done
```
Expected: three `bars ... rows` reports. GLD and BND are NYSE Arca listed, so
the default `--exchange-code P` is right for all three; confirm against
`../Lean/Data/equity/usa/map_files/` if a run later disagrees.

- [ ] **Step 2: Fetch BTCUSD**

Run:
```bash
cd lean_strategies
uv run convert_data.py convert --source bitfinex --ticker BTCUSD \
    --security-type crypto --market bitfinex \
    --start 2016-01-01 --end 2026-08-31
```
Expected: roughly 3889 rows, 2016-01-01 to 2026-08-31, and a
`data/crypto/bitfinex/daily/btcusd_trade.zip` with no factor or map file
beside it.

- [ ] **Step 3: Fetch the three FRED series**

Run:
```bash
cd lean_strategies
for SERIES in VIXCLS T10Y3M DFF; do
  uv run convert_data.py convert-series --source fred --ticker $SERIES \
      --start 2016-01-01 --end 2026-08-31
done
```
Expected: about 2780, 2781 and 3893 rows respectively, with a dropped-observation
warning for the first two and none for DFF.

- [ ] **Step 4: Confirm the folder is complete**

Run: `cd lean_strategies && uv run convert_data.py check`
Expected: SPY, GLD, BND, BTCUSD, VIXCLS, T10Y3M and DFF all listed, no
`MISSING reference data`, and a risk-free curve line.

- [ ] **Step 5: Run the backtest**

Run:
```bash
cd lean_strategies
uv run run_local.py "AI Stocks/Bonds Rotation" --data-folder data
```
Expected: the run completes and writes to `results/ai_stocks_bonds_rotation/`.

A log line reading that LEAN could not find `equity/usa/hour/<ticker>.zip` is
expected and harmless — README.md already documents it. If the run fails
instead, work the failure with superpowers:systematic-debugging; do **not**
edit the algorithm body to make it pass.

- [ ] **Step 6: Record the comparison in README.md**

Add a section after "Worked example: a stock LEAN does not ship". Fill the
`local` column from the actual run; do not invent numbers.

```markdown
### Replicating a QuantConnect leaderboard algorithm

`algorithms/ai_stocks_bonds_rotation.py` is QuantConnect's
`AIStocksBondsRotationAlgorithm`, run against data this repo downloads. Build
the folder first:

    uv run --group data convert_data.py convert --source yfinance --ticker SPY \
        --exchange-code P --start 2016-01-01 --end 2026-08-31     # and GLD, BND
    uv run convert_data.py convert --source bitfinex --ticker BTCUSD \
        --security-type crypto --market bitfinex --start 2016-01-01 --end 2026-08-31
    uv run convert_data.py convert-series --source fred --ticker VIXCLS \
        --start 2016-01-01 --end 2026-08-31                        # and T10Y3M, DFF
    uv run run_local.py "AI Stocks/Bonds Rotation" --data-folder data

The window reaches back to 2016 rather than to the 2021 backtest start because
`lookback_years` defaults to 4: the first rebalance asks for four years of
history before it will trade.

| Statistic | QuantConnect | local |
|---|---|---|
| Total Orders | 207 | _fill in_ |
| Compounding Annual Return | 26.262% | _fill in_ |
| Drawdown | 31.900% | _fill in_ |
| Sharpe Ratio | 0.77 | _fill in_ |
| Total Fees | $4043.42 | _fill in_ |

**These two columns are not the same experiment.** QuantConnect's run covers
five years ending on a date their published statistics do not record, and the
local run covers 2021-09-01 to 2026-08-31. Two of the three data kinds do come
from the same origin they use — Bitfinex for BTCUSD, FRED for the factors —
and SPY, GLD and BND agree with QuantConnect's equity data to 0.0000% over
this window. What does not match is the period, the bar resolution and the
sklearn build.

Read a divergence here carefully. `DecisionTreeRegressor(max_depth=12)` fitted
on roughly a thousand samples amplifies small input differences far more than
the moving-average strategies above do, where a 0.146% close difference already
produced ten extra round trips. A large gap is not by itself evidence of a
conversion bug.
```

- [ ] **Step 7: Add the new data paths to `.gitignore` if they are not covered**

Run: `cd lean_strategies && git status --short data/`
Expected: no `data/` entries appear. If `data/crypto/` or
`data/alternative/fred/` do show up, add them to `.gitignore` beside the
existing price-data entries — generated prices stay out of git, as README's
table explains.

- [ ] **Step 8: Run the whole suite one last time**

Run: `cd lean_strategies && uv run pytest`
Expected: PASS

Then the network-marked ones:

Run: `cd lean_strategies && uv run --group data pytest -m network`
Expected: PASS, three tests (yfinance, Bitfinex, FRED).

- [ ] **Step 9: Commit**

```bash
cd lean_strategies
git add README.md .gitignore
git commit -m "docs: record the AI rotation run against locally converted data"
```

---

## Self-review notes

Checked against the spec:

- Every spec section maps to a task. `AssetProfile` is Task 1, bars/writer
  Task 2, `read_daily_csv` and `describe` Task 3, `ValueSeries` and the custom
  data writer Task 4, the two sources Tasks 5 and 6, the CLI Task 7, the
  algorithm Task 9, dependencies and variants Task 10, and the measured
  comparison Task 11. Task 8 exists because the spec's testing section calls
  for `test_ai_rotation.py` driven through `tests/lean_stubs.py`, and today's
  stubs cover a single-symbol moving-average algorithm only.
- The spec's `write_value_series` path (`alternative/fred/<key>.csv`) and the
  algorithm's `Fred.get_source` path are the same string, built from
  `CUSTOM_DATA_ROOT` on one side and `os.path.join(Globals.data_folder,
  "alternative", "fred", ...)` on the other. This duplication is deliberate:
  `algorithms/` may not import `leandata`, because the file has to run on
  QuantConnect. Both sides carry a comment naming the other.
- `DailyBarLayout.entries` keeps its two-argument signature so
  `tests/test_lean_files.py:26` keeps working; the profile is derived from
  `symbol.security_type` instead. This is the one place the plan deviates from
  the spec's sketch, which showed a `profile` parameter.
- `read_daily_csv`'s `price_scaled` defaults to True for the same reason.
- `FredSource.fetch` deliberately raises. It is registered in the same registry
  as the security sources, so the CLI checks for `fetch_series` before calling.
