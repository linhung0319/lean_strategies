"""Bar layouts, map files, the writer and the overlay -- everything that
turns the canonical model into files LEAN will open."""

import zipfile
from datetime import date
from pathlib import Path, PurePosixPath

import pytest

from leandata.errors import LeanDataError, UnsupportedResolutionError
from leandata.lean import overlay
from leandata.lean.bars import DailyBarLayout, layout_for
from leandata.lean.mapfiles import build_map_rows, render_map_file
from leandata.lean.writer import LeanDataWriter
from leandata.model import Provenance, Resolution, SymbolSpec
from leandata_helpers import make_bars, make_dividends, make_history

from datetime import datetime, timezone


# --- bar layouts -----------------------------------------------------------


def test_daily_layout_paths_and_payload_match_the_bundled_shape():
    bars = make_bars([97.31, 97.53, 96.53], volume=2150000)
    entries = list(DailyBarLayout().entries(SymbolSpec("SPY"), bars))
    assert len(entries) == 1
    entry = entries[0]
    assert entry.zip_path == PurePosixPath("equity/usa/daily/spy.zip")
    assert entry.entry_name == "spy.csv"
    assert entry.payload == (
        b"20200106 00:00,973100,973100,973100,973100,2150000\n"
        b"20200107 00:00,975300,975300,975300,975300,2150000\n"
        b"20200108 00:00,965300,965300,965300,965300,2150000"
    )
    # LF separated, no trailing newline -- exactly like Lean's spy.csv.
    assert not entry.payload.endswith(b"\n")


def test_layout_lookup_reports_what_is_implemented():
    assert isinstance(layout_for(Resolution.DAILY), DailyBarLayout)
    with pytest.raises(UnsupportedResolutionError, match="implemented: daily"):
        layout_for(Resolution.MINUTE)


# --- map files -------------------------------------------------------------


def test_map_file_has_two_rows_and_a_trailing_crlf():
    rows = build_map_rows(SymbolSpec("SPY"), date(1998, 1, 2))
    assert render_map_file(rows) == b"19980102,spy,P\r\n20501231,spy,P\r\n"


def test_map_file_honours_an_explicit_listing_date_and_exchange():
    symbol = SymbolSpec("AAPL", exchange_code="Q", listed_from=date(1980, 12, 12))
    payload = render_map_file(build_map_rows(symbol, date(1998, 1, 2)))
    assert payload == b"19801212,aapl,Q\r\n20501231,aapl,Q\r\n"


# --- writer ----------------------------------------------------------------


def test_writer_lays_out_all_three_artifacts(tmp_path):
    history = make_history(
        bars=make_bars([10.0, 11.0, 12.0]),
        dividends=make_dividends({"2020-01-08": 0.5}),
    )
    report = LeanDataWriter(tmp_path).write(history)

    assert report.bar_files == (tmp_path / "equity" / "usa" / "daily" / "spy.zip",)
    assert report.factor_file == tmp_path / "equity" / "usa" / "factor_files" / "spy.csv"
    assert report.map_file == tmp_path / "equity" / "usa" / "map_files" / "spy.csv"
    assert report.bar_count == 3
    assert report.first_date == date(2020, 1, 6)

    with zipfile.ZipFile(report.bar_files[0]) as archive:
        assert archive.namelist() == ["spy.csv"]
        assert archive.read("spy.csv").startswith(b"20200106 00:00,100000,")

    # Binary writes throughout: a text-mode write would double the CR here.
    assert b"\r\r\n" not in report.factor_file.read_bytes()
    assert report.map_file.read_bytes() == b"20200106,spy,P\r\n20501231,spy,P\r\n"


def test_writer_records_provenance_when_present(tmp_path):
    history = make_history()
    object.__setattr__(
        history,
        "provenance",
        Provenance(source="csv", fetched_at=datetime(2026, 8, 8, tzinfo=timezone.utc), source_version="1"),
    )
    LeanDataWriter(tmp_path).write(history)
    sidecar = tmp_path / ".provenance" / "spy-daily.json"
    assert '"source": "csv"' in sidecar.read_text(encoding="utf-8")


def test_writer_refuses_to_clobber_when_overwrite_is_off(tmp_path):
    history = make_history()
    LeanDataWriter(tmp_path).write(history)
    with pytest.raises(LeanDataError, match="overwrite=False"):
        LeanDataWriter(tmp_path, overwrite=False).write(history)


def test_writer_leaves_no_temporary_files_behind(tmp_path):
    LeanDataWriter(tmp_path).write(make_history())
    assert list(tmp_path.rglob("*.tmp")) == []


# --- overlay ---------------------------------------------------------------


def fake_lean_data(root: Path) -> Path:
    lean_data = root / "Lean" / "Data"
    (lean_data / "market-hours").mkdir(parents=True)
    (lean_data / "market-hours" / "market-hours-database.json").write_text("{}", encoding="utf-8")
    (lean_data / "symbol-properties").mkdir(parents=True)
    (lean_data / "symbol-properties" / "symbol-properties-database.csv").write_text("a,b\n", encoding="utf-8")
    (lean_data / "alternative" / "interest-rate" / "usa").mkdir(parents=True)
    (lean_data / "alternative" / "interest-rate" / "usa" / "interest-rate.csv").write_text("1,2\n", encoding="utf-8")
    return lean_data


def snapshot(root: Path):
    return {
        path.relative_to(root): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_ensure_overlay_copies_every_reference_path(tmp_path):
    lean_data = fake_lean_data(tmp_path)
    target = tmp_path / "data"
    overlay.ensure_overlay(target, lean_data)
    for relative in overlay.MIRRORED:
        assert (target / relative).exists()
    assert overlay.missing_reference_data(target) == []


def test_ensure_overlay_never_touches_the_lean_clone(tmp_path):
    lean_data = fake_lean_data(tmp_path)
    before = snapshot(lean_data)
    overlay.ensure_overlay(tmp_path / "data", lean_data)
    assert snapshot(lean_data) == before


def test_ensure_overlay_is_idempotent(tmp_path):
    lean_data = fake_lean_data(tmp_path)
    target = tmp_path / "data"
    assert len(overlay.ensure_overlay(target, lean_data)) == 3
    assert overlay.ensure_overlay(target, lean_data) == []


def test_missing_reference_data_names_each_gap(tmp_path):
    lean_data = fake_lean_data(tmp_path)
    target = tmp_path / "data"
    overlay.ensure_overlay(target, lean_data)
    import shutil

    shutil.rmtree(target / "alternative" / "interest-rate" / "usa")
    assert overlay.missing_reference_data(target) == [PurePosixPath("alternative/interest-rate/usa")]


def test_describe_reads_back_what_the_writer_wrote(tmp_path):
    lean_data = fake_lean_data(tmp_path)
    target = tmp_path / "data"
    LeanDataWriter(target).write(make_history(bars=make_bars([10.0, 11.0, 12.0])))
    overlay.ensure_overlay(target, lean_data)

    info = overlay.describe(target)
    assert info.missing_reference == ()
    assert len(info.series) == 1
    assert info.series[0].ticker == "SPY"
    assert info.series[0].rows == 3
    assert info.series[0].first_date == date(2020, 1, 6)
    assert info.series[0].last_date == date(2020, 1, 8)


def test_read_daily_csv_unscales_prices(tmp_path):
    LeanDataWriter(tmp_path).write(make_history(bars=make_bars([97.31, 97.53], volume=2150000)))
    rows = overlay.read_daily_csv(tmp_path / "equity" / "usa" / "daily" / "spy.zip")
    assert rows[0] == (date(2020, 1, 6), 97.31, 97.31, 97.31, 97.31, 2150000.0)


def test_a_locally_edited_reference_file_survives_the_next_convert(tmp_path):
    # LEAN's interest-rate.csv stops in 2023, so backtesting past that means
    # appending rows to the mirror. The next convert must not undo that.
    lean_data = fake_lean_data(tmp_path)
    target = tmp_path / "data"
    overlay.ensure_overlay(target, lean_data)

    rates = target / "alternative" / "interest-rate" / "usa" / "interest-rate.csv"
    rates.write_text("1,2\n2024-09-19,5.0\n2024-12-19,4.5\n", encoding="utf-8")
    edited = rates.read_text(encoding="utf-8")

    assert overlay.ensure_overlay(target, lean_data) == []
    assert rates.read_text(encoding="utf-8") == edited


def test_refresh_overlay_does_replace_a_local_edit(tmp_path):
    lean_data = fake_lean_data(tmp_path)
    target = tmp_path / "data"
    overlay.ensure_overlay(target, lean_data)
    rates = target / "alternative" / "interest-rate" / "usa" / "interest-rate.csv"
    rates.write_text("edited\n", encoding="utf-8")

    overlay.ensure_overlay(target, lean_data, refresh=True)
    assert rates.read_text(encoding="utf-8") == "1,2\n"


# --- the interest-rate top-up ----------------------------------------------


def write_patch(root: Path, rows: str) -> Path:
    patch = root / "patch.csv"
    patch.write_text("DATE,PCREDIT8\n" + rows, encoding="utf-8")
    return patch


def rate_file(root: Path) -> Path:
    return root / "alternative" / "interest-rate" / "usa" / "interest-rate.csv"


def prepare_rates(tmp_path: Path) -> Path:
    lean_data = fake_lean_data(tmp_path)
    rate_file(lean_data).write_text(
        "DATE,PCREDIT8\n2023-05-04,5.25\n2023-07-27,5.5\n", encoding="utf-8"
    )
    target = tmp_path / "data"
    overlay.ensure_overlay(target, lean_data)
    return target


def test_only_rows_after_the_last_bundled_one_are_appended(tmp_path):
    target = prepare_rates(tmp_path)
    patch = write_patch(tmp_path, "2023-05-04,9.99\n2024-09-19,5\n2024-12-19,4.5\n")

    added = overlay.extend_interest_rates(target, patch)

    assert added == ["2024-09-19,5", "2024-12-19,4.5"]
    # The row that would contradict LEAN's own data is ignored, not applied.
    assert "9.99" not in rate_file(target).read_text(encoding="utf-8")


def test_appending_twice_changes_nothing(tmp_path):
    target = prepare_rates(tmp_path)
    patch = write_patch(tmp_path, "2024-09-19,5\n")
    overlay.extend_interest_rates(target, patch)
    once = rate_file(target).read_text(encoding="utf-8")

    assert overlay.extend_interest_rates(target, patch) == []
    assert rate_file(target).read_text(encoding="utf-8") == once


def test_interest_rate_range_reports_the_curve(tmp_path):
    target = prepare_rates(tmp_path)
    overlay.extend_interest_rates(target, write_patch(tmp_path, "2025-12-11,3.75\n"))
    assert overlay.interest_rate_range(target) == (date(2023, 5, 4), date(2025, 12, 11))


def test_the_shipped_patch_only_extends_the_real_lean_file():
    # Guards the patch actually committed to this repo, not a fixture.
    import csv

    from leandata.lean.overlay import INTEREST_RATE_PATCH

    rows = list(csv.reader(INTEREST_RATE_PATCH.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == ["DATE", "PCREDIT8"]
    dates = [row[0] for row in rows[1:]]
    assert dates == sorted(dates)
    assert min(dates) > "2023-07-27"  # LEAN's last bundled row
    assert all(0 < float(row[1]) < 25 for row in rows[1:])


def test_a_short_rate_hold_is_not_a_warning():
    # No FOMC change for eight months is ordinary, not stale data.
    assert overlay.stale_rate_warning(date(2025, 12, 11), date(2026, 8, 7)) is None


def test_an_implausibly_long_rate_hold_is_flagged():
    message = overlay.stale_rate_warning(date(2023, 7, 27), date(2026, 8, 7))
    assert message is not None
    assert "probably been missed" in message
    assert message.endswith("\n")
