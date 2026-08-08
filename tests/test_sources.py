"""The registry is the whole extensibility story, so it gets tested as such."""

import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from leandata.errors import SourceError
from leandata.model import Resolution
from leandata.snapshot import read_snapshot, write_snapshot
from leandata.sources import available, get_source, register
from leandata.sources.base import DataSource, FetchRequest
from leandata_helpers import make_bars, make_dividends, make_history, make_splits


def test_both_adapters_are_registered():
    assert available() == ("csv", "yfinance")


def test_an_unknown_source_names_the_ones_that_exist():
    with pytest.raises(SourceError, match="unknown data source 'quandl'; available: csv, yfinance"):
        get_source("quandl")


def test_bad_options_are_reported_against_the_source():
    with pytest.raises(SourceError, match="bad options for source 'csv'"):
        get_source("csv", nonsense=1)


def test_a_new_source_needs_only_a_register_call():
    class Stub:
        name = "stub"

        def supported_resolutions(self):
            return frozenset({Resolution.DAILY})

        def fetch(self, request):
            return make_history()

    register("stub", lambda **options: Stub())
    try:
        source = get_source("stub")
        assert isinstance(source, DataSource)
        assert source.fetch(FetchRequest(ticker="SPY")).symbol.ticker == "SPY"
    finally:
        from leandata.sources import _REGISTRY

        del _REGISTRY["stub"]


def test_importing_the_package_does_not_import_yfinance():
    # The LEAN runs share this virtualenv; loading yfinance and its compiled
    # dependency stack inside the embedded interpreter is pure risk.
    script = "import sys, leandata; print('yfinance' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


# --- the csv adapter and the snapshot format --------------------------------


def test_a_snapshot_round_trips_through_the_csv_source(tmp_path):
    original = make_history(
        bars=make_bars([10.0, 11.0, 12.0, 13.0]),
        dividends=make_dividends({"2020-01-08": 0.5}),
        splits=make_splits({"2020-01-09": 2.0}),
    )
    write_snapshot(original, tmp_path, "spy_daily")

    source = get_source("csv", path=tmp_path / "spy_daily.csv")
    restored = source.fetch(FetchRequest(ticker="SPY"))

    assert list(restored.bars["close"]) == list(original.bars["close"])
    assert list(restored.bars.index) == list(original.bars.index)
    assert list(restored.dividends["amount"]) == [0.5]
    assert list(restored.splits["ratio"]) == [2.0]


def test_a_snapshot_without_actions_reads_back_empty(tmp_path):
    write_snapshot(make_history(), tmp_path, "spy_daily")
    bars, dividends, splits = read_snapshot(tmp_path / "spy_daily.csv")
    assert len(bars) == 3
    assert dividends.empty and splits.empty


def test_the_csv_source_clips_to_the_requested_window(tmp_path):
    write_snapshot(make_history(bars=make_bars([10.0, 11.0, 12.0, 13.0])), tmp_path, "spy_daily")
    source = get_source("csv", path=tmp_path / "spy_daily.csv")
    history = source.fetch(
        FetchRequest(ticker="SPY", start=date(2020, 1, 7), end=date(2020, 1, 8))
    )
    assert len(history.bars) == 2


def test_a_missing_snapshot_says_where_it_looked(tmp_path):
    source = get_source("csv", path=tmp_path / "absent.csv")
    with pytest.raises(FileNotFoundError, match="absent.csv"):
        source.fetch(FetchRequest(ticker="SPY"))
