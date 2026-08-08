"""Cross-validate the converter against LEAN's own bundled data.

The synthetic tests prove the algorithm does what it says; these prove it
agrees with QuantConnect's production pipeline on real securities.

Where the two disagree, it is the underlying vendor data rather than the
conversion. Two observations pin that down:

* SPY closes agree to within half a cent on 99.8% of days from 2017 onwards
  (mean relative error 0.0000%) and diverge further back -- 0.146% mean
  before 2009. That gradient is the signature of different historical
  consolidation, not of a formatting bug. Both sources carry artifacts in
  the early years: QC's bundled spy.zip has a high of 148.88 on 2000-12-11
  against its own open of 137.38 and close of 138.63.
* AAPL's recent factor rows come out byte-identical, because that is the
  window where the price and dividend inputs agree.

Fixtures are snapshots truncated at 2021-03-31, the last bundled bar. The
end date matters: LEAN's Adjusted normalisation is anchored at the end of
the factor file, so a longer download would rescale every adjusted price by
a constant and make the row-by-row comparison meaningless.
"""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from leandata.lean.factors import build_factor_rows
from leandata.lean.overlay import read_daily_csv
from leandata.snapshot import read_snapshot

FIXTURES = Path(__file__).resolve().parent / "fixtures"
BUNDLED = Path(__file__).resolve().parents[2] / "Lean" / "Data" / "equity" / "usa"

pytestmark = pytest.mark.skipif(not BUNDLED.exists(), reason="the Lean clone is not checked out")

WINDOW = "1998-01-02_2021-03-31"


def snapshot_rows(ticker: str):
    bars, dividends, splits = read_snapshot(
        FIXTURES / f"{ticker.lower()}_yahoo_daily_{WINDOW}.csv"
    )
    rows, _ = build_factor_rows(bars, dividends, splits)
    return bars, rows


def bundled_factor_rows(ticker: str) -> dict:
    payload = (BUNDLED / "factor_files" / f"{ticker.lower()}.csv").read_bytes().decode("ascii")
    rows = {}
    for line in payload.splitlines():
        day, price_factor, split_factor, reference_price = line.split(",")[:4]
        rows[day] = (Decimal(price_factor), Decimal(split_factor), Decimal(reference_price))
    return rows


# --- SPY: the dividend chain -----------------------------------------------


def test_spy_factor_file_has_the_same_shape_as_the_bundled_one():
    _, rows = snapshot_rows("SPY")
    bundled = bundled_factor_rows("SPY")

    generated_dates = [row.date.strftime("%Y%m%d") for row in rows]
    # Identical dates on their own catch a wrong previous-trading-day rule, a
    # dividend snapped or dropped in error, and an off-by-one terminal row.
    assert generated_dates == list(bundled)
    assert len(rows) == 96


def test_spy_has_no_splits_so_every_split_factor_renders_as_one():
    _, rows = snapshot_rows("SPY")
    assert {row.render().split(",")[2] for row in rows} == {"1"}


def test_spy_price_factors_track_the_bundled_chain():
    _, rows = snapshot_rows("SPY")
    bundled = bundled_factor_rows("SPY")

    worst = max(
        abs(row.price_factor - bundled[row.date.strftime("%Y%m%d")][0]) / bundled[row.date.strftime("%Y%m%d")][0]
        for row in rows
        if bundled[row.date.strftime("%Y%m%d")][0]
    )
    # Yahoo publishes dividends rounded to three decimals (1.278 where QC's
    # data implies 1.27779). Ninety-four of those compound into a few parts
    # in ten thousand; the observed worst case is 4.1e-4.
    assert worst < 1e-3, f"price factors drifted by {worst:.2e}"


def test_spy_trading_calendar_matches_the_bundled_data_exactly():
    generated = {row[0] for row in _generated_daily("SPY")}
    reference = {row[0] for row in read_daily_csv(BUNDLED / "daily" / "spy.zip")}
    assert generated == reference
    assert len(generated) == 5849


def test_spy_closes_agree_once_both_feeds_are_modern():
    generated = {row[0]: row for row in _generated_daily("SPY")}
    reference = {row[0]: row for row in read_daily_csv(BUNDLED / "daily" / "spy.zip")}
    recent = [day for day in generated if day >= date(2017, 1, 1)]
    assert len(recent) > 1000

    within_half_a_cent = [day for day in recent if abs(generated[day][4] - reference[day][4]) <= 0.005]
    assert len(within_half_a_cent) / len(recent) > 0.99  # observed 99.8%
    worst = max(abs(generated[day][4] - reference[day][4]) / reference[day][4] for day in recent)
    assert worst < 0.0005, f"worst modern close disagreement {worst:.2e}"


def test_spy_early_closes_stay_within_vendor_noise():
    generated = {row[0]: row for row in _generated_daily("SPY")}
    reference = {row[0]: row for row in read_daily_csv(BUNDLED / "daily" / "spy.zip")}
    shared = sorted(set(generated) & set(reference))
    relative = [abs(generated[day][4] - reference[day][4]) / reference[day][4] for day in shared]
    assert sum(relative) / len(relative) < 0.002  # observed 0.0008


def test_spy_opens_rule_out_a_calendar_shift():
    # A one-day shift would put a whole day's move -- roughly 1% for SPY --
    # into every comparison. The opens sit two orders of magnitude below
    # that, and below the closes' own spread, so the two files describe the
    # same sessions and the close disagreement is a vendor difference.
    generated = {row[0]: row for row in _generated_daily("SPY")}
    reference = {row[0]: row for row in read_daily_csv(BUNDLED / "daily" / "spy.zip")}
    shared = sorted(set(generated) & set(reference))

    def mean_relative(column):
        return sum(
            abs(generated[day][column] - reference[day][column]) / reference[day][column]
            for day in shared
        ) / len(shared)

    assert mean_relative(1) < 0.0005  # opens, observed 0.018%
    assert mean_relative(1) < mean_relative(4)  # and tighter than the closes


# --- AAPL: the split chain -------------------------------------------------


def test_aapl_split_factors_match_the_bundled_chain():
    _, rows = snapshot_rows("AAPL")
    bundled = bundled_factor_rows("AAPL")

    checked = 0
    for row in rows:
        key = row.date.strftime("%Y%m%d")
        if key not in bundled:
            continue
        assert abs(row.split_factor - bundled[key][1]) < Decimal("1e-6"), key
        checked += 1
    assert checked > 30

    # 1/112, 1/56, 1/28 and 1/4 -- the cumulative effect of the 1998, 2000,
    # 2005, 2014 and 2020 splits.
    factors = {row.split_factor for row in rows}
    assert Decimal("0.25") in factors
    assert any(abs(value - Decimal(1) / Decimal(112)) < Decimal("1e-8") for value in factors)


def test_aapl_recent_factor_rows_are_byte_identical():
    # Where the vendor inputs agree, the output agrees exactly -- this is the
    # strongest evidence that the formatting and the walk are both right.
    _, rows = snapshot_rows("AAPL")
    rendered = {row.render().split(",")[0]: row.render() for row in rows}
    bundled_lines = (BUNDLED / "factor_files" / "aapl.csv").read_bytes().decode("ascii").splitlines()
    bundled = {line.split(",")[0]: line for line in bundled_lines}

    for key in ("20200507", "20200806", "20200828"):
        assert rendered[key] == bundled[key], key


def test_aapl_bars_are_un_split_adjusted():
    # Yahoo reports 2021-03-31 as roughly $122 only *after* un-adjusting the
    # 2020 4-for-1 split back out; the adjusted feed would read about $30.
    generated = {row[0]: row for row in _generated_daily("AAPL")}
    reference = {row[0]: row for row in read_daily_csv(BUNDLED / "daily" / "aapl.zip")}
    day = date(2021, 3, 31)
    assert round(generated[day][4], 2) == round(reference[day][4], 2) == 122.15


def _generated_daily(ticker: str):
    """Round-trip the fixture through the writer so the test reads real bytes."""
    import tempfile

    from leandata.lean.writer import LeanDataWriter
    from leandata.model import Provenance, Resolution, SecurityHistory, SymbolSpec

    bars, dividends, splits = read_snapshot(FIXTURES / f"{ticker.lower()}_yahoo_daily_{WINDOW}.csv")
    history = SecurityHistory(
        symbol=SymbolSpec(ticker),
        resolution=Resolution.DAILY,
        bars=bars,
        dividends=dividends,
        splits=splits,
    )
    with tempfile.TemporaryDirectory() as directory:
        report = LeanDataWriter(Path(directory)).write(history)
        return read_daily_csv(report.bar_files[0])
