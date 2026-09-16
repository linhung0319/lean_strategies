"""The constituents-universe half of the converter.

Three pieces, each tested against something external rather than against
itself:

``identifiers``
    pinned to security identifiers copied out of LEAN's own bundled SPY
    universe file, so a change in the packing arithmetic fails loudly;
``universes``
    pinned to the byte layout of that same file;
``weights``
    hand-computed market caps, including the split case that is the only part
    of the reconstruction that can be wrong in a way you would not notice.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from leandata.errors import SourceError, ValidationError
from leandata.lean.identifiers import encode_base36, equity_security_id
from leandata.lean.universes import UniverseWriter, render_number, render_universe_file
from leandata.model import ConstituentPanel, IndexMembership, SymbolSpec
from leandata.sources.sp500_github import parse_membership_csv
from leandata.tickers import lean_ticker, yahoo_ticker
from leandata.weights import (
    cap_weighted_panel,
    cumulative_split_factor,
    restated_market_caps,
)
from leandata_helpers import make_bars, make_history, make_splits

# Lean/Data/equity/usa/universes/etf/spy/20201130.csv, paired with the first
# row of the matching map file where LEAN ships one.
BUNDLED_IDENTIFIERS = {
    ("AAPL", date(1998, 1, 2)): "AAPL R735QTJ8XC9X",
    ("A", date(1999, 11, 18)): "A RPTMYV3VC57P",
    ("AAL", date(2013, 12, 9)): "AAL VM9RIYHM8ACL",
    ("BRKB", date(1998, 1, 2)): "BRKB R735QTJ8XC9X",
}


# --- identifiers ----------------------------------------------------------


def test_base36_matches_leans_encoding():
    assert encode_base36(0) == "0"
    assert encode_base36(35) == "Z"
    assert encode_base36(36) == "10"
    assert encode_base36(3579700000000000101) == "R735QTJ8XC9X"


@pytest.mark.parametrize(("ticker", "first_date", "expected"), [
    (ticker, first_date, expected)
    for (ticker, first_date), expected in BUNDLED_IDENTIFIERS.items()
])
def test_security_ids_match_leans_own_universe_file(ticker, first_date, expected):
    assert equity_security_id(ticker, first_date) == expected


def test_the_first_date_is_what_distinguishes_two_identifiers():
    # Which is why the SID has to be built from the map file the writer
    # produced, not from the window the caller asked for.
    assert equity_security_id("AAPL", date(1998, 1, 2)) != equity_security_id(
        "AAPL", date(1998, 1, 5)
    )


@pytest.mark.parametrize("ticker", ["", "   ", "TOOLONGATICKER", "BRK B", "A|B"])
def test_unusable_tickers_are_rejected_rather_than_silently_mangled(ticker):
    with pytest.raises(ValidationError):
        equity_security_id(ticker, date(2020, 1, 2))


def test_an_unknown_market_is_rejected():
    with pytest.raises(ValidationError):
        equity_security_id("AAPL", date(1998, 1, 2), market="lse")


# --- ticker spellings -----------------------------------------------------


def test_the_three_spellings_of_a_class_share():
    assert lean_ticker("BRK.B") == "BRKB"
    assert yahoo_ticker("BRK.B") == "BRK-B"
    assert lean_ticker("aapl") == "AAPL"
    assert yahoo_ticker("AAPL") == "AAPL"


# --- membership -----------------------------------------------------------


MEMBERSHIP_CSV = (
    "date,tickers\n"
    '2020-01-02,"AAPL,MSFT,XOM"\n'
    '2020-06-01,"AAPL,MSFT,NVDA"\n'
)


def test_membership_csv_becomes_sorted_change_rows():
    changes = parse_membership_csv(MEMBERSHIP_CSV)
    assert changes == (
        (date(2020, 1, 2), ("AAPL", "MSFT", "XOM")),
        (date(2020, 6, 1), ("AAPL", "MSFT", "NVDA")),
    )


def test_membership_rows_stay_in_force_until_the_next_one():
    membership = IndexMembership("test", parse_membership_csv(MEMBERSHIP_CSV))
    assert membership.members_on(date(2019, 12, 31)) == ()
    assert membership.members_on(date(2020, 3, 1)) == ("AAPL", "MSFT", "XOM")
    # Past the last row the roster is carried forward, not emptied.
    assert membership.members_on(date(2030, 1, 1)) == ("AAPL", "MSFT", "NVDA")


def test_the_union_is_the_download_list_for_a_window():
    membership = IndexMembership("test", parse_membership_csv(MEMBERSHIP_CSV))
    # XOM left in June, but a backtest opening in March could still hold it.
    assert membership.union(date(2020, 3, 1), date(2020, 12, 31)) == (
        "AAPL", "MSFT", "NVDA", "XOM",
    )
    # A window entirely after the change does not need it.
    assert membership.union(date(2020, 7, 1), date(2020, 12, 31)) == (
        "AAPL", "MSFT", "NVDA",
    )


def test_a_csv_without_the_expected_columns_is_rejected():
    with pytest.raises(SourceError):
        parse_membership_csv("day,names\n2020-01-02,AAPL\n")


# --- weights --------------------------------------------------------------


def test_the_split_factor_excludes_the_ex_date_itself():
    index = pd.DatetimeIndex(["2020-08-28", "2020-08-31", "2020-09-01"])
    splits = make_splits({"2020-08-31": 4.0})
    # A bar dated on the ex-date is already quoted post-split, so only the
    # bar before it needs multiplying up.
    assert list(cumulative_split_factor(index, splits)) == [4.0, 1.0, 1.0]


def test_market_cap_survives_a_split_between_two_filings():
    # Four sessions at a flat 100, then a 4-for-1 on the third, so the raw
    # close halves-and-halves again to 25. One share count, filed before the
    # split: 1,000,000 shares. The cap must stay 100,000,000 throughout --
    # multiplying the stale count by the post-split price would show a
    # quarter of the company disappearing.
    index = pd.DatetimeIndex(pd.bdate_range("2020-01-06", periods=4), name="time")
    history = make_history(
        bars=make_bars([100.0, 100.0, 25.0, 25.0], index=index),
        splits=make_splits({"2020-01-08": 4.0}),
        ticker="SPLITCO",
    )
    shares = pd.Series([1_000_000.0], index=pd.DatetimeIndex(["2020-01-06"], name="time"))

    caps = restated_market_caps({"SPLITCO": history}, {"SPLITCO": shares}, index)
    assert list(caps["SPLITCO"]) == [1e8, 1e8, 1e8, 1e8]


def test_a_ticker_with_no_share_count_has_no_market_cap():
    index = pd.DatetimeIndex(pd.bdate_range("2020-01-06", periods=2), name="time")
    history = make_history(bars=make_bars([10.0, 11.0], index=index), ticker="QUIET")
    caps = restated_market_caps({"QUIET": history}, {}, index)
    assert caps["QUIET"].isna().all()


def caps_index(index):
    index = pd.DatetimeIndex(index)
    index.name = "date"
    return index


def _panel_fixture():
    caps = pd.DataFrame(
        {"AAA": [300.0, 300.0], "BBB": [100.0, 100.0]},
        index=caps_index(pd.bdate_range("2020-01-06", periods=2)),
    )
    membership = IndexMembership("test", ((date(2020, 1, 6), ("AAA", "BBB", "CCC")),))
    return membership, caps


def test_weights_are_market_cap_shares_of_that_days_roster():
    membership, caps = _panel_fixture()
    panel = cap_weighted_panel(SymbolSpec(ticker="SPY"), membership, caps)

    first_day = panel.rows.xs(panel.rows.index.get_level_values("date")[0], level="date")
    assert first_day.loc["AAA", "weight"] == pytest.approx(0.75)
    assert first_day.loc["BBB", "weight"] == pytest.approx(0.25)
    # CCC is in the index but has no cap, so it is listed without a weight
    # rather than dropped -- the file still says who was a member.
    assert np.isnan(first_day.loc["CCC", "weight"])


def test_renaming_carries_index_spellings_onto_lean_ones():
    membership = IndexMembership("test", ((date(2020, 1, 6), ("BRK.B",)),))
    caps = pd.DataFrame({"BRK.B": [100.0]}, index=caps_index(["2020-01-06"]))
    panel = cap_weighted_panel(
        SymbolSpec(ticker="SPY"), membership, caps, rename={"BRK.B": "BRKB"}
    )
    assert panel.tickers == ("BRKB",)


def test_a_calendar_that_predates_the_index_produces_nothing():
    membership = IndexMembership("test", ((date(2021, 1, 4), ("AAA",)),))
    caps = pd.DataFrame({"AAA": [1.0]}, index=caps_index(["2020-01-06"]))
    with pytest.raises(ValidationError):
        cap_weighted_panel(SymbolSpec(ticker="SPY"), membership, caps)


# --- universe files -------------------------------------------------------


def test_an_unknown_number_is_written_as_an_empty_field():
    assert render_number(0.06213795) == "0.06213795"
    assert render_number(float("nan")) == ""
    assert render_number(None) == ""


def test_a_universe_line_matches_the_bundled_file_byte_for_byte():
    # Lean/Data/equity/usa/universes/etf/spy/20201130.csv, minus the shares
    # held that this pipeline does not know.
    payload = render_universe_file(
        [("AAPL", "AAPL R735QTJ8XC9X", 0.06213795, float("nan"), float("nan"))],
        date(2020, 11, 27),
    )
    assert payload == b"AAPL,AAPL R735QTJ8XC9X,20201127,0.06213795,,\r\n"


def test_rows_are_written_in_ticker_order():
    payload = render_universe_file(
        [
            ("MSFT", "MSFT X", 0.5, float("nan"), float("nan")),
            ("AAPL", "AAPL X", 0.5, float("nan"), float("nan")),
        ],
        date(2020, 11, 27),
    )
    assert payload.split(b"\r\n")[0].startswith(b"AAPL,")


def test_an_empty_file_is_refused():
    with pytest.raises(ValidationError):
        render_universe_file([], date(2020, 11, 27))


def _small_panel():
    membership = IndexMembership("test", ((date(2020, 1, 6), ("AAA", "BBB")),))
    caps = pd.DataFrame(
        {"AAA": [300.0, 300.0], "BBB": [100.0, 100.0]},
        index=caps_index(pd.bdate_range("2020-01-06", periods=2)),
    )
    return cap_weighted_panel(SymbolSpec(ticker="SPY"), membership, caps)


def test_the_writer_lays_files_out_where_lean_looks_for_them(tmp_path):
    panel = _small_panel()
    ids = {"AAA": equity_security_id("AAA", date(2020, 1, 6)),
           "BBB": equity_security_id("BBB", date(2020, 1, 6))}

    report = UniverseWriter(tmp_path).write(panel, ids)

    folder = tmp_path / "equity" / "usa" / "universes" / "etf" / "spy"
    assert sorted(path.name for path in folder.glob("*.csv")) == ["20200106.csv", "20200107.csv"]
    assert report.files == 2
    assert report.rows == 4
    assert report.unpriced_rows == 0


def test_the_writer_refuses_a_panel_it_has_no_identifier_for(tmp_path):
    with pytest.raises(ValidationError, match="no security id"):
        UniverseWriter(tmp_path).write(_small_panel(), {"AAA": "AAA X"})


def test_a_panel_rejects_the_wrong_index_shape():
    rows = pd.DataFrame(
        {"weight": [1.0], "shares_held": [np.nan], "market_value": [np.nan]},
        index=pd.Index(["AAA"], name="ticker"),
    )
    with pytest.raises(ValidationError):
        ConstituentPanel(etf=SymbolSpec(ticker="SPY"), rows=rows)
