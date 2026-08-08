"""Factor file construction, driven by hand-computable histories.

Bars are flat (open == high == low == close) so the arithmetic in each
comment can be checked without reading the builder.
"""

from datetime import date
from decimal import Decimal
from fractions import Fraction

import pandas as pd
import pytest

from leandata.errors import ValidationError
from leandata.lean.factors import TERMINAL_DATE, build_factor_rows, render_factor_file
from leandata_helpers import make_bars, make_dividends, make_splits
from leandata.model import empty_dividends, empty_splits

# 2020-01-06 .. 2020-01-17, ten business days.
CLOSES = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]


def build(dividends=None, splits=None, closes=None, **options):
    return build_factor_rows(
        make_bars(closes if closes is not None else CLOSES),
        dividends if dividends is not None else empty_dividends(),
        splits if splits is not None else empty_splits(),
        **options,
    )


def test_no_corporate_actions_gives_two_anchor_rows():
    rows, warnings = build()
    assert warnings == []
    assert [row.date for row in rows] == [date(2020, 1, 6), TERMINAL_DATE]
    assert rows[0].price_factor == 1 and rows[0].split_factor == 1
    # The oldest row's reference price is a marker, not a close -- every
    # bundled file (spy, aapl, fb) carries 1 there.
    assert rows[0].reference_price == 1
    assert rows[1].reference_price == 0


def test_a_dividend_writes_a_row_on_the_previous_trading_day():
    # Ex-date 2020-01-09, prior trading day 2020-01-08 closing at 12.0.
    rows, warnings = build(dividends=make_dividends({"2020-01-09": 1.5}))
    assert warnings == []
    assert [row.date for row in rows] == [date(2020, 1, 6), date(2020, 1, 8), TERMINAL_DATE]

    expected = Fraction(12) - Fraction(3, 2)  # (C - D) / C = 10.5 / 12
    expected = Decimal(expected.numerator) / Decimal(12 * expected.denominator) * 12
    assert rows[1].price_factor == Decimal("10.5") / Decimal("12")
    assert rows[1].reference_price == Decimal("12")
    # The synthetic oldest row inherits the factor and keeps reference 1.
    assert rows[0].price_factor == rows[1].price_factor
    assert rows[0].reference_price == 1


def test_dividends_compound_from_newest_to_oldest():
    rows, _ = build(dividends=make_dividends({"2020-01-09": 1.5, "2020-01-15": 2.0}))
    # 2020-01-15's prior day is 2020-01-14 closing 16.0 -> 14/16.
    newer = Decimal("14") / Decimal("16")
    # 2020-01-09's prior day is 2020-01-08 closing 12.0, applied on top.
    older = newer * (Decimal("10.5") / Decimal("12"))
    assert rows[2].date == date(2020, 1, 14) and rows[2].price_factor == newer
    assert rows[1].date == date(2020, 1, 8) and rows[1].price_factor == older
    assert rows[0].price_factor == older


def test_multiple_dividends_on_one_ex_date_are_summed():
    doubled = pd.DataFrame(
        {"amount": [1.0, 0.5]},
        index=pd.DatetimeIndex([pd.Timestamp("2020-01-09")] * 2, name="ex_date"),
    )
    rows, _ = build(dividends=doubled)
    assert rows[1].price_factor == Decimal("10.5") / Decimal("12")


def test_a_two_for_one_split_halves_the_split_factor():
    rows, _ = build(splits=make_splits({"2020-01-09": 2.0}))
    assert [row.date for row in rows] == [date(2020, 1, 6), date(2020, 1, 8), TERMINAL_DATE]
    assert rows[1].split_factor == Decimal("0.5")
    assert rows[1].price_factor == 1
    assert rows[0].split_factor == Decimal("0.5")


def test_split_factors_compound_and_a_reverse_split_inverts():
    rows, _ = build(splits=make_splits({"2020-01-09": 2.0, "2020-01-15": 0.5}))
    # Newest first: the 1-for-2 reverse doubles the factor going back.
    assert rows[2].date == date(2020, 1, 14) and rows[2].split_factor == Decimal("2")
    assert rows[1].date == date(2020, 1, 8) and rows[1].split_factor == Decimal("1")


def test_a_one_for_one_split_is_not_a_factor_row():
    rows, warnings = build(splits=make_splits({"2020-01-09": 1.0}))
    assert [row.date for row in rows] == [date(2020, 1, 6), TERMINAL_DATE]
    assert warnings == []


def test_same_day_dividend_and_split_share_one_row():
    rows, _ = build(
        dividends=make_dividends({"2020-01-09": 1.5}),
        splits=make_splits({"2020-01-09": 2.0}),
    )
    assert [row.date for row in rows] == [date(2020, 1, 6), date(2020, 1, 8), TERMINAL_DATE]
    # The dividend uses the raw prior close, independent of the split leg --
    # which is where FactorFileGenerator.cs and the runtime disagree.
    assert rows[1].price_factor == Decimal("10.5") / Decimal("12")
    assert rows[1].split_factor == Decimal("0.5")


def test_an_action_before_the_first_bar_stops_the_walk():
    rows, warnings = build(dividends=make_dividends({"2020-01-06": 1.0, "2020-01-09": 1.5}))
    assert any("stopping the factor walk" in message for message in warnings)
    # The 2020-01-09 dividend still made it in; nothing older can.
    assert [row.date for row in rows] == [date(2020, 1, 6), date(2020, 1, 8), TERMINAL_DATE]
    assert rows[0].reference_price == 1


def test_an_action_after_the_last_bar_is_dropped():
    rows, warnings = build(dividends=make_dividends({"2020-02-03": 1.0}))
    assert any("after the last bar" in message for message in warnings)
    assert [row.date for row in rows] == [date(2020, 1, 6), TERMINAL_DATE]


@pytest.mark.parametrize(
    "policy,expected_dates,fragment",
    [
        ("snap", [date(2020, 1, 6), date(2020, 1, 10), TERMINAL_DATE], "snapping it to 2020-01-13"),
        ("skip", [date(2020, 1, 6), TERMINAL_DATE], "dropping it"),
    ],
)
def test_an_action_off_the_trading_calendar_follows_the_policy(policy, expected_dates, fragment):
    # 2020-01-11 is a Saturday; the next trading day is Monday 2020-01-13.
    rows, warnings = build(
        dividends=make_dividends({"2020-01-11": 1.0}), on_missing_event_day=policy
    )
    assert [row.date for row in rows] == expected_dates
    assert any(fragment in message for message in warnings)


def test_the_error_policy_refuses_an_action_off_the_calendar():
    with pytest.raises(ValidationError, match="does not fall on a trading day"):
        build(dividends=make_dividends({"2020-01-11": 1.0}), on_missing_event_day="error")


def test_a_dividend_at_or_above_the_prior_close_is_ignored():
    rows, warnings = build(dividends=make_dividends({"2020-01-09": 12.0}))
    assert any("ignoring it" in message for message in warnings)
    assert all(row.price_factor == 1 for row in rows)


def test_rendering_matches_the_bundled_file_shape():
    rows, _ = build(dividends=make_dividends({"2020-01-09": 1.5}))
    payload = render_factor_file(rows)
    assert payload.endswith(b"20501231,1,1,0")  # no trailing newline
    assert b"\r\n" in payload
    lines = payload.decode("ascii").split("\r\n")
    assert lines[0] == "20200106,0.875,1,1"
    assert lines[1] == "20200108,0.875,1,12"
