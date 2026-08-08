"""Every literal here was lifted from a bundled LEAN data file, so these
tests fail the moment the .NET decimal emulation drifts."""

from decimal import Decimal

import pytest

from leandata.lean.formatting import (
    format_volume,
    net_round,
    scale_price,
    strip_zeros,
    to_decimal,
    to_str,
)


def D(text):
    return Decimal(text)


def test_to_decimal_uses_the_shortest_string_form():
    # Decimal(0.1) is 0.1000000000000000055511151231257827; that noise would
    # end up in every price factor.
    assert to_decimal(0.1) == D("0.1")
    assert to_decimal(97.31) == D("97.31")
    assert to_decimal(D("1.5")) == D("1.5")


@pytest.mark.parametrize(
    "value,places,expected",
    [
        # Math.Round only ever reduces scale, so a decimal that already has
        # fewer places than requested comes back untouched. This is why
        # aapl.csv prints a split factor of 0.25 as "0.25" and not "0.25000000".
        ("0.25", 8, "0.25"),
        ("1", 7, "1"),
        ("1", 8, "1"),
        # And a division result with a long tail gets cut to exactly `places`,
        # trailing zero included: spy.csv row 20210318.
        ("0.99673601", 7, "0.9967360"),
        ("0.65666055555", 7, "0.6566606"),
        # Banker's rounding, matching C#'s MidpointRounding.ToEven default.
        ("0.12345675", 7, "0.1234568"),
        ("0.12345665", 7, "0.1234566"),
    ],
)
def test_net_round_matches_dotnet_math_round(value, places, expected):
    assert to_str(net_round(D(value), places)) == expected


def test_price_factor_from_a_real_division_reproduces_a_bundled_row():
    # The last dividend row of Lean/Data/equity/usa/factor_files/spy.csv is
    # "20210318,0.9967360,1,391.48". Inverting it gives the dividend QC used,
    # 1.27779; running that back through (C - D) / C has to land on the same
    # seven-place string, trailing zero included.
    factor = (D("391.48") - D("1.27779")) / D("391.48")
    assert to_str(net_round(factor, 7)) == "0.9967360"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("109.34", "109.34"),
        ("109.3400", "109.34"),
        ("1", "1"),
        ("0", "0"),
        ("0.0000", "0"),
        ("118.5", "118.5"),
        # The trap: normalize() alone gives 9.731E+5, which LEAN cannot parse.
        ("973100.00", "973100"),
    ],
)
def test_strip_zeros_never_emits_scientific_notation(value, expected):
    assert strip_zeros(D(value)) == expected


@pytest.mark.parametrize(
    "price,expected",
    [
        # spy.zip first row: 97.31 -> 973100 deci-cents.
        (97.31, "973100"),
        (97.53, "975300"),
        (395.35, "3953500"),
        (396.33, "3963300"),
        # Float noise from a JSON feed still lands on the integer.
        (97.31000137329102, "973100"),
        # Genuine sub-penny data survives: pre-decimalisation sixteenths.
        (97.3125, "973125"),
    ],
)
def test_scale_price_matches_bundled_rows(price, expected):
    assert scale_price(price) == expected


def test_volume_is_not_scaled():
    assert format_volume(2150000) == "2150000"
    assert format_volume(90673494.0) == "90673494"
    assert format_volume(0) == "0"
