"""The canonical contract is the only thing keeping adapters honest, so the
rejections matter as much as the acceptances."""

import numpy as np
import pandas as pd
import pytest

from leandata.errors import ValidationError
from leandata.model import BAR_COLUMNS, Resolution, SecurityHistory, SymbolSpec
from leandata_helpers import make_bars, make_dividends, make_history, make_splits


def build(bars):
    return SecurityHistory(symbol=SymbolSpec("SPY"), resolution=Resolution.DAILY, bars=bars)


def test_a_well_formed_history_is_accepted():
    history = make_history()
    assert history.first_date.isoformat() == "2020-01-06"
    assert history.last_date.isoformat() == "2020-01-08"


def test_symbol_key_is_the_lowercase_ticker():
    assert SymbolSpec(" spy ").ticker == "SPY"
    assert SymbolSpec("spy").key == "spy"


def test_empty_ticker_is_rejected():
    with pytest.raises(ValidationError, match="non-empty"):
        SymbolSpec("   ")


def test_tz_aware_index_is_rejected():
    bars = make_bars([10.0, 11.0])
    bars.index = bars.index.tz_localize("America/New_York")
    with pytest.raises(ValidationError, match="tz-naive"):
        build(bars)


def test_unsorted_index_is_rejected():
    bars = make_bars([10.0, 11.0, 12.0]).iloc[[2, 0, 1]]
    with pytest.raises(ValidationError, match="sorted oldest to newest"):
        build(bars)


def test_duplicate_timestamps_are_rejected():
    bars = make_bars([10.0, 11.0])
    bars.index = pd.DatetimeIndex([bars.index[0], bars.index[0]])
    with pytest.raises(ValidationError, match="unique"):
        build(bars)


def test_empty_bars_are_rejected():
    with pytest.raises(ValidationError, match="must not be empty"):
        build(make_bars([10.0]).iloc[:0])


def test_missing_and_extra_columns_are_rejected():
    with pytest.raises(ValidationError, match="columns must be exactly"):
        build(make_bars([10.0, 11.0]).drop(columns=["volume"]))
    extra = make_bars([10.0, 11.0])
    extra["adj_close"] = 1.0
    with pytest.raises(ValidationError, match="columns must be exactly"):
        build(extra)


def test_nan_price_is_rejected():
    bars = make_bars([10.0, 11.0])
    bars.iloc[1, bars.columns.get_loc("close")] = np.nan
    with pytest.raises(ValidationError, match="NaN or inf"):
        build(bars)


def test_non_positive_price_is_rejected():
    bars = make_bars([10.0, 11.0])
    bars.iloc[0, bars.columns.get_loc("low")] = 0.0
    with pytest.raises(ValidationError, match="strictly positive"):
        build(bars)


def test_negative_volume_is_rejected():
    bars = make_bars([10.0, 11.0])
    bars.iloc[0, bars.columns.get_loc("volume")] = -1.0
    with pytest.raises(ValidationError, match="volume must be >= 0"):
        build(bars)


def test_zero_volume_is_accepted():
    # Thin early ETF sessions have them, and dropping them would punch holes
    # in the trading calendar the factor file depends on.
    bars = make_bars([10.0, 11.0])
    bars.iloc[0, bars.columns.get_loc("volume")] = 0.0
    assert len(build(bars).bars) == 2


def test_impossible_high_low_range_is_rejected():
    bars = make_bars([10.0, 11.0])
    bars.iloc[1, bars.columns.get_loc("high")] = 10.5  # below close 11.0
    with pytest.raises(ValidationError, match="high must be >="):
        build(bars)

    bars = make_bars([10.0, 11.0])
    bars.iloc[1, bars.columns.get_loc("low")] = 11.5  # above close 11.0
    with pytest.raises(ValidationError, match="low must be <="):
        build(bars)


def test_non_positive_dividend_is_rejected():
    with pytest.raises(ValidationError, match="amount must be strictly positive"):
        make_history(dividends=make_dividends({"2020-01-07": 0.0}))


def test_non_positive_split_ratio_is_rejected():
    with pytest.raises(ValidationError, match="ratio must be strictly positive"):
        make_history(splits=make_splits({"2020-01-07": 0.0}))


def test_clip_boundaries_are_inclusive():
    from datetime import date

    history = make_history(bars=make_bars([10.0, 11.0, 12.0, 13.0, 14.0]))
    clipped = history.clip(date(2020, 1, 7), date(2020, 1, 9))
    assert [stamp.date().isoformat() for stamp in clipped.bars.index] == [
        "2020-01-07",
        "2020-01-08",
        "2020-01-09",
    ]


def test_clip_keeps_the_symbol_and_actions():
    history = make_history(
        bars=make_bars([10.0, 11.0, 12.0]),
        dividends=make_dividends({"2020-01-07": 0.5}),
    )
    clipped = history.clip(None, None)
    assert clipped.symbol == history.symbol
    assert len(clipped.dividends) == 1
    assert tuple(clipped.bars.columns) == BAR_COLUMNS
