"""The yfinance adapter's transformations, driven entirely by a fake client."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from leandata.errors import EmptyHistoryError, SourceError, UnsupportedResolutionError
from leandata.model import Resolution
from leandata.sources.base import FetchRequest
from leandata.sources.yfinance_source import YFinanceSource
from yfinance_fakes import FakeYFinanceClient, empty_yahoo_frame, set_cell, yahoo_frame


def fetch(frame, **request_options):
    source = YFinanceSource(FakeYFinanceClient(frame), **request_options.pop("source_options", {}))
    history = source.fetch(FetchRequest(ticker="SPY", **request_options))
    return history, source


def test_exchange_local_index_becomes_naive_midnight():
    history, _ = fetch(yahoo_frame([10.0, 11.0]))
    assert history.bars.index.tz is None
    assert [stamp.isoformat() for stamp in history.bars.index] == [
        "2020-01-06T00:00:00",
        "2020-01-07T00:00:00",
    ]


def test_a_utc_stamped_feed_is_converted_before_the_zone_is_dropped():
    # 05:00 UTC is midnight in New York; localising without converting would
    # keep 05:00, and a pre-open stamp would land on the wrong calendar day.
    frame = yahoo_frame([10.0, 11.0], timezone=None)
    frame.index = pd.DatetimeIndex(frame.index + pd.Timedelta(hours=5)).tz_localize("UTC")
    history, _ = fetch(frame)
    assert [stamp.date().isoformat() for stamp in history.bars.index] == ["2020-01-06", "2020-01-07"]


def test_naive_index_is_accepted_as_exchange_local():
    history, _ = fetch(yahoo_frame([10.0, 11.0], timezone=None))
    assert history.bars.index[0] == pd.Timestamp("2020-01-06")


def test_multiindex_columns_are_flattened():
    history, _ = fetch(yahoo_frame([10.0, 11.0], multi_index=True))
    assert tuple(history.bars.columns) == ("open", "high", "low", "close", "volume")


def test_adj_close_is_dropped_rather_than_used():
    # Adj Close in the fake is 0.9 * close; if it leaked in, the prices below
    # would be 9.0 and 9.9.
    history, _ = fetch(yahoo_frame([10.0, 11.0]))
    assert list(history.bars["close"]) == [10.0, 11.0]


def test_a_missing_bar_column_is_an_error():
    frame = yahoo_frame([10.0, 11.0]).drop(columns=["High"])
    with pytest.raises(SourceError, match="missing column"):
        fetch(frame)


def test_rows_with_missing_prices_are_dropped_not_filled():
    frame = set_cell(yahoo_frame([10.0, 11.0, 12.0]), 1, "Close", np.nan)
    history, source = fetch(frame)
    assert list(history.bars["close"]) == [10.0, 12.0]
    assert any("missing or non-positive prices" in note for note in source.warnings)


def test_zero_volume_rows_survive():
    frame = set_cell(yahoo_frame([10.0, 11.0]), 0, "Volume", 0.0)
    history, _ = fetch(frame)
    assert list(history.bars["volume"]) == [0.0, 1000.0]


def test_missing_volume_is_filled_with_zero():
    frame = set_cell(yahoo_frame([10.0, 11.0]), 0, "Volume", np.nan)
    history, source = fetch(frame)
    assert list(history.bars["volume"]) == [0.0, 1000.0]
    assert any("filled missing volume" in note for note in source.warnings)


def test_an_impossible_range_is_clamped_by_default():
    frame = set_cell(yahoo_frame([10.0, 11.0]), 1, "High", 10.5)  # below the close
    history, source = fetch(frame)
    assert history.bars["high"].iloc[1] == 11.0
    assert any("widened high/low" in note for note in source.warnings)


def test_an_impossible_range_can_be_dropped_or_raised():
    frame = set_cell(yahoo_frame([10.0, 11.0]), 1, "High", 10.5)
    history, _ = fetch(frame, source_options={"on_invalid_bar": "drop"})
    assert len(history.bars) == 1

    frame = set_cell(yahoo_frame([10.0, 11.0]), 1, "High", 10.5)
    with pytest.raises(SourceError, match="impossible high/low range"):
        fetch(frame, source_options={"on_invalid_bar": "error"})


def test_a_zero_in_the_stock_splits_column_is_not_a_split():
    # Yahoo writes 0.0 for "no split", so filtering on notna() would invent one.
    history, _ = fetch(yahoo_frame([10.0, 11.0]))
    assert history.splits.empty


def test_dividends_are_extracted_from_the_actions_column():
    history, _ = fetch(yahoo_frame([10.0, 11.0, 12.0], dividends={1: 0.5}))
    assert list(history.dividends["amount"]) == [0.5]
    assert history.dividends.index[0] == pd.Timestamp("2020-01-07")


def test_a_split_un_adjusts_prices_volume_and_dividends():
    # Raw history: 100 for two days, then a 2-for-1 on day 3 takes it to 50.
    # Yahoo reports every close split-adjusted, so all four read 50, and the
    # pre-split volume reads double. A $2 raw dividend on day 2 reads $1.
    frame = yahoo_frame(
        [50.0, 50.0, 50.0, 50.0], volume=2000.0, dividends={1: 1.0}, splits={2: 2.0}
    )
    frame = set_cell(frame, 2, "Volume", 1000.0)
    frame = set_cell(frame, 3, "Volume", 1000.0)

    history, _ = fetch(frame)
    assert list(history.bars["close"]) == [100.0, 100.0, 50.0, 50.0]
    assert list(history.bars["volume"]) == [1000.0, 1000.0, 1000.0, 1000.0]
    # Strictly-after semantics: the dividend on day 2 predates the split.
    assert list(history.dividends["amount"]) == [2.0]
    assert list(history.splits["ratio"]) == [2.0]


def test_a_dividend_on_the_split_ex_date_is_already_post_split():
    frame = yahoo_frame([50.0, 50.0, 50.0], dividends={2: 1.0}, splits={2: 2.0})
    history, _ = fetch(frame)
    assert list(history.dividends["amount"]) == [1.0]


def test_an_empty_response_is_reported_as_such():
    with pytest.raises(EmptyHistoryError, match="no daily bars for SPY"):
        fetch(empty_yahoo_frame())


def test_only_daily_is_supported_today():
    with pytest.raises(UnsupportedResolutionError, match="supported: daily"):
        fetch(yahoo_frame([10.0]), resolution=Resolution.MINUTE)


def test_the_request_window_is_passed_through_and_clipped_inclusively():
    client = FakeYFinanceClient(yahoo_frame([10.0, 11.0, 12.0, 13.0]))
    source = YFinanceSource(client)
    history = source.fetch(
        FetchRequest(ticker="SPY", start=date(2020, 1, 7), end=date(2020, 1, 8))
    )
    assert client.calls[0]["interval"] == "1d"
    assert client.calls[0]["start"] == date(2020, 1, 7)
    assert [stamp.date().isoformat() for stamp in history.bars.index] == ["2020-01-07", "2020-01-08"]


def test_provenance_records_how_the_data_was_pulled():
    history, _ = fetch(yahoo_frame([10.0, 11.0]))
    assert history.provenance.source == "yfinance"
    assert history.provenance.options["auto_adjust"] == "False"
    assert history.provenance.options["repair"] == "False"
