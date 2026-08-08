"""Builders for leandata test data.

Hand-written rather than generated, in the same spirit as ``lean_stubs``:
every test states the exact prices it depends on so the expected numbers can
be checked by hand.
"""

import pandas as pd

from leandata.model import BAR_COLUMNS, Resolution, SecurityHistory, SymbolSpec

# A Monday, so N consecutive business days stay inside one calendar month.
DEFAULT_START = "2020-01-06"


def business_days(count: int, start: str = DEFAULT_START) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.bdate_range(start=start, periods=count), name="time")


def make_bars(closes, start: str = DEFAULT_START, volume: float = 1000.0, index=None) -> pd.DataFrame:
    """Flat bars where open == high == low == close, one per business day."""
    index = index if index is not None else business_days(len(closes), start)
    frame = pd.DataFrame(
        {
            "open": [float(value) for value in closes],
            "high": [float(value) for value in closes],
            "low": [float(value) for value in closes],
            "close": [float(value) for value in closes],
            "volume": [float(volume)] * len(closes),
        },
        index=index,
    )
    return frame[list(BAR_COLUMNS)]


def make_dividends(events: dict) -> pd.DataFrame:
    """{'2020-01-09': 1.5} -> the canonical dividends frame."""
    index = pd.DatetimeIndex([pd.Timestamp(day) for day in events], name="ex_date")
    frame = pd.DataFrame({"amount": [float(value) for value in events.values()]}, index=index)
    return frame.sort_index()


def make_splits(events: dict) -> pd.DataFrame:
    """{'2020-01-09': 2.0} -> the canonical splits frame (2.0 == 2-for-1)."""
    index = pd.DatetimeIndex([pd.Timestamp(day) for day in events], name="ex_date")
    frame = pd.DataFrame({"ratio": [float(value) for value in events.values()]}, index=index)
    return frame.sort_index()


def make_history(bars=None, dividends=None, splits=None, ticker="SPY", **symbol_options) -> SecurityHistory:
    from leandata.model import empty_dividends, empty_splits

    return SecurityHistory(
        symbol=SymbolSpec(ticker=ticker, **symbol_options),
        resolution=Resolution.DAILY,
        bars=make_bars([10.0, 11.0, 12.0]) if bars is None else bars,
        dividends=empty_dividends() if dividends is None else dividends,
        splits=empty_splits() if splits is None else splits,
    )
