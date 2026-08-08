"""A stand-in for YFinanceClient, plus a builder for Yahoo-shaped frames.

Injected through the constructor rather than patched into ``sys.modules``
like ``lean_stubs`` does: masking a real yfinance install globally would hide
the one thing the network-marked smoke test is there to check.
"""

import numpy as np
import pandas as pd

YAHOO_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]


def yahoo_frame(
    closes,
    start="2020-01-06",
    timezone="America/New_York",
    volume=1000.0,
    dividends=None,
    splits=None,
    multi_index=False,
    ticker="SPY",
):
    """Build the frame ``Ticker.history(auto_adjust=False, actions=True)`` returns.

    Prices are flat (open == high == low == close) unless a test overwrites
    them afterwards. Remember that Yahoo's OHLCV here is *split-adjusted*.
    """
    index = pd.DatetimeIndex(pd.bdate_range(start=start, periods=len(closes)))
    if timezone:
        index = index.tz_localize(timezone)

    frame = pd.DataFrame(
        {
            "Open": [float(value) for value in closes],
            "High": [float(value) for value in closes],
            "Low": [float(value) for value in closes],
            "Close": [float(value) for value in closes],
            "Adj Close": [float(value) * 0.9 for value in closes],
            "Volume": [float(volume)] * len(closes),
            "Dividends": [0.0] * len(closes),
            "Stock Splits": [0.0] * len(closes),
        },
        index=index,
    )
    for offset, amount in (dividends or {}).items():
        frame.iloc[offset, frame.columns.get_loc("Dividends")] = float(amount)
    for offset, ratio in (splits or {}).items():
        frame.iloc[offset, frame.columns.get_loc("Stock Splits")] = float(ratio)

    if multi_index:
        frame.columns = pd.MultiIndex.from_product([frame.columns, [ticker]])
    return frame


def empty_yahoo_frame():
    return pd.DataFrame(columns=YAHOO_COLUMNS, index=pd.DatetimeIndex([]))


class FakeYFinanceClient:
    """Returns a canned frame and records what it was asked for."""

    def __init__(self, frame, *, version="fake-1.0", repair=False):
        self.frame = frame
        self.version = version
        self.repair = repair
        self.calls = []

    def history(self, ticker, start, end, interval):
        self.calls.append({"ticker": ticker, "start": start, "end": end, "interval": interval})
        return self.frame.copy()


def set_cell(frame, row, column, value):
    frame.iloc[row, frame.columns.get_loc(column)] = value
    return frame


NAN = np.nan
