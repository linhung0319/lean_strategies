"""A minimal stand-in for LEAN's ``AlgorithmImports``.

Enough of the API surface to drive the files in ``algorithms/`` as plain
Python and assert on the allocation decisions they make. This tests decision
logic only — order fills, fees, slippage and real warm-up are the engine's
job and are verified by running LEAN itself (see README.md).
"""

import sys
import types
from datetime import datetime, timedelta


class Resolution:
    DAILY = "Daily"
    HOUR = "Hour"
    MINUTE = "Minute"


class Symbol(str):
    """LEAN symbols behave as opaque keys; a str subclass is close enough."""


class TradeBar:
    def __init__(self, close):
        self.close = close
        self.open = close
        self.high = close
        self.low = close
        self.volume = 0


class _Bars(dict):
    def contains_key(self, key):
        return key in self


class Slice:
    def __init__(self, bars):
        self.bars = _Bars(bars)


class RollingWindow:
    """Mirrors LEAN's RollingWindow: index 0 is newest, count-1 is oldest."""

    def __class_getitem__(cls, item):
        return cls

    def __init__(self, size):
        self.size = size
        self._items = []

    def add(self, item):
        self._items.insert(0, item)
        if len(self._items) > self.size:
            self._items.pop()

    def __getitem__(self, index):
        return self._items[index]

    @property
    def count(self):
        return len(self._items)

    @property
    def is_ready(self):
        return len(self._items) == self.size


class _SimpleMovingAverage:
    def __init__(self, period):
        self.period = period
        self._values = []
        self.current = types.SimpleNamespace(value=0.0)

    @property
    def is_ready(self):
        return len(self._values) >= self.period

    def update(self, value):
        self._values.append(value)
        if len(self._values) > self.period:
            self._values.pop(0)
        self.current.value = sum(self._values) / len(self._values)


class _Holding:
    def __init__(self):
        self.quantity = 0.0
        self.holdings_value = 0.0

    @property
    def invested(self):
        return self.quantity != 0.0


class _Portfolio:
    def __init__(self):
        self.cash = 0.0
        self._holdings = {}

    def __getitem__(self, symbol):
        return self._holdings.setdefault(symbol, _Holding())

    @property
    def total_portfolio_value(self):
        return self.cash + sum(h.holdings_value for h in self._holdings.values())

    @property
    def invested(self):
        return any(h.invested for h in self._holdings.values())


class QCAlgorithm:
    def __init__(self):
        self._parameters = {}
        self._indicators = []
        self._price = 0.0
        self.portfolio = _Portfolio()
        self.time = datetime(1970, 1, 1)
        self.is_warming_up = False
        self.orders = []

    # -- configuration -------------------------------------------------
    def set_start_date(self, year, month, day):
        self.start_date = datetime(year, month, day)

    def set_end_date(self, year, month, day):
        self.end_date = datetime(year, month, day)

    def set_cash(self, amount):
        self.portfolio.cash = float(amount)

    def get_parameter(self, name, default=None):
        return self._parameters.get(name, default)

    def add_equity(self, ticker, resolution=None):
        return types.SimpleNamespace(symbol=Symbol(ticker))

    def set_benchmark(self, symbol):
        pass

    def set_warm_up(self, period, resolution=None):
        self.warm_up_period = period

    def sma(self, symbol, period, resolution=None):
        indicator = _SimpleMovingAverage(period)
        self._indicators.append(indicator)
        return indicator

    def log(self, message):
        pass

    def debug(self, message):
        pass

    # -- trading -------------------------------------------------------
    def set_holdings(self, symbol, weight):
        self.orders.append((self.time, symbol, float(weight)))
        total = self.portfolio.total_portfolio_value
        target_value = total * float(weight)
        holding = self.portfolio[symbol]
        holding.quantity = target_value / self._price if self._price else 0.0
        holding.holdings_value = target_value
        self.portfolio.cash = total - target_value

    # -- driven by feed() ----------------------------------------------
    def _mark_to_market(self, symbol, price):
        holding = self.portfolio[symbol]
        holding.holdings_value = holding.quantity * price


def feed(algorithm, bars, warmup_bars=0):
    """Push ``(datetime, close)`` pairs through the algorithm as SPY bars.

    The first ``warmup_bars`` are delivered with ``is_warming_up`` set, which
    is how LEAN behaves: ``on_data`` is still called and framework indicators
    still update, but the algorithm is expected to skip trading.
    """
    symbol = Symbol("SPY")
    for index, (when, close) in enumerate(bars):
        close = float(close)
        algorithm.time = when
        algorithm._price = close
        algorithm._mark_to_market(symbol, close)
        for indicator in algorithm._indicators:
            indicator.update(close)
        algorithm.is_warming_up = index < warmup_bars
        algorithm.on_data(Slice({symbol: TradeBar(close)}))


def run_algorithm(algorithm_class, prices, parameters=None, warmup_bars=0):
    """Construct, configure, initialise and drive an algorithm in one call.

    ``prices`` is a list of closes delivered as consecutive daily bars starting
    2020-01-01. Parameters are stringified the way LEAN delivers them. Returns
    the algorithm so tests can assert on ``.orders`` and portfolio state.

    Tests needing specific calendar dates should build bars themselves and call
    ``feed`` directly.
    """
    algorithm = algorithm_class()
    algorithm._parameters = {
        key: str(value) for key, value in (parameters or {}).items()
    }
    algorithm.initialize()
    start = datetime(2020, 1, 1)
    bars = [
        (start + timedelta(days=index), float(price))
        for index, price in enumerate(prices)
    ]
    feed(algorithm, bars, warmup_bars=warmup_bars)
    return algorithm


def install():
    """Register the fakes as the ``AlgorithmImports`` module."""
    module = types.ModuleType("AlgorithmImports")
    exported = [
        "QCAlgorithm",
        "Resolution",
        "RollingWindow",
        "Slice",
        "Symbol",
        "TradeBar",
    ]
    for name in exported:
        setattr(module, name, globals()[name])
    module.__all__ = exported
    sys.modules["AlgorithmImports"] = module
