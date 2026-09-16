from AlgorithmImports import *


class StockMonthlyRebalance(QCAlgorithm):
    """Hold a fixed fraction of one stock, pulled back to target every month.

    Built to be run twice -- once against a locally converted data folder,
    once on quantconnect.com -- so that the two results can be diffed. The
    rebalance dates come from the calendar, not from prices, so both runs
    trade on the same days by construction and any difference in the
    statistics is a difference in the data rather than a signal that flipped.

    Paste it into a QuantConnect project unedited: every parameter already
    defaults to the comparison run, the end date included.
    """

    def initialize(self):
        self.ticker = self.get_parameter("ticker", "NVDA")
        self._set_date("start_date", "2020-01-01", self.set_start_date)
        self._set_date("end_date", "2025-12-31", self.set_end_date)
        self.set_cash(10000)

        self.weight = float(self.get_parameter("weight", 0.60))
        self.stock = self.add_equity(self.ticker, Resolution.DAILY).symbol
        self.set_benchmark(self.stock)
        self._last_month = None

    def _set_date(self, name, default, apply):
        year, month, day = (int(part) for part in self.get_parameter(name, default).split("-"))
        apply(year, month, day)

    def on_data(self, data: Slice):
        if not data.bars.contains_key(self.stock):
            return
        month = (self.time.year, self.time.month)
        if month == self._last_month:
            return
        self._last_month = month
        self.set_holdings(self.stock, self.weight)
