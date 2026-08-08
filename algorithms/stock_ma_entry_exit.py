from AlgorithmImports import *


class StockMaEntryExit(QCAlgorithm):
    """Hold a single stock while it closes above its moving average.

    Same rule as spy_ma_entry_exit, but the ticker and the date window are
    parameters, so it runs against any symbol present in the data folder --
    including one converted from yfinance by convert_data.py.
    """

    def initialize(self):
        self.ticker = self.get_parameter("ticker", "NVDA")
        self._set_date("start_date", "2015-01-02", self.set_start_date)
        self._set_date("end_date", "2026-08-07", self.set_end_date)
        self.set_cash(10000)

        self.ma_period = int(self.get_parameter("ma_period", 200))
        self.stock = self.add_equity(self.ticker, Resolution.DAILY).symbol
        self.set_benchmark(self.stock)
        self.moving_average = self.sma(self.stock, self.ma_period, Resolution.DAILY)
        self.set_warm_up(self.ma_period, Resolution.DAILY)
        self._in_market = False

    def _set_date(self, name, default, apply):
        year, month, day = (int(part) for part in self.get_parameter(name, default).split("-"))
        apply(year, month, day)

    def on_data(self, data: Slice):
        if self.is_warming_up or not self.moving_average.is_ready:
            return
        if not data.bars.contains_key(self.stock):
            return
        close = float(data.bars[self.stock].close)
        should_be_in = close >= float(self.moving_average.current.value)
        if should_be_in == self._in_market:
            return
        self._in_market = should_be_in
        self.set_holdings(self.stock, 1.0 if should_be_in else 0.0)
