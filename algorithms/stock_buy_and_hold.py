from AlgorithmImports import *


class StockBuyAndHold(QCAlgorithm):
    """Buy a single stock on the first bar and never trade again.

    The baseline stock_ma_entry_exit has to beat: without it, a timing rule's
    return says nothing about whether the timing helped.
    """

    def initialize(self):
        self.ticker = self.get_parameter("ticker", "NVDA")
        self._set_date("start_date", "2015-01-02", self.set_start_date)
        self._set_date("end_date", "2026-08-07", self.set_end_date)
        self.set_cash(10000)

        self.weight = float(self.get_parameter("weight", 1.0))
        self.stock = self.add_equity(self.ticker, Resolution.DAILY).symbol
        self.set_benchmark(self.stock)

    def _set_date(self, name, default, apply):
        year, month, day = (int(part) for part in self.get_parameter(name, default).split("-"))
        apply(year, month, day)

    def on_data(self, data: Slice):
        if self.portfolio.invested:
            return
        if not data.bars.contains_key(self.stock):
            return
        self.set_holdings(self.stock, self.weight)
