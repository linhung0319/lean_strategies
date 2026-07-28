from AlgorithmImports import *


class SpyMaEntryExit(QCAlgorithm):
    """All-or-nothing market timing on a simple moving average.

    100% SPY while the close is at or above the average, 100% cash below it.
    Port of MovingAverageEntryExit in sp500/strategies/moving_average.py.
    """

    def initialize(self):
        self.set_start_date(2000, 1, 3)
        self.set_end_date(2021, 3, 31)   # bundled local data ends here; delete this line on QC
        self.set_cash(10000)

        self.ma_period = int(self.get_parameter("ma_period", 200))

        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.set_benchmark(self.spy)

        self.moving_average = self.sma(self.spy, self.ma_period, Resolution.DAILY)
        self.set_warm_up(self.ma_period, Resolution.DAILY)

        self._in_market = False

    def on_data(self, data: Slice):
        if self.is_warming_up or not self.moving_average.is_ready:
            return
        if not data.bars.contains_key(self.spy):
            return

        close = float(data.bars[self.spy].close)
        should_be_in = close >= float(self.moving_average.current.value)
        if should_be_in == self._in_market:
            return

        self._in_market = should_be_in
        self.set_holdings(self.spy, 1.0 if should_be_in else 0.0)
