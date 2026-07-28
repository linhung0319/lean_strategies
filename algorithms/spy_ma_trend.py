from AlgorithmImports import *


class SpyMaTrend(QCAlgorithm):
    """Dynamic SPY allocation driven by a simple moving average.

    Overweight SPY above the average, underweight below it, always partially
    invested. Port of MovingAverageTrend in
    sp500/strategies/moving_average.py. Rebalances only on a signal flip, so
    the allocation drifts between flips exactly as the original does.
    """

    def initialize(self):
        self.set_start_date(2000, 1, 3)
        self.set_end_date(2021, 3, 31)   # bundled local data ends here; delete this line on QC
        self.set_cash(10000)

        self.ma_period = int(self.get_parameter("ma_period", 200))
        self.above_weight = float(self.get_parameter("above_weight", 0.60))
        self.below_weight = float(self.get_parameter("below_weight", 0.40))

        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.set_benchmark(self.spy)

        self.moving_average = self.sma(self.spy, self.ma_period, Resolution.DAILY)
        self.set_warm_up(self.ma_period, Resolution.DAILY)

        self._current_target = None

    def on_data(self, data: Slice):
        if self.is_warming_up or not self.moving_average.is_ready:
            return
        if not data.bars.contains_key(self.spy):
            return

        close = float(data.bars[self.spy].close)
        above = close >= float(self.moving_average.current.value)
        target = self.above_weight if above else self.below_weight

        if target == self._current_target:
            return

        self._current_target = target
        self.set_holdings(self.spy, target)
