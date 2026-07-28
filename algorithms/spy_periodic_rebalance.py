from AlgorithmImports import *


class SpyPeriodicRebalance(QCAlgorithm):
    """Rebalance to a fixed target on the first trading day of each period.

    Port of PeriodicRebalance in sp500/strategies/rebalance_periodic.py.
    Between rebalance dates the portfolio floats freely.
    """

    def initialize(self):
        self.set_start_date(2000, 1, 3)
        self.set_end_date(2021, 3, 31)   # bundled local data ends here; delete this line on QC
        self.set_cash(10000)

        self.target = float(self.get_parameter("target", 0.50))
        self.frequency = str(self.get_parameter("frequency", "M")).upper()
        if self.frequency not in ("M", "Q", "Y"):
            raise ValueError(
                f"frequency must be M, Q or Y, got {self.frequency!r}"
            )

        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.set_benchmark(self.spy)

        self._last_period = None

    def _period_key(self, when):
        if self.frequency == "M":
            return (when.year, when.month)
        if self.frequency == "Q":
            return (when.year, (when.month - 1) // 3)
        return (when.year,)

    def on_data(self, data: Slice):
        if not data.bars.contains_key(self.spy):
            return

        period = self._period_key(self.time)
        if period == self._last_period:
            return

        self._last_period = period
        self.set_holdings(self.spy, self.target)
