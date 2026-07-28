from AlgorithmImports import *


class SpyBuyAndHold(QCAlgorithm):
    """Buy SPY once at the configured weight and never trade again.

    Port of BuyAndHold in sp500/strategies/buy_and_hold.py. With spy_weight
    1.0 this is the plain SPY benchmark; with 0.5 it shows how an unmanaged
    50/50 portfolio drifts.
    """

    def initialize(self):
        self.set_start_date(2000, 1, 3)
        self.set_end_date(2021, 3, 31)   # bundled local data ends here; delete this line on QC
        self.set_cash(10000)

        self.spy_weight = float(self.get_parameter("spy_weight", 1.0))

        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.set_benchmark(self.spy)

        self._initialized = False

    def on_data(self, data: Slice):
        if self._initialized or not data.bars.contains_key(self.spy):
            return

        self.set_holdings(self.spy, self.spy_weight)
        self._initialized = True
