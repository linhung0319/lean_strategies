from AlgorithmImports import *


class SpyThresholdRebalance(QCAlgorithm):
    """Hold SPY at a target weight, rebalancing only when drift leaves a band.

    Port of Rebalance5050 in sp500/strategies/rebalance_50_50.py. Between
    triggers the portfolio drifts with the market.
    """

    def initialize(self):
        self.set_start_date(2000, 1, 3)
        self.set_end_date(2021, 3, 31)   # bundled local data ends here; delete this line on QC
        self.set_cash(10000)

        self.target = float(self.get_parameter("target", 0.50))
        self.threshold = float(self.get_parameter("threshold", 0.05))

        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.set_benchmark(self.spy)

        self._initialized = False

    def on_data(self, data: Slice):
        if not data.bars.contains_key(self.spy):
            return

        if not self._initialized:
            self.set_holdings(self.spy, self.target)
            self._initialized = True
            return

        total = self.portfolio.total_portfolio_value
        if total <= 0:
            return

        weight = float(self.portfolio[self.spy].holdings_value) / float(total)
        if abs(weight - self.target) >= self.threshold:
            self.set_holdings(self.spy, self.target)
