import math

from AlgorithmImports import *


class SpyVolAdjusted(QCAlgorithm):
    """Inverse-volatility position sizing, rebalanced every day.

    SPY weight = target_vol / realised_vol, clamped to [min_weight,
    max_weight]. Port of VolatilityAdjusted in
    sp500/strategies/volatility_adjusted.py.

    Note: the original rebalances daily at zero cost. Under a real fee model
    this strategy pays for that churn; that cost is genuine, not a defect.
    """

    TRADING_DAYS_PER_YEAR = 252

    def initialize(self):
        self.set_start_date(2000, 1, 3)
        self.set_end_date(2021, 3, 31)   # bundled local data ends here; delete this line on QC
        self.set_cash(10000)

        self.lookback = int(self.get_parameter("lookback", 20))
        self.target_vol = float(self.get_parameter("target_vol", 0.15))
        self.min_weight = float(self.get_parameter("min_weight", 0.10))
        self.max_weight = float(self.get_parameter("max_weight", 0.90))

        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.set_benchmark(self.spy)

        self.returns = RollingWindow[float](self.lookback)
        self._previous_close = None

        # lookback returns need lookback + 1 prices.
        self.set_warm_up(self.lookback + 1, Resolution.DAILY)

    def on_data(self, data: Slice):
        if not data.bars.contains_key(self.spy):
            return

        close = float(data.bars[self.spy].close)

        # Update the window before the warm-up guard: on_data runs during
        # warm-up, so guarding first would leave the window permanently empty.
        if self._previous_close:
            self.returns.add(close / self._previous_close - 1.0)
        self._previous_close = close

        if self.is_warming_up or not self.returns.is_ready:
            return

        volatility = self._annualised_volatility()
        if volatility == 0:
            return

        raw_weight = self.target_vol / volatility
        weight = max(self.min_weight, min(self.max_weight, raw_weight))
        self.set_holdings(self.spy, weight)

    def _annualised_volatility(self):
        values = [self.returns[i] for i in range(self.returns.count)]
        count = len(values)
        if count < 2:
            return 0.0
        mean = sum(values) / count
        # ddof=1 sample variance, matching pandas Series.std().
        variance = sum((value - mean) ** 2 for value in values) / (count - 1)
        return math.sqrt(variance) * math.sqrt(self.TRADING_DAYS_PER_YEAR)
