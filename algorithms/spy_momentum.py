from AlgorithmImports import *


class SpyMomentum(QCAlgorithm):
    """Time-series momentum, re-evaluated on the first trading day of a month.

    100% SPY when the price is above where it stood lookback_months ago,
    otherwise 100% cash. Port of Momentum in sp500/strategies/momentum.py,
    keeping the original's 21-trading-days-per-month convention.
    """

    TRADING_DAYS_PER_MONTH = 21

    def initialize(self):
        self.set_start_date(2000, 1, 3)
        self.set_end_date(2021, 3, 31)   # bundled local data ends here; delete this line on QC
        self.set_cash(10000)

        self.lookback_months = int(self.get_parameter("lookback_months", 12))
        self.required_days = self.lookback_months * self.TRADING_DAYS_PER_MONTH

        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.set_benchmark(self.spy)

        self.closes = RollingWindow[float](self.required_days)
        self.set_warm_up(self.required_days, Resolution.DAILY)

        self._in_market = False
        self._last_check_month = None

    def on_data(self, data: Slice):
        if not data.bars.contains_key(self.spy):
            return

        close = float(data.bars[self.spy].close)

        # Update before the warm-up guard so warm-up history fills the window.
        self.closes.add(close)

        if self.is_warming_up or not self.closes.is_ready:
            return

        month = (self.time.year, self.time.month)
        if month == self._last_check_month:
            return
        self._last_check_month = month

        past_close = self.closes[self.closes.count - 1]
        should_be_in = close > past_close
        if should_be_in == self._in_market:
            return

        self._in_market = should_be_in
        self.set_holdings(self.spy, 1.0 if should_be_in else 0.0)
