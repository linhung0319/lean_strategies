from AlgorithmImports import *


# Stocks Yahoo Finance files under the "Technology" sector. Yahoo's sector
# field is Morningstar's classification, which is what the original selects on
# (MorningstarSectorCode.TECHNOLOGY): GOOGL and META are Communication
# Services and AMZN is Consumer Cyclical there, so none of them qualify.
# Built from research/tech_ema_reversion/sectors.csv, which covers every ticker
# that was ever in SPY's top 120 by weight between 2020 and 2026-09 -- far
# deeper than a top-5 Technology cut ever reaches. It is today's
# classification applied to the whole window, not a point-in-time one.
TECHNOLOGY = frozenset({
    "AAPL", "ACN", "ADBE", "ADI", "ADP", "ADSK", "AMAT", "AMD", "ANET", "APH",
    "AVGO", "CDNS", "CRM", "CRWD", "CSCO", "DDOG", "DELL", "FIS", "FISV",
    "FTNT", "GLW", "IBM", "INTC", "INTU", "KLAC", "LRCX", "MRVL", "MSFT", "MU",
    "NOW", "NVDA", "ORCL", "PANW", "PLTR", "QCOM", "SNDK", "SNPS", "STX",
    "TXN", "UBER", "WDC",
})


class TechEmaReversion(QCAlgorithm):
    """Hold the five largest Technology stocks only while they are cheap.

    Every day the universe is the five largest Technology names. A name is
    bought (at 1/5 of the portfolio) when it closes below its 100-day EMA and
    sold when it closes at or above its 252-day high. Names that drop out of
    the top five are sold.

    That high is stricter than it reads: MAX takes each bar's High, and its
    window includes the bar just closed, so the exit fires only on a day that
    closes at its own high *and* sets a 252-day high. See
    research/tech_ema_reversion/strategy-explained.md.

    Local port of the QuantConnect original. That one selects from
    Morningstar fundamentals, which the free LEAN distribution does not ship,
    so this one selects from the SPY constituents universe instead:

        uv run --group data convert_data.py universe --etf SPY \\
            --start 2020-01-02 --end 2026-09-15 --out data-sp500

    Market cap becomes the constituent weight (price x shares outstanding,
    normalised within the day -- the ranking is the same) and the sector
    comes from TECHNOLOGY above. The trading logic is unchanged.
    """

    def initialize(self):
        # The original is set_start_date(self.end_date - timedelta(5*365)).
        # Both ends are pinned here so a local run stays inside the data.
        self._set_date("end_date", "2026-09-15", self.set_end_date)
        self._set_date("start_date", "2021-09-16", self.set_start_date)
        self.set_cash(100_000)
        self.universe_settings.resolution = Resolution.DAILY
        self.settings.automatic_indicator_warm_up = True
        self.settings.seed_initial_prices = True

        self.universe_size = int(self.get_parameter("universe_size", 5))
        self.ema_period = int(self.get_parameter("ema_period", 100))
        self.max_period = int(self.get_parameter("max_period", 252))

        spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self._universe = self.add_universe(
            self.universe.etf(spy, self.universe_settings, self._select)
        )
        self.schedule.on(
            self.date_rules.every_day(spy),
            self.time_rules.at(8, 0),
            self._rebalance
        )

    def _set_date(self, name, default, apply):
        year, month, day = (int(part) for part in self.get_parameter(name, default).split("-"))
        apply(year, month, day)

    def _select(self, constituents):
        filtered = [
            c for c in constituents
            if c.weight and c.symbol.value in TECHNOLOGY
        ]
        return [c.symbol for c in sorted(filtered, key=lambda c: c.weight)[-self.universe_size:]]

    def on_securities_changed(self, changes):
        for security in changes.added_securities:
            security.ema = self.ema(security, self.ema_period)
            security.max = self.max(security, self.max_period)
        for security in changes.removed_securities:
            self.liquidate(security)

    def _rebalance(self):
        for symbol in self._universe.selected:
            security = self.securities[symbol]
            if not security.max.is_ready:
                continue
            # Buy signal: price below EMA and not invested
            if not security.invested and security.price < security.ema.current.value:
                self.set_holdings(security, 1 / len(self._universe.selected))
            # Sell signal: price at or above 1yr-high and invested
            elif security.invested and security.price >= security.max.current.value:
                self.liquidate(security)
