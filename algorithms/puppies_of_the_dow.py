from AlgorithmImports import *


class PuppiesOfTheDow(QCAlgorithm):
    """Hold the five cheapest of the ten highest-yielding Dow stocks for a year.

    At each year start the DIA constituents (the Dow 30) above $5 with a
    positive dividend yield are ranked by yield. The ten highest are the Dogs
    of the Dow; the five lowest-priced of those are the Puppies, held at 20%
    each until the next year start.

    Local port of QuantConnect's PuppiesOfTheDow. The original selects from
    Morningstar fundamentals (Fundamental.price and
    valuation_ratios.total_yield), which the free LEAN distribution does not
    ship. This one selects from a DIA constituents universe built from
    leandata/reference/dow-30-membership.csv:

        uv run --group data convert_data.py universe --etf DIA \\
            --membership-file leandata/reference/dow-30-membership.csv \\
            --start 2020-01-02 --end 2026-09-15 --out data-dow \\
            --cache data-sp500/.cache

    Price is the previous session's raw close (what Fundamental.price holds).
    Yield is trailing-twelve-month dividends per share over that price,
    computed from LEAN's own dividend and split history. Morningstar's
    total_yield also adds buyback yield; the strategy description ranks by
    dividend yield, which is what this does. The trading logic is unchanged.
    See research/puppies_of_the_dow/strategy-explained.md.
    """

    def initialize(self):
        # The original is set_start_date(self.end_date - timedelta(5*365)).
        # Both ends are pinned here so a local run stays inside the data.
        self._set_date("end_date", "2026-09-15", self.set_end_date)
        self._set_date("start_date", "2021-09-16", self.set_start_date)
        self.set_cash(100_000)
        self.portfolio_size = int(self.get_parameter("portfolio_size", 5))
        self.dogs_size = int(self.get_parameter("dogs_size", 10))
        self.settings.seed_initial_prices = True
        self.universe_settings.resolution = Resolution.DAILY
        # DIA stands in for "SPY" in the date rules: the data folder is built
        # around it, so its bars are the trading calendar.
        dia = self.add_equity("DIA", Resolution.DAILY).symbol
        self._date_rule = self.date_rules.year_start(dia)
        self.universe_settings.schedule.on(self._date_rule)
        # Add universe using DIA ETF constituents (Dow 30)
        self._universe = self.add_universe(
            self.universe.etf(dia, self.universe_settings, self._select)
        )
        self.schedule.on(self._date_rule, self.time_rules.at(8, 0), self._rebalance)
        self.set_warm_up(timedelta(365))

    def _set_date(self, name, default, apply):
        year, month, day = (int(part) for part in self.get_parameter(name, default).split("-"))
        apply(year, month, day)

    def _select(self, constituents):
        # A constituent with no weight has no price data (WBA: taken private
        # in 2025, no longer served by the vendor).
        symbols = [c.symbol for c in constituents if c.weight]
        last = self._last_raw_close(symbols)
        if last is None:
            return Universe.UNCHANGED
        dividends = self._trailing_dividends(symbols)
        # Filter for valid price and dividend yield
        dow = []
        for symbol in symbols:
            price = last.get(symbol, 0)
            dividend = dividends.get(symbol, 0)
            # `> 5` is False for NaN, so a missing bar cannot slip through.
            if price > 5 and dividend > 0:
                dow.append((symbol, price, dividend / price))
        # Dogs of the Dow: top 10 highest dividend yielders from the Dow 30
        dogs = sorted(dow, key=lambda x: x[2])[-self.dogs_size:]
        # Puppies: the 5 lowest-priced stocks from the Dogs
        puppies = sorted(dogs, key=lambda x: x[1])[:self.portfolio_size]
        self.log("SELECT " + ",".join(
            f"{s.value}:{p:.2f}:{y:.4f}:{'P' if (s, p, y) in puppies else 'D'}"
            for s, p, y in sorted(dow, key=lambda x: -x[2])))
        return [p[0] for p in puppies]

    def _last_raw_close(self, symbols):
        history = self.history(symbols, 1, Resolution.DAILY,
                               data_normalization_mode=DataNormalizationMode.RAW)
        if history.empty:
            return None
        closes = history["close"].groupby(level="symbol").last()
        return {symbol: closes[symbol] for symbol in symbols if symbol in closes.index}

    def _trailing_dividends(self, symbols):
        """Dividends per share paid over the past year, in today's shares."""
        dividends = self.history(Dividend, symbols, timedelta(365))
        splits = self.history(Split, symbols, timedelta(365))
        occurred = []
        if not splits.empty:
            splits = splits[splits["type"] == SplitType.SPLIT_OCCURRED]
            occurred = list(splits.itertuples())
        totals = {}
        if dividends.empty:
            return totals
        for row in dividends.itertuples():
            symbol, time = row.Index
            # Stamped on the ex-date, and the window includes today. A
            # dividend going ex today is not paid to today's buyer, and
            # counting it can fit five quarterly payments into one year.
            if time >= self.time:
                continue
            amount = row.distribution
            # A dividend paid before a split was paid on pre-split shares.
            for split in occurred:
                if split.Index[0] == symbol and split.Index[1] > time:
                    amount *= split.splitfactor
            totals[symbol] = totals.get(symbol, 0) + amount
        return totals

    def _rebalance(self):
        if self.is_warming_up:
            return
        # Create equal-weight portfolio targets from selected universe symbols
        targets = [PortfolioTarget(s, 1 / self.portfolio_size) for s in self._universe.selected]
        self.set_holdings(targets, True)
        self.log("REBALANCE " + ",".join(s.value for s in self._universe.selected))

    def on_warmup_finished(self):
        self._rebalance()
