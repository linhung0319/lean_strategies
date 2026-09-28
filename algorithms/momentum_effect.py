from AlgorithmImports import *
# https://quantpedia.com/Screener/Details/14


class MomentumEffect(QCAlgorithm):
    """Hold the five large caps with the strongest 12-month momentum.

    At each month start the universe is rebuilt: stocks above $5, the 100 with
    the highest dollar volume, then the 50 largest of those by market cap. The
    five of those 50 with the highest 252-day momentum are held at 20% each
    until the next month start. Anything that leaves the universe is sold.

    Local port of QuantConnect's MomentumEffectAlgorithm. The original selects
    from Morningstar fundamentals, which the free LEAN distribution does not
    ship -- locally CoarseFundamentalDataProvider answers HasFundamentalData
    with false and MarketCap with 0, so the original selects nothing. This one
    selects from the SPY constituents universe instead:

        uv run --group data convert_data.py universe --etf SPY \\
            --start 2020-01-02 --end 2026-09-15 --out data-sp500

    Price and dollar volume come from the previous session's raw daily bar
    (what Fundamental.price and .dollar_volume hold); market cap becomes the
    constituent weight (price x shares outstanding, normalised within the day
    -- the ranking is the same). The candidates are therefore S&P 500 members
    rather than every US listing. The trading logic is unchanged. See
    research/momentum_effect/strategy-explained.md.
    """

    def initialize(self):
        # The original is set_start_date(self.end_date - timedelta(5*365)).
        # Both ends are pinned here so a local run stays inside the data.
        self._set_date("end_date", "2026-09-15", self.set_end_date)
        self._set_date("start_date", "2021-09-16", self.set_start_date)
        self.set_cash(100_000)
        self.settings.seed_initial_prices = True
        self.settings.automatic_indicator_warm_up = True
        # Define some parameters.
        self.lookback = int(self.get_parameter("lookback", 252))
        self.liquidity_filter_size = int(self.get_parameter("liquidity_filter_size", 100))
        self.universe_size = int(self.get_parameter("universe_size", 50))
        self.portfolio_size = int(self.get_parameter("portfolio_size", 5))
        # Add a universe of S&P 500 constituents.
        self.universe_settings.resolution = Resolution.DAILY
        spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self._date_rule = self.date_rules.month_start(spy)
        self.universe_settings.schedule.on(self._date_rule)
        self._universe = self.add_universe(
            self.universe.etf(spy, self.universe_settings, self._select_assets)
        )
        # Add a warm-up period so the algorithm trades right away
        # instead of waiting for the next month to start.
        self.set_warm_up(timedelta(45))

    def _set_date(self, name, default, apply):
        year, month, day = (int(part) for part in self.get_parameter(name, default).split("-"))
        apply(year, month, day)

    def _select_assets(self, constituents):
        # A constituent with no weight has no price data (acquired names the
        # vendor no longer serves) -- the local stand-in for has_fundamental_data.
        weights = {c.symbol: c.weight for c in constituents if c.weight}
        # The previous session's raw bar gives Fundamental.price/.dollar_volume.
        history = self.history(list(weights), 1, Resolution.DAILY,
                               data_normalization_mode=DataNormalizationMode.RAW)
        if history.empty:
            return Universe.UNCHANGED
        last = history.groupby(level="symbol").last()
        # Select the most liquid stocks trading above $5/share.
        filtered = []
        for symbol in weights:
            if symbol not in last.index:
                continue
            bar = last.loc[symbol]
            dollar_volume = bar.close * bar.volume
            # A NaN would silently scramble sorted(); `> 0` is False for NaN.
            if bar.close > 5 and dollar_volume > 0:
                filtered.append((symbol, dollar_volume))
        filtered = sorted(filtered, key=lambda f: f[1])[-self.liquidity_filter_size:]
        # Select the subset of stocks with highest market caps.
        filtered = sorted(filtered, key=lambda f: weights[f[0]], reverse=True)[:self.universe_size]
        return [s for s, _ in filtered]

    def on_securities_changed(self, changes):
        # As assets enter the universe, add a momentum indicator to each one.
        for security in changes.added_securities:
            if security.symbol.value == "SPY":
                continue
            security.momentum = self.momp(security.symbol, self.lookback)
        # As assets leave the universe, remove their indicator and liquidate it.
        for security in changes.removed_securities:
            if hasattr(security, "momentum"):
                self.deregister_indicator(security.momentum)
            self.liquidate(security.symbol, 'Removed from universe')

    def on_warmup_finished(self):
        # Add a Scheduled Event to rebalance the portofolio monthly.
        time_rule = self.time_rules.at(8, 0)
        self.schedule.on(self._date_rule, time_rule, self._rebalance)
        # Rebalance today too.
        if self.live_mode:
            self._rebalance()
        else:
            self.schedule.on(self.date_rules.today, time_rule, self._rebalance)

    def _rebalance(self):
        # Get the stocks that have their momentum indicators primed.
        securities = []
        for symbol in self._universe.selected:
            security = self.securities[symbol]
            if security.momentum.is_ready:
                securities.append(security)
        # Select the securities with greatest momentum.
        selected = sorted(securities, key=lambda security: security.momentum.current.value)[-self.portfolio_size:]
        if not selected:
            return
        # Form an equal-weighted portfolio.
        targets = [PortfolioTarget(security.symbol, 1/len(selected)) for security in selected]
        self.set_holdings(targets, True)
        self.log("REBALANCE " + ",".join(
            f"{s.symbol.value}:{s.momentum.current.value:.2f}" for s in reversed(selected)))
