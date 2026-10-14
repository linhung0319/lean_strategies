# region imports
from AlgorithmImports import *
import statsmodels.api as sm
from sklearn.decomposition import PCA
# endregion


class PcaStatArbitrageAlgorithm(QCAlgorithm):
    """Hold the 20 most liquid stocks whose PCA residual is unusually low.

    At each month start the universe is the 20 stocks above $5 with the
    highest dollar volume. Before the open, 60 daily closes of those 20 are
    log transformed and de-meaned, the first three principal components are
    taken as factors, and each stock is regressed on them. A stock whose
    latest residual z-score is below -1.5 is bought, weighted by |z| so the
    positions sum to 100% of the (95% usable) portfolio. (The description
    calls these shorts; the code's double negative makes them longs -- see
    the `side` parameter.) Everything else is
    closed. Positions are held until the next month start.

    Local port of QuantConnect's PCA statistical arbitrage example. The
    original selects from Morningstar fundamentals, which the free LEAN
    distribution does not ship, so this one selects from the SPY constituents
    universe instead, the same way momentum_effect.py does:

        uv run --group data convert_data.py universe --etf SPY \\
            --start 2020-01-02 --end 2026-09-15 --out data-sp500

    Price and dollar volume come from the previous session's raw daily bar
    (what Fundamental.price and .dollar_volume hold). The candidates are
    therefore S&P 500 members rather than every US listing. The trading logic
    is unchanged. See research/pca_stat_arbitrage/strategy-explained.md.
    """

    def initialize(self):
        # The original is set_start_date(self.end_date - timedelta(5*365)).
        # Both ends are pinned here so a local run stays inside the data.
        self._set_date("end_date", "2026-09-15", self.set_end_date)
        self._set_date("start_date", "2021-09-16", self.set_start_date)
        self.set_cash(100000)
        self.settings.seed_initial_prices = True
        self.settings.free_portfolio_value_percentage = 0.05
        self.lookback_days = int(self.get_parameter("lookback_days", 60))
        self.pca_components = int(self.get_parameter("pca_components", 3))
        self.entry_z_score = float(self.get_parameter("entry_z_score", -1.5))
        self.universe_size = int(self.get_parameter("universe_size", 20))
        # selected_scores are all negative, so -weight is positive: as written
        # the original buys the cheap names. side = -1 flips that into the
        # shorts the strategy description talks about.
        self.side = int(self.get_parameter("side", 1))
        self.universe_settings.resolution = Resolution.DAILY
        # The original also sets universe_settings.asynchronous = True. That
        # is a speed setting only, and the local selector calls self.history,
        # so it is left off here.
        spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self._date_rule = self.date_rules.month_start(spy)
        self.universe_settings.schedule.on(self._date_rule)
        # Add universe of top 20 stocks by dollar volume.
        self._universe = self.add_universe(
            self.universe.etf(spy, self.universe_settings, self._select_assets)
        )
        self.set_warm_up(365)

    def _set_date(self, name, default, apply):
        year, month, day = (int(part) for part in self.get_parameter(name, default).split("-"))
        apply(year, month, day)

    def _select_assets(self, constituents):
        # A constituent with no weight has no price data (acquired names the
        # vendor no longer serves) -- the local stand-in for has_fundamental_data.
        symbols = [c.symbol for c in constituents if c.weight]
        # The previous session's raw bar gives Fundamental.price/.dollar_volume.
        history = self.history(symbols, 1, Resolution.DAILY,
                               data_normalization_mode=DataNormalizationMode.RAW)
        if history.empty:
            return Universe.UNCHANGED
        last = history.groupby(level="symbol").last()
        filtered = []
        for symbol in symbols:
            if symbol not in last.index:
                continue
            bar = last.loc[symbol]
            dollar_volume = bar.close * bar.volume
            # A NaN would silently scramble sorted(); `> 0` is False for NaN.
            if bar.close > 5 and dollar_volume > 0:
                filtered.append((symbol, dollar_volume))
        return [s for s, _ in sorted(filtered, key=lambda f: f[1])[-self.universe_size:]]

    def on_warmup_finished(self):
        # Add a Scheduled event to rebalance the portfolio monthly.
        time_rule = self.time_rules.at(8, 0)
        self.schedule.on(self._date_rule, time_rule, self._rebalance)
        # Rebalance the portfolio today too.
        if self.live_mode:
            self._rebalance()
        else:
            self.schedule.on(self.date_rules.today, time_rule, self._rebalance)

    def _rebalance(self):
        # Build targets from the current universe selection.
        weight_by_symbol = self._compute_weights_by_symbol(self._universe.selected)
        targets = [PortfolioTarget(symbol, -weight * self.side) for symbol, weight in weight_by_symbol.items()]
        self.set_holdings(targets, True)
        self.log("REBALANCE " + ",".join(
            f"{symbol.value}:{weight:.4f}" for symbol, weight in weight_by_symbol.items()))

    def _compute_weights_by_symbol(self, symbols):
        if not symbols:
            return {}
        history = self.history(symbols, self.lookback_days, Resolution.DAILY).close.unstack(0)
        # Sample data for PCA (smooth using np.log function).
        sample = np.log(history.dropna(axis=1))
        # Center the sample column-wise.
        sample = sample - sample.mean()
        # Fit the PCA model.
        pca_model = PCA().fit(sample)
        # Get the first n factors from the PCA.
        factors = np.dot(sample, pca_model.components_.T)[:, :self.pca_components]
        # Add constants to fit the linear regression (intercept).
        factors = sm.add_constant(factors)
        # Train ordinary least squares linear model for each symbol.
        ols_models = {symbol: sm.OLS(sample[symbol], factors).fit() for symbol in sample.columns}
        # Get residuals from the linear regression after PCA.
        residuals = pd.DataFrame({symbol: model.resid for symbol, model in ols_models.items()}, index=sample.index)
        # Get z-scores of residuals from the most recent day.
        z_scores = ((residuals - residuals.mean()) / residuals.std()).iloc[-1]
        # Select stocks far from mean for mean reversion.
        selected_scores = z_scores[z_scores < self.entry_z_score]
        denominator = selected_scores.abs().sum()
        # Return weights for each selected stock.
        return selected_scores * (1 / denominator)
