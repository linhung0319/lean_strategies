from AlgorithmImports import *
# https://quantpedia.com/strategies/pairs-trading-with-stocks


class PairsTrading(QCAlgorithm):
    """Gatev, Goetzmann & Rouwenhorst (2006) distance-method pairs trading.

    Every `trading_months` months (default 6) a new cycle starts:

    1. Formation. Take every current S&P 500 member with a complete
       `formation_days` (default 252, ~12 months) history of adjusted
       (total-return) closes. Scale each series to start at 1.
    2. Matching. For every stock find the partner whose normalized price path
       has the smallest sum of squared deviations (SSD). Keep the
       `pair_count` (default 20) pairs with the smallest SSD.
    3. Trading. Over the next cycle, prices are re-scaled to 1 at the start.
       When a pair's spread (A_norm - B_norm) moves more than `entry_std`
       (default 2) formation-period standard deviations away from zero, buy
       the cheap leg and short the expensive one, `leg_weight` (default 5%)
       of equity each. Close when the spread crosses zero again. A pair may
       open and close several times. Everything is closed when the cycle ends.

    Local port: the universe is the SPY constituents universe built by

        uv run --group data convert_data.py universe --etf SPY \\
            --start 2020-01-02 --end 2026-09-15 --out data-sp500

    rather than all NYSE/AMEX/NASDAQ listings. Same-company share classes
    (GOOG/GOOGL, ...) are excluded: they are one stock, not a pair. See
    research/pairs_trading/strategy-explained.md.
    """

    SAME_COMPANY = [
        {"GOOG", "GOOGL"}, {"FOX", "FOXA"}, {"NWS", "NWSA"},
        {"DISCA", "DISCK"}, {"UA", "UAA"}, {"BF.A", "BF.B"}, {"LEN", "LEN.B"},
        {"BRK.A", "BRK.B"}, {"VIA", "VIAB"}, {"CMCSA", "CMCSK"},
    ]

    def initialize(self):
        self._set_date("end_date", "2026-09-15", self.set_end_date)
        self._set_date("start_date", "2021-09-16", self.set_start_date)
        self.set_cash(100_000)
        self.settings.seed_initial_prices = True
        self.formation_days = int(self.get_parameter("formation_days", 252))
        self.trading_months = int(self.get_parameter("trading_months", 6))
        self.pair_count = int(self.get_parameter("pair_count", 20))
        self.entry_std = float(self.get_parameter("entry_std", 2.0))
        self.leg_weight = float(self.get_parameter("leg_weight", 0.05))

        self.universe_settings.resolution = Resolution.DAILY
        spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self._spy = spy
        self.universe_settings.schedule.on(self.date_rules.month_start(spy))
        self.add_universe(self.universe.etf(spy, self.universe_settings, self._select))

        self._months_left = 0          # months until the next formation
        self._pending = None           # pairs formed, not yet installed
        self._pairs = []               # active cycle's pairs
        self._cycle = 0

    def _set_date(self, name, default, apply):
        year, month, day = (int(p) for p in self.get_parameter(name, default).split("-"))
        apply(year, month, day)

    # ------------------------------------------------------------------
    # formation
    # ------------------------------------------------------------------

    def _select(self, constituents):
        if self._months_left > 0:
            self._months_left -= 1
            return Universe.UNCHANGED
        # A constituent with no weight has no price data locally.
        symbols = [c.symbol for c in constituents if c.weight]
        history = self.history(symbols, self.formation_days, Resolution.DAILY)
        if history.empty:
            return Universe.UNCHANGED
        closes = history.close.unstack(0)
        # GGR drop stocks with any day without a trade: require a full window.
        closes = closes.dropna(axis=1)
        if closes.shape[1] < 2:
            return Universe.UNCHANGED

        pairs = self._form_pairs(closes)
        self._months_left = self.trading_months - 1
        self._pending = pairs
        selected = set()
        for p in pairs:
            selected.add(p["a"])
            selected.add(p["b"])
        return list(selected)

    def _form_pairs(self, closes):
        tickers = [s.value for s in closes.columns]
        norm = closes.values / closes.values[0]                  # T x N, all start at 1
        sq = (norm ** 2).sum(axis=0)
        ssd = sq[:, None] + sq[None, :] - 2.0 * norm.T @ norm     # N x N
        n = len(tickers)
        np.fill_diagonal(ssd, np.inf)
        index = {t: i for i, t in enumerate(tickers)}
        for group in self.SAME_COMPANY:
            members = [index[t] for t in group if t in index]
            for i in members:
                for j in members:
                    ssd[i, j] = np.inf
        # Each stock's best partner, de-duplicated, then the smallest SSDs.
        best = ssd.argmin(axis=1)
        candidates = {}
        for i in range(n):
            j = int(best[i])
            if not np.isfinite(ssd[i, j]):
                continue
            key = (min(i, j), max(i, j))
            candidates[key] = ssd[i, j]
        chosen = sorted(candidates.items(), key=lambda kv: kv[1])[:self.pair_count]

        self._cycle += 1
        pairs = []
        for (i, j), distance in chosen:
            spread = norm[:, i] - norm[:, j]
            pair = {
                "a": closes.columns[i], "b": closes.columns[j],
                "ssd": float(distance), "sigma": float(spread.std()),
                "base_a": float(closes.values[-1, i]), "base_b": float(closes.values[-1, j]),
                "side": 0, "qty_a": 0, "qty_b": 0,
            }
            pairs.append(pair)
            self.log(f"PAIR cycle={self._cycle} {tickers[i]}/{tickers[j]} "
                     f"ssd={distance:.4f} sigma={pair['sigma']:.4f} "
                     f"base={pair['base_a']:.4f}/{pair['base_b']:.4f} "
                     f"from={closes.index[0]:%Y-%m-%d} to={closes.index[-1]:%Y-%m-%d}")
        return pairs

    # ------------------------------------------------------------------
    # trading
    # ------------------------------------------------------------------

    def on_data(self, data):
        if self._pending is not None:
            # End of the old cycle: everything is closed. The new cycle's
            # pairs are watched from the next bar on.
            for pair in self._pairs:
                if pair["side"] != 0:
                    self._close(pair, "cycle end")
            self._pairs = self._pending
            self._pending = None
            self._rebalance()
            return

        for delisting in data.delistings.values():
            if delisting.type == DelistingType.WARNING:
                for pair in self._pairs:
                    if pair["side"] != 0 and delisting.symbol in (pair["a"], pair["b"]):
                        self._close(pair, "delisting")

        for pair in self._pairs:
            pa = self.securities[pair["a"]].price
            pb = self.securities[pair["b"]].price
            if pa <= 0 or pb <= 0:
                continue
            spread = pa / pair["base_a"] - pb / pair["base_b"]
            if pair["side"] == 0:
                if abs(spread) > self.entry_std * pair["sigma"]:
                    self._open(pair, spread, pa, pb)
            elif spread * pair["side"] <= 0:
                # side = +1 means A was rich (spread > 0); crossing zero closes.
                self._close(pair, "converged", spread)
        self._rebalance()

    def _open(self, pair, spread, pa, pb):
        dollars = self.portfolio.total_portfolio_value * self.leg_weight
        side = 1 if spread > 0 else -1       # +1: A rich -> short A, long B
        qty_a = -side * int(dollars / pa)
        qty_b = side * int(dollars / pb)
        if qty_a == 0 or qty_b == 0:
            return
        pair.update(side=side, qty_a=qty_a, qty_b=qty_b)
        self.log(f"OPEN {pair['a'].value}/{pair['b'].value} spread={spread:.4f} "
                 f"limit={self.entry_std * pair['sigma']:.4f} qty={qty_a}/{qty_b} "
                 f"px={pa:.4f}/{pb:.4f}")

    def _close(self, pair, reason, spread=float("nan")):
        self.log(f"CLOSE {pair['a'].value}/{pair['b'].value} reason={reason} spread={spread:.4f} "
                 f"px={self.securities[pair['a']].price:.4f}/{self.securities[pair['b']].price:.4f}")
        pair.update(side=0, qty_a=0, qty_b=0)

    def _rebalance(self):
        """Trade every symbol to the net of its open pairs' legs.

        A stock can sit in two pairs at once (DTE/LNT and EVRG/LNT), so
        orders are netted per symbol, and exposure-reducing orders go first
        so the margin they free is available to the rest.
        """
        target = {}
        for pair in self._pairs:
            for leg in ("a", "b"):
                target[pair[leg]] = target.get(pair[leg], 0) + pair["qty_" + leg]
        for kvp in self.portfolio:
            holding = kvp.value
            if holding.invested and holding.symbol not in target:
                target[holding.symbol] = 0
        orders = []
        for symbol, qty in target.items():
            # Daily bars can produce two slices a day (16:00 and 00:00), and
            # a market order placed at 16:00 only fills at the next open, so
            # count what is already on its way.
            held = self.portfolio[symbol].quantity + sum(
                o.quantity for o in self.transactions.get_open_orders(symbol))
            if qty != held:
                reduces = abs(qty) < abs(held) or qty * held < 0
                orders.append((0 if reduces else 1, symbol, qty - held))
        for _, symbol, delta in sorted(orders, key=lambda o: o[0]):
            self.market_order(symbol, delta, tag="pairs rebalance")
