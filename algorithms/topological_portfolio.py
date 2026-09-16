from AlgorithmImports import *
from Selection.ETFConstituentsUniverseSelectionModel import (
    ETFConstituentsUniverseSelectionModel
)
import kmapper as km
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from umap import UMAP


class TopologicalGraphUniverseSelectionModel(ETFConstituentsUniverseSelectionModel):
    """Groups the ETF's constituents by the shape of their return correlations.

    The Mapper algorithm turns a point cloud into a simplicial complex: project
    the data to one dimension, cover the projection in overlapping intervals,
    cluster within each interval, and join two nodes whenever they share a
    member. Here each point is a stock and its coordinates are its daily log
    returns, so a node is a set of stocks that move together and a connected
    component is a set of such groups linked by stocks that belong to several.
    """

    def __init__(
            self,
            etf_symbol: Symbol,
            lookback_window: int = 250,
            recalibration_period: int = None,
            universe_filter_func: Callable[
                [list[ETFConstituentUniverse]],
                list[Symbol]
            ] = None
        ) -> None:
        self._symbol = etf_symbol
        self.lookback_window = lookback_window
        self.recalibration_period = recalibration_period
        self.clustered_symbols = None
        super().__init__(etf_symbol, None, universe_filter_func)

    def create_universes(self, algorithm: QCAlgorithm) -> list[Universe]:
        universe_list = super().create_universes(algorithm)
        # Initial warm up of the universe.
        security = algorithm.securities[self._symbol]
        next_open = security.exchange.hours.get_next_market_open(
            algorithm.time,
            False
        )
        algorithm.schedule.on(
            algorithm.date_rules.on([next_open]),
            algorithm.time_rules.at(9, 31),
            lambda: self.get_graph_symbols(algorithm)
        )
        return universe_list

    def get_graph_symbols(self, algorithm: QCAlgorithm) -> None:
        # Construct simplicial complex.
        graph, symbol_list = self.construct_simplicial_complex(
            algorithm, self.lookback_window
        )
        if list(symbol_list):
            self.clustered_symbols = self.clustering_symbols(graph, symbol_list)
        # Set schedule event to reconstruct the topological structure.
        next_recalibration = (
            algorithm.time + timedelta(self.recalibration_period)
        )
        algorithm.schedule.on(
            algorithm.date_rules.on([next_recalibration]),
            algorithm.time_rules.at(0, 1),
            lambda: self.get_graph_symbols(algorithm)
        )

    def construct_simplicial_complex(self, algorithm: QCAlgorithm, lookback_window: int):
        if not self.universe.selected:
            return {}, []
        # Obtain historical data to construct a graph of stock
        # relationship.
        history = algorithm.history(
            self.universe.selected,
            lookback_window,
            Resolution.DAILY
        )
        if history.empty:
            return {}, []
        prices = history.unstack(0).close
        # Calculate daily log return. Then, transpose the data since
        # we're relating stocks.
        log_returns = np.log(prices / prices.shift(1)).dropna().T
        if log_returns.empty:
            return {}, []

        # Initialize the mapper algorithm.
        mapper = km.KeplerMapper()
        # Project the data into a 2d subspace via 2 transformation,
        # PCA and UMAP.
        # PCA: since it can retain the most variance while denoising,
        # as well as fast.
        # UMAP: handles non-linear relationships well and preserves
        # both local and global structures.
        # MDS and Isomap are not included due to their potential
        # sensitivity to noise and outliers in financial data.
        projection = [
            PCA(n_components=0.8, random_state=1),
            UMAP(n_components=1, random_state=1, n_jobs=-1)
        ]
        projected_data = mapper.fit_transform(
            log_returns,
            projection=projection
        )
        # Cluster the data with DBSCAN since it is better in handling
        # noise.
        # We are interested in the correlation distance to cluster and
        # form a portfolio.
        clusterer = DBSCAN(metric='correlation', n_jobs=-1)
        graph = mapper.map(projected_data, log_returns, clusterer)
        return graph, prices.columns

    def clustering_symbols(self, graph: dict[str, object], symbol_list: list[Symbol]) -> list[list[object]]:
        # Each connected structure as a giant cluster.
        linked_clusters = []
        for x, y in graph['links'].items():
            isin = False
            for i in range(len(linked_clusters)):
                if x in linked_clusters[i] or y in linked_clusters[i]:
                    linked_clusters[i] = list(set(linked_clusters[i] + [x] + y))
                    isin = True
            if isin:
                continue
            linked_clusters.append([x] + y)
        linked_clusters += [[x] for x in graph['nodes'] if x not in [z for y in linked_clusters for z in y]]
        # Convert the node into symbol.
        return [[list([symbol_list[graph['nodes'][x]]][0]) for x in linked_cluster] for linked_cluster in linked_clusters]


class TopologicalPortfolio(QCAlgorithm):
    """Equal-weight the stocks inside each node of a Mapper graph of SPY.

    Every `recalibrate_period` calendar days the universe model rebuilds the
    simplicial complex from `history_lookback` days of returns; in between, the
    portfolio is rebalanced daily onto the resulting clusters. Stocks that
    DBSCAN treats as noise are outliers, get no node, and are not held.

    Running this locally needs a data folder carrying the SPY constituents
    universe, which the free LEAN distribution does not ship:

        uv run --group data convert_data.py universe --etf SPY \\
            --start 2020-01-02 --end 2026-09-15 --out data-sp500
        uv run --group topological run_local.py "Topological Portfolio" \\
            --data-folder data-sp500

    kmapper, umap-learn and scikit-learn are in the `topological` dependency
    group; the run loads them through LEAN's embedded interpreter.
    """

    def initialize(self) -> None:
        np.random.seed(0)
        self.set_cash(1_000_000)
        # The original derives the window from today: end_date - 5 years. Both
        # ends are parameters here so a local run is reproducible and stays
        # inside the converted data -- on QuantConnect, drop them and keep
        # self.set_start_date(self.end_date - timedelta(5*365)).
        self._set_date("end_date", "2026-09-15", self.set_end_date)
        self._set_date("start_date", "2021-09-16", self.set_start_date)
        self.settings.seed_initial_prices = True
        # Daily, because that is what convert_data.py writes. The cloud
        # dataset this was written against is minute, and leaving the default
        # in place locally would subscribe every constituent to minute files
        # that are not there -- so nothing would ever have a price. Set before
        # add_universe_selection: it is what the universe adds securities with.
        self.universe_settings.resolution = Resolution.DAILY

        # We would like to compare with SPY for the correlation
        # and risk-adjusted return.
        spy = self.add_equity("SPY", Resolution.DAILY)
        # Locally this leaves Alpha, Beta and Treynor Ratio at zero, and that
        # is LEAN rather than the strategy: SetBenchmark always builds its own
        # security and the internal feed for it is hard-coded to hourly in
        # backtesting (Lean/Engine/DataFeeds/UniverseSelection.cs:337), which
        # a daily-only data folder cannot answer. The "SPY Buy & Hold
        # 2021-2026" variant is the like-for-like comparison instead.
        self.set_benchmark(spy)

        # Lookback window to construct and analyze the topological
        # structure.
        self.history_lookback = int(self.get_parameter("history_lookback", 150))
        # Set the period to reconstruct the topological complex.
        self.recalibrate_period = int(self.get_parameter("recalibrate_period", 125))
        # Number of constituents to keep, heaviest first.
        self.universe_size = int(self.get_parameter("universe_size", 200))
        # Construct a portfolio with SPY constituents.
        self.universe_model = TopologicalGraphUniverseSelectionModel(
            spy.symbol,
            self.history_lookback,
            self.recalibrate_period,
            lambda u: [
                x.symbol for x in sorted(
                    [x for x in u if x.weight],
                    key=lambda x: x.weight,
                    reverse=True
                )[:self.universe_size]
            ]
        )
        self.add_universe_selection(self.universe_model)

        # Set a scheduled event to rebalance the portfolio daily.
        self.schedule.on(
            self.date_rules.every_day(spy.symbol),
            self.time_rules.at(9, 31),
            self._rebalance
        )

        # Set the warm up to warm up the universe selection.
        self.set_warm_up(timedelta(365))

    def _set_date(self, name, default, apply):
        year, month, day = (int(part) for part in self.get_parameter(name, default).split("-"))
        apply(year, month, day)

    def _rebalance(self) -> None:
        if not self.universe_model.clustered_symbols:
            return
        # Obtain the weights invested in each constituents.
        weights = self._weight_distribution(
            self.universe_model.clustered_symbols
        )
        # Rebalance by designated weights.
        self.set_holdings(
            [
                PortfolioTarget(symbol, weight)
                for symbol, weight in weights.items()
            ],
            liquidate_existing_holdings=True
        )

    def _weight_distribution(self, clustered_symbols):
        # Assign weight between and within giant and small clusters.
        # Note that we do not invest in outliers.
        weights = {}
        def assign_weights(nested_list, level=1):
            num_elements = len(nested_list)
            if num_elements == 0:
                return
            weight_per_element = 1 / num_elements
            for item in nested_list:
                if isinstance(item, list):
                    assign_weights(item, level + 1)
                else:
                    weights[item] = (
                        weights.get(item, 0) + weight_per_element / (2 ** (level - 1))
                    )
        # Calculate the overall weights.
        assign_weights(clustered_symbols)
        total_weight = sum(weights.values())
        return {k: v / total_weight for k, v in weights.items()}

    def on_warmup_finished(self):
        self._rebalance()
