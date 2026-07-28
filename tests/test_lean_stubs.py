from datetime import datetime

from lean_stubs import QCAlgorithm, Resolution, RollingWindow, feed


class _Probe(QCAlgorithm):
    def initialize(self):
        self.set_cash(10000)
        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.sma3 = self.sma(self.spy, 3, Resolution.DAILY)
        self.set_warm_up(3, Resolution.DAILY)
        self.window = RollingWindow[float](2)
        self.seen = []

    def on_data(self, data):
        self.window.add(float(data.bars[self.spy].close))
        if self.is_warming_up:
            return
        self.seen.append((self.sma3.is_ready, self.sma3.current.value))
        if not self.portfolio.invested:
            self.set_holdings(self.spy, 1.0)


def _bars(prices):
    return [(datetime(2020, 1, d + 1), p) for d, p in enumerate(prices)]


def test_warm_up_suppresses_on_data_body_but_not_indicators():
    algo = _Probe()
    algo.initialize()
    feed(algo, _bars([10.0, 20.0, 30.0, 40.0]), warmup_bars=3)

    # Three warm-up bars produced no recorded observations.
    assert len(algo.seen) == 1
    is_ready, value = algo.seen[0]
    assert is_ready is True
    assert value == 30.0  # mean of 20, 30, 40


def test_rolling_window_fills_during_warm_up_and_is_newest_first():
    algo = _Probe()
    algo.initialize()
    feed(algo, _bars([10.0, 20.0, 30.0, 40.0]), warmup_bars=3)

    assert algo.window.is_ready is True
    assert algo.window[0] == 40.0
    assert algo.window[algo.window.count - 1] == 30.0


def test_set_holdings_is_recorded_and_moves_the_portfolio():
    algo = _Probe()
    algo.initialize()
    feed(algo, _bars([10.0, 20.0, 30.0, 40.0]), warmup_bars=3)

    assert len(algo.orders) == 1
    when, symbol, weight = algo.orders[0]
    assert symbol == "SPY"
    assert weight == 1.0
    assert algo.portfolio[algo.spy].holdings_value == 10000.0
    assert algo.portfolio.cash == 0.0


def test_parameters_default_when_unset_and_stringify_when_set():
    algo = _Probe()
    assert algo.get_parameter("missing", 0.5) == 0.5
    algo._parameters = {"missing": "0.25"}
    assert float(algo.get_parameter("missing", 0.5)) == 0.25


def test_run_algorithm_constructs_initialises_and_feeds():
    from lean_stubs import run_algorithm

    algo = run_algorithm(_Probe, [10.0, 20.0, 30.0, 40.0], warmup_bars=3)

    assert len(algo.seen) == 1
    assert [weight for _, _, weight in algo.orders] == [1.0]


def test_run_algorithm_stringifies_parameters_like_lean_does():
    from lean_stubs import run_algorithm

    algo = run_algorithm(_Probe, [10.0], parameters={"depth": 3})

    assert algo.get_parameter("depth", 0) == "3"
