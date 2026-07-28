from lean_stubs import run_algorithm
from spy_threshold_rebalance import SpyThresholdRebalance


def test_first_bar_establishes_the_target_allocation():
    algo = run_algorithm(SpyThresholdRebalance, [100.0])
    assert [weight for _, _, weight in algo.orders] == [0.5]
    assert algo.portfolio.cash == 5000.0


def test_small_drift_does_not_trigger_a_rebalance():
    # 5000 in SPY at 100 -> 50 shares. At 105 the SPY weight is
    # 5250 / 10250 = 51.2%, drift 1.2% < 5% band.
    algo = run_algorithm(SpyThresholdRebalance, [100.0, 105.0])
    assert len(algo.orders) == 1


def test_drift_beyond_the_band_rebalances_back_to_target():
    # At 150 the SPY weight is 7500 / 12500 = 60%, drift 10% >= 5% band.
    algo = run_algorithm(SpyThresholdRebalance, [100.0, 150.0])
    assert [weight for _, _, weight in algo.orders] == [0.5, 0.5]
    assert algo.portfolio[algo.spy].holdings_value == 6250.0


def test_drift_exactly_on_the_band_rebalances():
    # 5000 at 100 -> 50 shares. Solve 50p / (50p + 5000) = 0.55 -> p = 122.222...
    algo = run_algorithm(
        SpyThresholdRebalance, [100.0, 5000.0 * 0.55 / (50.0 * 0.45)]
    )
    assert len(algo.orders) == 2


def test_downside_drift_also_rebalances():
    # At 50 the SPY weight is 2500 / 7500 = 33.3%, drift 16.7% >= 5%.
    algo = run_algorithm(SpyThresholdRebalance, [100.0, 50.0])
    assert len(algo.orders) == 2


def test_target_and_threshold_are_configurable():
    algo = run_algorithm(
        SpyThresholdRebalance, [100.0], {"target": 0.8, "threshold": 0.2}
    )
    assert [weight for _, _, weight in algo.orders] == [0.8]
    assert algo.threshold == 0.2
