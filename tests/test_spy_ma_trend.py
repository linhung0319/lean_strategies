from lean_stubs import run_algorithm
from spy_ma_trend import SpyMaTrend

SHORT_MA = {"ma_period": 3}


def test_first_post_warmup_bar_establishes_the_above_weight():
    algo = run_algorithm(SpyMaTrend, [10.0, 20.0, 30.0, 40.0], SHORT_MA, warmup_bars=3)
    assert [weight for _, _, weight in algo.orders] == [0.6]


def test_first_post_warmup_bar_establishes_the_below_weight():
    algo = run_algorithm(SpyMaTrend, [40.0, 30.0, 20.0, 5.0], SHORT_MA, warmup_bars=3)
    assert [weight for _, _, weight in algo.orders] == [0.4]


def test_flipping_below_the_average_switches_to_the_below_weight():
    algo = run_algorithm(
        SpyMaTrend, [10.0, 20.0, 30.0, 40.0, 1.0], SHORT_MA, warmup_bars=3
    )
    assert [weight for _, _, weight in algo.orders] == [0.6, 0.4]


def test_no_trade_while_the_regime_is_unchanged():
    algo = run_algorithm(
        SpyMaTrend, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0], SHORT_MA, warmup_bars=3
    )
    assert [weight for _, _, weight in algo.orders] == [0.6]


def test_weights_are_configurable():
    algo = run_algorithm(
        SpyMaTrend,
        [10.0, 20.0, 30.0, 40.0],
        {"ma_period": 3, "above_weight": 0.9, "below_weight": 0.1},
        warmup_bars=3,
    )
    assert [weight for _, _, weight in algo.orders] == [0.9]


def test_defaults_match_the_original_strategy():
    algo = SpyMaTrend()
    algo._parameters = {}
    algo.initialize()
    assert (algo.ma_period, algo.above_weight, algo.below_weight) == (200, 0.60, 0.40)
