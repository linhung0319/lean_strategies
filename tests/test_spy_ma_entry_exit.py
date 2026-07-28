from lean_stubs import run_algorithm
from spy_ma_entry_exit import SpyMaEntryExit

SHORT_MA = {"ma_period": 3}


def test_enters_when_price_closes_above_the_moving_average():
    # 3-period SMA. Bars 1-3 are warm-up. Bar 4 close 40 > SMA(20,30,40)=30.
    algo = run_algorithm(
        SpyMaEntryExit, [10.0, 20.0, 30.0, 40.0], SHORT_MA, warmup_bars=3
    )
    assert [weight for _, _, weight in algo.orders] == [1.0]


def test_stays_flat_when_the_first_signal_is_bearish():
    # Bar 4 close 5 < SMA(30,20,5) = 18.33, and the algorithm starts flat.
    algo = run_algorithm(
        SpyMaEntryExit, [40.0, 30.0, 20.0, 5.0], SHORT_MA, warmup_bars=3
    )
    assert algo.orders == []


def test_exits_when_price_falls_back_below_the_moving_average():
    algo = run_algorithm(
        SpyMaEntryExit, [10.0, 20.0, 30.0, 40.0, 1.0], SHORT_MA, warmup_bars=3
    )
    assert [weight for _, _, weight in algo.orders] == [1.0, 0.0]


def test_does_not_retrade_while_the_signal_is_unchanged():
    algo = run_algorithm(
        SpyMaEntryExit,
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        SHORT_MA,
        warmup_bars=3,
    )
    assert [weight for _, _, weight in algo.orders] == [1.0]


def test_price_equal_to_the_average_counts_as_in_market():
    # Constant prices make close == SMA exactly; the comparison is >=.
    algo = run_algorithm(
        SpyMaEntryExit, [10.0, 10.0, 10.0, 10.0], SHORT_MA, warmup_bars=3
    )
    assert [weight for _, _, weight in algo.orders] == [1.0]


def test_defaults_to_a_200_day_average():
    algo = SpyMaEntryExit()
    algo._parameters = {}
    algo.initialize()
    assert algo.ma_period == 200
