import math

from lean_stubs import run_algorithm
from spy_vol_adjusted import SpyVolAdjusted

SHORT_LOOKBACK = {"lookback": 2}


def test_no_trade_before_the_return_window_is_full():
    # lookback 2 needs 3 prices; only 2 supplied.
    algo = run_algorithm(SpyVolAdjusted, [100.0, 110.0], SHORT_LOOKBACK)
    assert algo.orders == []


def test_window_fills_during_warm_up_so_the_first_live_bar_trades():
    # 3 warm-up bars fill the 2-return window; bar 4 trades immediately.
    algo = run_algorithm(
        SpyVolAdjusted, [100.0, 110.0, 121.0, 133.1], SHORT_LOOKBACK, warmup_bars=3
    )
    assert len(algo.orders) == 1


def test_weight_is_target_vol_over_realised_vol():
    # Returns +10% then -10%; build an explicit series and recompute.
    algo = run_algorithm(
        SpyVolAdjusted,
        [100.0, 110.0, 99.0],
        {"lookback": 2, "min_weight": 0.0, "max_weight": 10.0},
    )

    returns = [110.0 / 100.0 - 1.0, 99.0 / 110.0 - 1.0]
    mean = sum(returns) / 2
    variance = sum((r - mean) ** 2 for r in returns) / 1
    expected_vol = math.sqrt(variance) * math.sqrt(252)
    expected_weight = 0.15 / expected_vol

    assert algo.orders[-1][2] == expected_weight


def test_weight_is_clamped_to_the_maximum_in_calm_markets():
    # A linear (not compounding) drift is used deliberately: with an exact
    # ~1% compounding series the per-bar returns are bit-for-bit identical
    # IEEE754 doubles, so every 2-return window has variance exactly 0.0 and
    # silently takes the zero-volatility skip branch instead of clamping.
    # The linear series below gives each window a non-degenerate, non-zero
    # variance (~4.9e-9 then ~4.71e-9), and in both cases the resulting
    # volatility is small enough that raw_weight = 0.15 / vol clamps to the
    # 0.90 ceiling. Verified with a standalone Python computation.
    algo = run_algorithm(
        SpyVolAdjusted, [100.0, 101.0, 102.0, 103.0], SHORT_LOOKBACK
    )
    assert [weight for _, _, weight in algo.orders] == [0.90, 0.90]


def test_weight_is_clamped_to_the_minimum_in_violent_markets():
    algo = run_algorithm(SpyVolAdjusted, [100.0, 200.0, 50.0], SHORT_LOOKBACK)
    assert algo.orders[-1][2] == 0.10


def test_zero_volatility_places_no_order():
    # Flat prices give zero returns, zero variance, zero volatility.
    algo = run_algorithm(SpyVolAdjusted, [100.0, 100.0, 100.0], SHORT_LOOKBACK)
    assert algo.orders == []


def test_it_rebalances_on_every_bar_once_ready():
    algo = run_algorithm(
        SpyVolAdjusted, [100.0, 110.0, 99.0, 105.0, 100.0], SHORT_LOOKBACK
    )
    assert len(algo.orders) == 3


def test_defaults_match_the_original_strategy():
    algo = SpyVolAdjusted()
    algo._parameters = {}
    algo.initialize()
    assert (algo.lookback, algo.target_vol, algo.min_weight, algo.max_weight) == (
        20,
        0.15,
        0.10,
        0.90,
    )
    assert algo.warm_up_period == 21
