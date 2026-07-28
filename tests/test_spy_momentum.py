from lean_stubs import run_algorithm
from spy_momentum import SpyMomentum

# lookback_months 1 -> required_days 21, so bars start 2020-01-01 and the
# 21st bar (2020-01-21) is the first one that can trade.
ONE_MONTH = {"lookback_months": 1}


def test_required_days_uses_21_trading_days_per_month():
    algo = SpyMomentum()
    algo._parameters = {"lookback_months": "12"}
    algo.initialize()
    assert algo.required_days == 252


def test_no_trade_before_the_window_is_full():
    algo = run_algorithm(SpyMomentum, [100.0] * 20, ONE_MONTH)
    assert algo.orders == []


def test_enters_when_price_is_above_the_lookback_price():
    algo = run_algorithm(SpyMomentum, [100.0] * 20 + [150.0], ONE_MONTH)
    assert [weight for _, _, weight in algo.orders] == [1.0]


def test_stays_flat_when_momentum_is_negative_on_the_first_check():
    algo = run_algorithm(SpyMomentum, [100.0] * 20 + [50.0], ONE_MONTH)
    assert algo.orders == []


def test_signal_is_only_re_evaluated_once_per_calendar_month():
    # 21 bars in January fill the window; the 21st is the first check.
    # Remaining January bars must not re-check even as prices swing.
    algo = run_algorithm(
        SpyMomentum, [100.0] * 20 + [150.0, 10.0, 10.0], ONE_MONTH
    )
    assert [weight for _, _, weight in algo.orders] == [1.0]


def test_a_new_month_re_evaluates_and_can_exit():
    # 31 January bars then February: the February bar re-checks and exits.
    algo = run_algorithm(
        SpyMomentum, [100.0] * 20 + [150.0] * 11 + [1.0], ONE_MONTH
    )
    assert [weight for _, _, weight in algo.orders] == [1.0, 0.0]


def test_window_fills_during_warm_up():
    algo = run_algorithm(
        SpyMomentum, [100.0] * 20 + [150.0], ONE_MONTH, warmup_bars=20
    )
    assert [weight for _, _, weight in algo.orders] == [1.0]


def test_lookback_compares_against_the_oldest_bar_not_a_neighbour():
    # Ramp prices so every window slot holds a distinct value: bars 0..19
    # (2020-01-01 .. 2020-01-20) close at 100.0..119.0, then bar 20
    # (2020-01-21, the 21st bar) closes at 100.5, filling the 21-slot
    # window and triggering the first check.
    # RollingWindow index 0 is newest, so after the 21st add the window
    # holds [100.5, 119.0, 118.0, ..., 101.0, 100.0]: index 20
    # (closes.count - 1) is the oldest bar's close, 100.0, and index 19 is
    # the second-oldest, 101.0.
    # past_close = closes[closes.count - 1] must read the oldest (100.0):
    # 100.5 > 100.0 -> enters (order 1.0). An off-by-one that read
    # closes[19] instead (101.0) would see 100.5 > 101.0 == False and stay
    # flat, placing no order. Verified against the fake harness directly.
    algo = run_algorithm(
        SpyMomentum, [100.0 + i for i in range(20)] + [100.5], ONE_MONTH
    )
    assert [weight for _, _, weight in algo.orders] == [1.0]


def test_defaults_to_twelve_months():
    algo = SpyMomentum()
    algo._parameters = {}
    algo.initialize()
    assert algo.lookback_months == 12
    assert algo.warm_up_period == 252
