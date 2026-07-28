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


def test_defaults_to_twelve_months():
    algo = SpyMomentum()
    algo._parameters = {}
    algo.initialize()
    assert algo.lookback_months == 12
