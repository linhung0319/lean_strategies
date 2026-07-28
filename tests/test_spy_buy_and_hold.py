from datetime import datetime

from lean_stubs import run_algorithm
from spy_buy_and_hold import SpyBuyAndHold


def test_buys_once_on_the_first_bar_and_never_again():
    algo = run_algorithm(SpyBuyAndHold, [100.0, 120.0, 80.0, 140.0])
    assert [weight for _, _, weight in algo.orders] == [1.0]
    assert algo.orders[0][0] == datetime(2020, 1, 1)


def test_allocation_drifts_freely_after_the_initial_buy():
    algo = run_algorithm(SpyBuyAndHold, [100.0, 200.0])
    # 100% SPY doubled: no rebalance, so the whole portfolio rode the move.
    assert algo.portfolio.total_portfolio_value == 20000.0


def test_partial_weight_leaves_the_remainder_in_cash():
    algo = run_algorithm(SpyBuyAndHold, [100.0, 100.0], {"spy_weight": 0.5})
    assert [weight for _, _, weight in algo.orders] == [0.5]
    assert algo.portfolio.cash == 5000.0


def test_defaults_to_fully_invested():
    algo = SpyBuyAndHold()
    algo._parameters = {}
    algo.initialize()
    assert algo.spy_weight == 1.0
