"""The ticker-parameterised algorithms that run against converted data.

Same style as the SPY tests: decision logic only, driven through the fake
AlgorithmImports. Fills, fees and warm-up are LEAN's job.
"""

from lean_stubs import run_algorithm
from stock_buy_and_hold import StockBuyAndHold
from stock_ma_entry_exit import StockMaEntryExit

NVDA = {"ticker": "NVDA", "ma_period": 3}


def test_it_subscribes_to_the_ticker_parameter():
    algo = run_algorithm(StockMaEntryExit, [10.0], NVDA)
    assert str(algo.stock) == "NVDA"
    assert algo.ticker == "NVDA"


def test_the_ticker_defaults_to_nvda_so_the_file_pastes_into_quantconnect():
    algo = run_algorithm(StockMaEntryExit, [10.0])
    assert str(algo.stock) == "NVDA"
    assert algo.ma_period == 200


def test_the_date_window_comes_from_parameters():
    algo = run_algorithm(
        StockMaEntryExit,
        [10.0],
        {**NVDA, "start_date": "2015-01-02", "end_date": "2026-08-07"},
    )
    assert (algo.start_date.year, algo.start_date.month, algo.start_date.day) == (2015, 1, 2)
    assert (algo.end_date.year, algo.end_date.month, algo.end_date.day) == (2026, 8, 7)


def test_it_enters_when_the_close_reaches_the_moving_average():
    # ma_period=3, prices [30, 20, 10, 40]. The inclusive SMA over the last
    # three (20, 10, 40) is 23.33; close 40 clears it, so one entry.
    algo = run_algorithm(StockMaEntryExit, [30.0, 20.0, 10.0, 40.0], NVDA, warmup_bars=3)
    assert [weight for _, _, weight in algo.orders] == [1.0]


def test_it_exits_when_the_close_falls_below_and_does_not_re_order():
    # Rise into the trend, then break down: one entry, one exit, no churn
    # while the signal is unchanged.
    prices = [10.0, 20.0, 30.0, 40.0, 50.0, 5.0, 4.0]
    algo = run_algorithm(StockMaEntryExit, prices, NVDA, warmup_bars=3)
    assert [weight for _, _, weight in algo.orders] == [1.0, 0.0]


def test_it_stays_flat_while_the_close_is_below_the_average():
    algo = run_algorithm(StockMaEntryExit, [50.0, 40.0, 30.0, 20.0, 10.0], NVDA, warmup_bars=3)
    assert algo.orders == []


def test_buy_and_hold_buys_once_and_never_again():
    algo = run_algorithm(StockBuyAndHold, [10.0, 20.0, 5.0, 30.0], {"ticker": "NVDA"})
    assert [weight for _, _, weight in algo.orders] == [1.0]
    assert str(algo.orders[0][1]) == "NVDA"


def test_buy_and_hold_honours_a_partial_weight():
    algo = run_algorithm(StockBuyAndHold, [10.0, 20.0], {"ticker": "NVDA", "weight": 0.5})
    assert [weight for _, _, weight in algo.orders] == [0.5]
