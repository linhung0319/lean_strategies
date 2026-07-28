from datetime import datetime

import pytest

from lean_stubs import feed
from spy_periodic_rebalance import SpyPeriodicRebalance


def _run(bars, parameters=None):
    algo = SpyPeriodicRebalance()
    algo._parameters = parameters or {}
    algo.initialize()
    feed(algo, bars)
    return algo


def _monthly_bars():
    # One bar on the 15th of each month across two years, price constant so
    # drift never matters — only the calendar drives trades.
    return [
        (datetime(year, month, 15), 100.0)
        for year in (2020, 2021)
        for month in range(1, 13)
    ]


def test_monthly_rebalances_once_per_month():
    algo = _run(_monthly_bars(), {"frequency": "M"})
    assert len(algo.orders) == 24


def test_quarterly_rebalances_once_per_quarter():
    algo = _run(_monthly_bars(), {"frequency": "Q"})
    assert len(algo.orders) == 8
    assert [when.month for when, _, _ in algo.orders][:4] == [1, 4, 7, 10]


def test_annual_rebalances_once_per_year():
    algo = _run(_monthly_bars(), {"frequency": "Y"})
    assert len(algo.orders) == 2
    assert [when.year for when, _, _ in algo.orders] == [2020, 2021]


def test_extra_bars_within_a_period_do_not_retrigger():
    bars = [(datetime(2020, 1, day), 100.0) for day in (2, 3, 6, 7)]
    algo = _run(bars, {"frequency": "M"})
    assert len(algo.orders) == 1


def test_first_bar_establishes_the_allocation():
    algo = _run([(datetime(2020, 1, 2), 100.0)], {"target": "0.7"})
    assert [weight for _, _, weight in algo.orders] == [0.7]


def test_invalid_frequency_is_rejected():
    algo = SpyPeriodicRebalance()
    algo._parameters = {"frequency": "W"}
    with pytest.raises(ValueError, match="frequency must be M, Q or Y"):
        algo.initialize()
