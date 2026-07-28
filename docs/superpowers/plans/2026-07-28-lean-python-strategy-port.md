# LEAN Python Strategy Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the seven strategy classes in `Stock/sp500/strategies/` to standalone LEAN Python `QCAlgorithm` files that paste unchanged into quantconnect.com and also run against the local LEAN engine.

**Architecture:** Each algorithm is a single self-contained file under `algorithms/` whose only import is `from AlgorithmImports import *` (plus stdlib `math` where needed). Configuration comes from `self.get_parameter(name, default)`, so a pasted file runs with no setup. A test-only fake of `AlgorithmImports` lets the allocation logic be driven as plain Python under pytest without .NET or pythonnet, which is what makes TDD possible before the local LEAN prerequisites are installed. `variants.py` and `run_local.py` are local batch-run tooling and never leave the repo.

**Tech Stack:** Python 3.11, pytest, uv. Runtime target is LEAN (C#/pythonnet) and QuantConnect Cloud.

## Global Constraints

- Python 3.11 (`requires-python = ">=3.11,<3.12"`). Provisioned by uv as a standalone x64 CPython 3.11.11 at `C:\Users\user\AppData\Roaming\uv\python\cpython-3.11.11-windows-x86_64-none\`, whose `python311.dll` is what pythonnet loads.
- Runtime dependencies pinned to LEAN's requirements: `pandas==2.2.3`, `wrapt==1.16.0`. Source: `Stock/Lean/Algorithm.Python/readme.md`.
- Dev dependencies: `pytest`, `quantconnect-stubs`.
- Files in `algorithms/` may import **only** `from AlgorithmImports import *` and Python stdlib modules. No numpy, no pandas, no cross-file imports — they must paste into QC as a single file.
- Every algorithm uses `set_start_date(2000, 1, 3)`, `set_end_date(2021, 3, 31)`, `set_cash(10000)`. Source: `sp500/main.py:43`, `sp500/main.py:52`.
- Every `get_parameter` call supplies a default so the file runs unconfigured.
- Prices read off `data.bars[...]` are wrapped in `float(...)` before arithmetic — pythonnet can surface them as `Decimal`, which does not mix with float literals.
- The repo is `Stock/lean_strategies/`, already git-initialised with `.gitignore` covering `.venv/`, `__pycache__/`, `results/`, `*.pyc`.
- Do not modify anything under `Stock/Lean/` or `Stock/sp500/`.

### Deviation from the spec, recorded deliberately

The spec's `spy_vol_adjusted` section says `numpy.std(returns, ddof=1) * numpy.sqrt(252)`. This plan uses the same formula written with stdlib `math` instead. Reason: it removes any question of whether `np` is in scope after `from AlgorithmImports import *`, keeping the file provably paste-safe. The arithmetic is identical (`ddof=1` sample standard deviation over 20 values).

---

### Task 1: uv project scaffold and the fake `AlgorithmImports` harness

**Files:**
- Create: `pyproject.toml`
- Create: `tests/lean_stubs.py`
- Create: `tests/conftest.py`
- Test: `tests/test_lean_stubs.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `tests/lean_stubs.py`: `install()` (registers a fake `AlgorithmImports` in `sys.modules`), `feed(algorithm, bars, warmup_bars=0)` where `bars` is a list of `(datetime, float)` and returns `None`, `run_algorithm(algorithm_class, prices, parameters=None, warmup_bars=0)` returning the driven algorithm, and the fake classes `QCAlgorithm`, `Resolution`, `RollingWindow`, `Slice`, `TradeBar`.
  - `run_algorithm` is the helper every later task's tests use; only Task 4 builds bars by hand, because its assertions depend on calendar dates.
  - Every algorithm instance exposes `algorithm.orders`, a list of `(datetime, symbol, weight)` tuples recording each `set_holdings` call. All later tasks assert against this list.
  - `algorithm._parameters` is a plain dict written before `initialize()` to simulate LEAN parameters.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "lean-strategies"
version = "0.1.0"
description = "LEAN Python ports of the sp500 SPY allocation strategies"
requires-python = ">=3.11,<3.12"
dependencies = [
    "pandas==2.2.3",
    "wrapt==1.16.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "quantconnect-stubs",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write the failing test for the harness**

Create `tests/test_lean_stubs.py`:

```python
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
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_lean_stubs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lean_stubs'`

- [ ] **Step 4: Write `tests/lean_stubs.py`**

```python
"""A minimal stand-in for LEAN's ``AlgorithmImports``.

Enough of the API surface to drive the files in ``algorithms/`` as plain
Python and assert on the allocation decisions they make. This tests decision
logic only — order fills, fees, slippage and real warm-up are the engine's
job and are verified by running LEAN itself (see README.md).
"""

import sys
import types
from datetime import datetime, timedelta


class Resolution:
    DAILY = "Daily"
    HOUR = "Hour"
    MINUTE = "Minute"


class Symbol(str):
    """LEAN symbols behave as opaque keys; a str subclass is close enough."""


class TradeBar:
    def __init__(self, close):
        self.close = close
        self.open = close
        self.high = close
        self.low = close
        self.volume = 0


class _Bars(dict):
    def contains_key(self, key):
        return key in self


class Slice:
    def __init__(self, bars):
        self.bars = _Bars(bars)


class RollingWindow:
    """Mirrors LEAN's RollingWindow: index 0 is newest, count-1 is oldest."""

    def __class_getitem__(cls, item):
        return cls

    def __init__(self, size):
        self.size = size
        self._items = []

    def add(self, item):
        self._items.insert(0, item)
        if len(self._items) > self.size:
            self._items.pop()

    def __getitem__(self, index):
        return self._items[index]

    @property
    def count(self):
        return len(self._items)

    @property
    def is_ready(self):
        return len(self._items) == self.size


class _SimpleMovingAverage:
    def __init__(self, period):
        self.period = period
        self._values = []
        self.current = types.SimpleNamespace(value=0.0)

    @property
    def is_ready(self):
        return len(self._values) >= self.period

    def update(self, value):
        self._values.append(value)
        if len(self._values) > self.period:
            self._values.pop(0)
        self.current.value = sum(self._values) / len(self._values)


class _Holding:
    def __init__(self):
        self.quantity = 0.0
        self.holdings_value = 0.0

    @property
    def invested(self):
        return self.quantity != 0.0


class _Portfolio:
    def __init__(self):
        self.cash = 0.0
        self._holdings = {}

    def __getitem__(self, symbol):
        return self._holdings.setdefault(symbol, _Holding())

    @property
    def total_portfolio_value(self):
        return self.cash + sum(h.holdings_value for h in self._holdings.values())

    @property
    def invested(self):
        return any(h.invested for h in self._holdings.values())


class QCAlgorithm:
    def __init__(self):
        self._parameters = {}
        self._indicators = []
        self._price = 0.0
        self.portfolio = _Portfolio()
        self.time = datetime(1970, 1, 1)
        self.is_warming_up = False
        self.orders = []

    # -- configuration -------------------------------------------------
    def set_start_date(self, year, month, day):
        self.start_date = datetime(year, month, day)

    def set_end_date(self, year, month, day):
        self.end_date = datetime(year, month, day)

    def set_cash(self, amount):
        self.portfolio.cash = float(amount)

    def get_parameter(self, name, default=None):
        return self._parameters.get(name, default)

    def add_equity(self, ticker, resolution=None):
        return types.SimpleNamespace(symbol=Symbol(ticker))

    def set_benchmark(self, symbol):
        pass

    def set_warm_up(self, period, resolution=None):
        self.warm_up_period = period

    def sma(self, symbol, period, resolution=None):
        indicator = _SimpleMovingAverage(period)
        self._indicators.append(indicator)
        return indicator

    def log(self, message):
        pass

    def debug(self, message):
        pass

    # -- trading -------------------------------------------------------
    def set_holdings(self, symbol, weight):
        self.orders.append((self.time, symbol, float(weight)))
        total = self.portfolio.total_portfolio_value
        target_value = total * float(weight)
        holding = self.portfolio[symbol]
        holding.quantity = target_value / self._price if self._price else 0.0
        holding.holdings_value = target_value
        self.portfolio.cash = total - target_value

    # -- driven by feed() ----------------------------------------------
    def _mark_to_market(self, symbol, price):
        holding = self.portfolio[symbol]
        holding.holdings_value = holding.quantity * price


def feed(algorithm, bars, warmup_bars=0):
    """Push ``(datetime, close)`` pairs through the algorithm as SPY bars.

    The first ``warmup_bars`` are delivered with ``is_warming_up`` set, which
    is how LEAN behaves: ``on_data`` is still called and framework indicators
    still update, but the algorithm is expected to skip trading.
    """
    symbol = Symbol("SPY")
    for index, (when, close) in enumerate(bars):
        close = float(close)
        algorithm.time = when
        algorithm._price = close
        algorithm._mark_to_market(symbol, close)
        for indicator in algorithm._indicators:
            indicator.update(close)
        algorithm.is_warming_up = index < warmup_bars
        algorithm.on_data(Slice({symbol: TradeBar(close)}))


def run_algorithm(algorithm_class, prices, parameters=None, warmup_bars=0):
    """Construct, configure, initialise and drive an algorithm in one call.

    ``prices`` is a list of closes delivered as consecutive daily bars starting
    2020-01-01. Parameters are stringified the way LEAN delivers them. Returns
    the algorithm so tests can assert on ``.orders`` and portfolio state.

    Tests needing specific calendar dates should build bars themselves and call
    ``feed`` directly.
    """
    algorithm = algorithm_class()
    algorithm._parameters = {
        key: str(value) for key, value in (parameters or {}).items()
    }
    algorithm.initialize()
    start = datetime(2020, 1, 1)
    bars = [
        (start + timedelta(days=index), float(price))
        for index, price in enumerate(prices)
    ]
    feed(algorithm, bars, warmup_bars=warmup_bars)
    return algorithm


def install():
    """Register the fakes as the ``AlgorithmImports`` module."""
    module = types.ModuleType("AlgorithmImports")
    exported = [
        "QCAlgorithm",
        "Resolution",
        "RollingWindow",
        "Slice",
        "Symbol",
        "TradeBar",
    ]
    for name in exported:
        setattr(module, name, globals()[name])
    module.__all__ = exported
    sys.modules["AlgorithmImports"] = module
```

- [ ] **Step 5: Write `tests/conftest.py`**

```python
"""Make the fakes importable and register them before any algorithm module
is imported, and put ``algorithms/`` on the path so tests can import the
files under test by module name.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "algorithms"))

import lean_stubs

lean_stubs.install()
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_lean_stubs.py -v`
Expected: PASS, 6 tests

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock tests/
git commit -m "test: add uv project and fake AlgorithmImports harness"
```

---

### Task 2: `spy_buy_and_hold.py`

**Files:**
- Create: `algorithms/spy_buy_and_hold.py`
- Test: `tests/test_spy_buy_and_hold.py`

**Interfaces:**
- Consumes: `lean_stubs.feed`, `algorithm.orders` from Task 1.
- Produces: class `SpyBuyAndHold`, parameter `spy_weight` (default `1.0`).

Source of truth: `Stock/sp500/strategies/buy_and_hold.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_spy_buy_and_hold.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_spy_buy_and_hold.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spy_buy_and_hold'`

- [ ] **Step 3: Write `algorithms/spy_buy_and_hold.py`**

```python
from AlgorithmImports import *


class SpyBuyAndHold(QCAlgorithm):
    """Buy SPY once at the configured weight and never trade again.

    Port of BuyAndHold in sp500/strategies/buy_and_hold.py. With spy_weight
    1.0 this is the plain SPY benchmark; with 0.5 it shows how an unmanaged
    50/50 portfolio drifts.
    """

    def initialize(self):
        self.set_start_date(2000, 1, 3)
        self.set_end_date(2021, 3, 31)   # bundled local data ends here; delete this line on QC
        self.set_cash(10000)

        self.spy_weight = float(self.get_parameter("spy_weight", 1.0))

        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.set_benchmark(self.spy)

        self._initialized = False

    def on_data(self, data: Slice):
        if self._initialized or not data.bars.contains_key(self.spy):
            return

        self.set_holdings(self.spy, self.spy_weight)
        self._initialized = True
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_spy_buy_and_hold.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add algorithms/spy_buy_and_hold.py tests/test_spy_buy_and_hold.py
git commit -m "feat: port buy-and-hold strategy to LEAN Python"
```

---

### Task 3: `spy_threshold_rebalance.py`

**Files:**
- Create: `algorithms/spy_threshold_rebalance.py`
- Test: `tests/test_spy_threshold_rebalance.py`

**Interfaces:**
- Consumes: `lean_stubs.feed`, `algorithm.orders` from Task 1.
- Produces: class `SpyThresholdRebalance`, parameters `target` (default `0.50`), `threshold` (default `0.05`).

Source of truth: `Stock/sp500/strategies/rebalance_50_50.py`. The drift comparison uses `>=`, matching line 38 of the original.

- [ ] **Step 1: Write the failing test**

Create `tests/test_spy_threshold_rebalance.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_spy_threshold_rebalance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spy_threshold_rebalance'`

- [ ] **Step 3: Write `algorithms/spy_threshold_rebalance.py`**

```python
from AlgorithmImports import *


class SpyThresholdRebalance(QCAlgorithm):
    """Hold SPY at a target weight, rebalancing only when drift leaves a band.

    Port of Rebalance5050 in sp500/strategies/rebalance_50_50.py. Between
    triggers the portfolio drifts with the market.
    """

    def initialize(self):
        self.set_start_date(2000, 1, 3)
        self.set_end_date(2021, 3, 31)   # bundled local data ends here; delete this line on QC
        self.set_cash(10000)

        self.target = float(self.get_parameter("target", 0.50))
        self.threshold = float(self.get_parameter("threshold", 0.05))

        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.set_benchmark(self.spy)

        self._initialized = False

    def on_data(self, data: Slice):
        if not data.bars.contains_key(self.spy):
            return

        if not self._initialized:
            self.set_holdings(self.spy, self.target)
            self._initialized = True
            return

        total = self.portfolio.total_portfolio_value
        if total <= 0:
            return

        weight = float(self.portfolio[self.spy].holdings_value) / float(total)
        if abs(weight - self.target) >= self.threshold:
            self.set_holdings(self.spy, self.target)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_spy_threshold_rebalance.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add algorithms/spy_threshold_rebalance.py tests/test_spy_threshold_rebalance.py
git commit -m "feat: port threshold rebalance strategy to LEAN Python"
```

---

### Task 4: `spy_periodic_rebalance.py`

**Files:**
- Create: `algorithms/spy_periodic_rebalance.py`
- Test: `tests/test_spy_periodic_rebalance.py`

**Interfaces:**
- Consumes: `lean_stubs.feed`, `algorithm.orders` from Task 1.
- Produces: class `SpyPeriodicRebalance`, parameters `target` (default `0.50`), `frequency` (default `"M"`, accepts `"M"`, `"Q"`, `"Y"`).

Source of truth: `Stock/sp500/strategies/rebalance_periodic.py`. The period keys mirror `_period_key()` exactly.

- [ ] **Step 1: Write the failing test**

Create `tests/test_spy_periodic_rebalance.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_spy_periodic_rebalance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spy_periodic_rebalance'`

- [ ] **Step 3: Write `algorithms/spy_periodic_rebalance.py`**

```python
from AlgorithmImports import *


class SpyPeriodicRebalance(QCAlgorithm):
    """Rebalance to a fixed target on the first trading day of each period.

    Port of PeriodicRebalance in sp500/strategies/rebalance_periodic.py.
    Between rebalance dates the portfolio floats freely.
    """

    def initialize(self):
        self.set_start_date(2000, 1, 3)
        self.set_end_date(2021, 3, 31)   # bundled local data ends here; delete this line on QC
        self.set_cash(10000)

        self.target = float(self.get_parameter("target", 0.50))
        self.frequency = str(self.get_parameter("frequency", "M")).upper()
        if self.frequency not in ("M", "Q", "Y"):
            raise ValueError(
                f"frequency must be M, Q or Y, got {self.frequency!r}"
            )

        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.set_benchmark(self.spy)

        self._last_period = None

    def _period_key(self, when):
        if self.frequency == "M":
            return (when.year, when.month)
        if self.frequency == "Q":
            return (when.year, (when.month - 1) // 3)
        return (when.year,)

    def on_data(self, data: Slice):
        if not data.bars.contains_key(self.spy):
            return

        period = self._period_key(self.time)
        if period == self._last_period:
            return

        self._last_period = period
        self.set_holdings(self.spy, self.target)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_spy_periodic_rebalance.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add algorithms/spy_periodic_rebalance.py tests/test_spy_periodic_rebalance.py
git commit -m "feat: port periodic rebalance strategy to LEAN Python"
```

---

### Task 5: `spy_ma_entry_exit.py`

**Files:**
- Create: `algorithms/spy_ma_entry_exit.py`
- Test: `tests/test_spy_ma_entry_exit.py`

**Interfaces:**
- Consumes: `lean_stubs.feed`, `algorithm.orders` from Task 1.
- Produces: class `SpyMaEntryExit`, parameter `ma_period` (default `200`).

Source of truth: `MovingAverageEntryExit` in `Stock/sp500/strategies/moving_average.py:59`. This is the anchor algorithm — its live LEAN result must reproduce the already-measured C# figures (see Task 9's README step).

`_in_market` starts `False`, matching the validated C# port: the algorithm begins flat and acts on the first real signal rather than assuming it is invested.

- [ ] **Step 1: Write the failing test**

Create `tests/test_spy_ma_entry_exit.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_spy_ma_entry_exit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spy_ma_entry_exit'`

- [ ] **Step 3: Write `algorithms/spy_ma_entry_exit.py`**

```python
from AlgorithmImports import *


class SpyMaEntryExit(QCAlgorithm):
    """All-or-nothing market timing on a simple moving average.

    100% SPY while the close is at or above the average, 100% cash below it.
    Port of MovingAverageEntryExit in sp500/strategies/moving_average.py.
    """

    def initialize(self):
        self.set_start_date(2000, 1, 3)
        self.set_end_date(2021, 3, 31)   # bundled local data ends here; delete this line on QC
        self.set_cash(10000)

        self.ma_period = int(self.get_parameter("ma_period", 200))

        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.set_benchmark(self.spy)

        self.moving_average = self.sma(self.spy, self.ma_period, Resolution.DAILY)
        self.set_warm_up(self.ma_period, Resolution.DAILY)

        self._in_market = False

    def on_data(self, data: Slice):
        if self.is_warming_up or not self.moving_average.is_ready:
            return
        if not data.bars.contains_key(self.spy):
            return

        close = float(data.bars[self.spy].close)
        should_be_in = close >= float(self.moving_average.current.value)
        if should_be_in == self._in_market:
            return

        self._in_market = should_be_in
        self.set_holdings(self.spy, 1.0 if should_be_in else 0.0)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_spy_ma_entry_exit.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add algorithms/spy_ma_entry_exit.py tests/test_spy_ma_entry_exit.py
git commit -m "feat: port 200-day MA entry/exit strategy to LEAN Python"
```

---

### Task 6: `spy_ma_trend.py`

**Files:**
- Create: `algorithms/spy_ma_trend.py`
- Test: `tests/test_spy_ma_trend.py`

**Interfaces:**
- Consumes: `lean_stubs.feed`, `algorithm.orders` from Task 1.
- Produces: class `SpyMaTrend`, parameters `ma_period` (default `200`), `above_weight` (default `0.60`), `below_weight` (default `0.40`).

Source of truth: `MovingAverageTrend` in `Stock/sp500/strategies/moving_average.py:5`. Like the original, it trades only when the target changes — drift between flips is not corrected. `_current_target` starts as `None` so the first post-warm-up bar always establishes the position.

- [ ] **Step 1: Write the failing test**

Create `tests/test_spy_ma_trend.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_spy_ma_trend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spy_ma_trend'`

- [ ] **Step 3: Write `algorithms/spy_ma_trend.py`**

```python
from AlgorithmImports import *


class SpyMaTrend(QCAlgorithm):
    """Dynamic SPY allocation driven by a simple moving average.

    Overweight SPY above the average, underweight below it, always partially
    invested. Port of MovingAverageTrend in
    sp500/strategies/moving_average.py. Rebalances only on a signal flip, so
    the allocation drifts between flips exactly as the original does.
    """

    def initialize(self):
        self.set_start_date(2000, 1, 3)
        self.set_end_date(2021, 3, 31)   # bundled local data ends here; delete this line on QC
        self.set_cash(10000)

        self.ma_period = int(self.get_parameter("ma_period", 200))
        self.above_weight = float(self.get_parameter("above_weight", 0.60))
        self.below_weight = float(self.get_parameter("below_weight", 0.40))

        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.set_benchmark(self.spy)

        self.moving_average = self.sma(self.spy, self.ma_period, Resolution.DAILY)
        self.set_warm_up(self.ma_period, Resolution.DAILY)

        self._current_target = None

    def on_data(self, data: Slice):
        if self.is_warming_up or not self.moving_average.is_ready:
            return
        if not data.bars.contains_key(self.spy):
            return

        close = float(data.bars[self.spy].close)
        above = close >= float(self.moving_average.current.value)
        target = self.above_weight if above else self.below_weight

        if target == self._current_target:
            return

        self._current_target = target
        self.set_holdings(self.spy, target)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_spy_ma_trend.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add algorithms/spy_ma_trend.py tests/test_spy_ma_trend.py
git commit -m "feat: port MA trend following strategy to LEAN Python"
```

---

### Task 7: `spy_vol_adjusted.py`

**Files:**
- Create: `algorithms/spy_vol_adjusted.py`
- Test: `tests/test_spy_vol_adjusted.py`

**Interfaces:**
- Consumes: `lean_stubs.feed`, `algorithm.orders` from Task 1.
- Produces: class `SpyVolAdjusted`, parameters `lookback` (default `20`), `target_vol` (default `0.15`), `min_weight` (default `0.10`), `max_weight` (default `0.90`); class attribute `TRADING_DAYS_PER_YEAR = 252`.

Source of truth: `Stock/sp500/strategies/volatility_adjusted.py`.

Two details that matter:
- The return window is updated **before** the `is_warming_up` guard. LEAN calls `on_data` during warm-up, so a window updated after the guard would never fill from warm-up history.
- `lookback` returns require `lookback + 1` prices, matching `volatility_adjusted.py:48-53`, hence `set_warm_up(lookback + 1, ...)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_spy_vol_adjusted.py`:

```python
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
    # A steady 1% compounding drift gives near-zero vol -> huge raw weight.
    algo = run_algorithm(
        SpyVolAdjusted, [100.0, 101.0, 102.01, 103.0301], SHORT_LOOKBACK
    )
    assert algo.orders[-1][2] == 0.90


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_spy_vol_adjusted.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spy_vol_adjusted'`

- [ ] **Step 3: Write `algorithms/spy_vol_adjusted.py`**

```python
import math

from AlgorithmImports import *


class SpyVolAdjusted(QCAlgorithm):
    """Inverse-volatility position sizing, rebalanced every day.

    SPY weight = target_vol / realised_vol, clamped to [min_weight,
    max_weight]. Port of VolatilityAdjusted in
    sp500/strategies/volatility_adjusted.py.

    Note: the original rebalances daily at zero cost. Under a real fee model
    this strategy pays for that churn; that cost is genuine, not a defect.
    """

    TRADING_DAYS_PER_YEAR = 252

    def initialize(self):
        self.set_start_date(2000, 1, 3)
        self.set_end_date(2021, 3, 31)   # bundled local data ends here; delete this line on QC
        self.set_cash(10000)

        self.lookback = int(self.get_parameter("lookback", 20))
        self.target_vol = float(self.get_parameter("target_vol", 0.15))
        self.min_weight = float(self.get_parameter("min_weight", 0.10))
        self.max_weight = float(self.get_parameter("max_weight", 0.90))

        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.set_benchmark(self.spy)

        self.returns = RollingWindow[float](self.lookback)
        self._previous_close = None

        # lookback returns need lookback + 1 prices.
        self.set_warm_up(self.lookback + 1, Resolution.DAILY)

    def on_data(self, data: Slice):
        if not data.bars.contains_key(self.spy):
            return

        close = float(data.bars[self.spy].close)

        # Update the window before the warm-up guard: on_data runs during
        # warm-up, so guarding first would leave the window permanently empty.
        if self._previous_close:
            self.returns.add(close / self._previous_close - 1.0)
        self._previous_close = close

        if self.is_warming_up or not self.returns.is_ready:
            return

        volatility = self._annualised_volatility()
        if volatility == 0:
            return

        raw_weight = self.target_vol / volatility
        weight = max(self.min_weight, min(self.max_weight, raw_weight))
        self.set_holdings(self.spy, weight)

    def _annualised_volatility(self):
        values = [self.returns[i] for i in range(self.returns.count)]
        count = len(values)
        if count < 2:
            return 0.0
        mean = sum(values) / count
        # ddof=1 sample variance, matching pandas Series.std().
        variance = sum((value - mean) ** 2 for value in values) / (count - 1)
        return math.sqrt(variance) * math.sqrt(self.TRADING_DAYS_PER_YEAR)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_spy_vol_adjusted.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add algorithms/spy_vol_adjusted.py tests/test_spy_vol_adjusted.py
git commit -m "feat: port volatility adjusted strategy to LEAN Python"
```

---

### Task 8: `spy_momentum.py`

**Files:**
- Create: `algorithms/spy_momentum.py`
- Test: `tests/test_spy_momentum.py`

**Interfaces:**
- Consumes: `lean_stubs.feed`, `algorithm.orders` from Task 1.
- Produces: class `SpyMomentum`, parameter `lookback_months` (default `12`); class attribute `TRADING_DAYS_PER_MONTH = 21`, instance attribute `required_days = lookback_months * 21`.

Source of truth: `Stock/sp500/strategies/momentum.py`. Keeps the original's "one month = 21 trading days" convention (`momentum.py:48`) rather than calendar months. The comparison price is the oldest entry in the window, equal to `prices_history.iloc[-required_days]`. `_in_market` starts `False`, as in `spy_ma_entry_exit`. As in Task 7, the close window is updated before the warm-up guard.

- [ ] **Step 1: Write the failing test**

Create `tests/test_spy_momentum.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_spy_momentum.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spy_momentum'`

- [ ] **Step 3: Write `algorithms/spy_momentum.py`**

```python
from AlgorithmImports import *


class SpyMomentum(QCAlgorithm):
    """Time-series momentum, re-evaluated on the first trading day of a month.

    100% SPY when the price is above where it stood lookback_months ago,
    otherwise 100% cash. Port of Momentum in sp500/strategies/momentum.py,
    keeping the original's 21-trading-days-per-month convention.
    """

    TRADING_DAYS_PER_MONTH = 21

    def initialize(self):
        self.set_start_date(2000, 1, 3)
        self.set_end_date(2021, 3, 31)   # bundled local data ends here; delete this line on QC
        self.set_cash(10000)

        self.lookback_months = int(self.get_parameter("lookback_months", 12))
        self.required_days = self.lookback_months * self.TRADING_DAYS_PER_MONTH

        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.set_benchmark(self.spy)

        self.closes = RollingWindow[float](self.required_days)
        self.set_warm_up(self.required_days, Resolution.DAILY)

        self._in_market = False
        self._last_check_month = None

    def on_data(self, data: Slice):
        if not data.bars.contains_key(self.spy):
            return

        close = float(data.bars[self.spy].close)

        # Update before the warm-up guard so warm-up history fills the window.
        self.closes.add(close)

        if self.is_warming_up or not self.closes.is_ready:
            return

        month = (self.time.year, self.time.month)
        if month == self._last_check_month:
            return
        self._last_check_month = month

        past_close = self.closes[self.closes.count - 1]
        should_be_in = close > past_close
        if should_be_in == self._in_market:
            return

        self._in_market = should_be_in
        self.set_holdings(self.spy, 1.0 if should_be_in else 0.0)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_spy_momentum.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add algorithms/spy_momentum.py tests/test_spy_momentum.py
git commit -m "feat: port 12-month momentum strategy to LEAN Python"
```

---

### Task 9: `variants.py`, `run_local.py` and the README

**Files:**
- Create: `variants.py`
- Create: `run_local.py`
- Create: `README.md`
- Test: `tests/test_variants.py`

**Interfaces:**
- Consumes: the seven algorithm modules and their class names from Tasks 2-8.
- Produces: `variants.ALGORITHM_CLASSES` (module name → class name), `variants.VARIANTS` (display name → `(module_name, parameters_dict)`), `variants.slug(name)` → filesystem-safe string.

`run_local.py` is not unit-tested — it shells out to the LEAN engine, which cannot run Python algorithms until the prerequisites in the README are installed. `variants.py` is tested because it is pure data that must stay in step with the algorithm files.

- [ ] **Step 1: Write the failing test**

Create `tests/test_variants.py`:

```python
import importlib

from variants import ALGORITHM_CLASSES, VARIANTS, slug


def test_all_sixteen_variants_from_main_py_are_present():
    assert len(VARIANTS) == 16


def test_every_variant_points_at_a_known_algorithm_module():
    for name, (module_name, _) in VARIANTS.items():
        assert module_name in ALGORITHM_CLASSES, name


def test_every_algorithm_module_imports_and_defines_its_class():
    for module_name, class_name in ALGORITHM_CLASSES.items():
        module = importlib.import_module(module_name)
        assert hasattr(module, class_name), f"{module_name}.{class_name}"


def test_every_variant_parameter_is_accepted_by_its_algorithm():
    for name, (module_name, parameters) in VARIANTS.items():
        module = importlib.import_module(module_name)
        algorithm = getattr(module, ALGORITHM_CLASSES[module_name])()
        algorithm._parameters = {k: str(v) for k, v in parameters.items()}
        algorithm.initialize()
        for key in parameters:
            assert hasattr(algorithm, key), f"{name}: {key}"


def test_slugs_are_unique_and_filesystem_safe():
    slugs = [slug(name) for name in VARIANTS]
    assert len(set(slugs)) == len(slugs)
    for value in slugs:
        assert value.replace("_", "").isalnum(), value
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_variants.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'variants'`

- [ ] **Step 3: Add the repo root to the test path**

Modify `tests/conftest.py` — add this line after the two existing `sys.path.insert` calls:

```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```

- [ ] **Step 4: Write `variants.py`**

```python
"""The 16 named strategy variants from sp500/main.py:14-37.

Local tooling for run_local.py. Never uploaded to QuantConnect — the files in
algorithms/ are self-contained and carry their own defaults.
"""

import re

ALGORITHM_CLASSES = {
    "spy_buy_and_hold": "SpyBuyAndHold",
    "spy_threshold_rebalance": "SpyThresholdRebalance",
    "spy_periodic_rebalance": "SpyPeriodicRebalance",
    "spy_ma_trend": "SpyMaTrend",
    "spy_ma_entry_exit": "SpyMaEntryExit",
    "spy_vol_adjusted": "SpyVolAdjusted",
    "spy_momentum": "SpyMomentum",
}

VARIANTS = {
    # Threshold variants
    "50/50 Rebalance ±1%": ("spy_threshold_rebalance", {"target": 0.50, "threshold": 0.01}),
    "50/50 Rebalance ±5%": ("spy_threshold_rebalance", {"target": 0.50, "threshold": 0.05}),
    "50/50 Rebalance ±10%": ("spy_threshold_rebalance", {"target": 0.50, "threshold": 0.10}),
    "50/50 Rebalance ±20%": ("spy_threshold_rebalance", {"target": 0.50, "threshold": 0.20}),
    # Different target allocations
    "60/40 Rebalance ±5%": ("spy_threshold_rebalance", {"target": 0.60, "threshold": 0.05}),
    "70/30 Rebalance ±5%": ("spy_threshold_rebalance", {"target": 0.70, "threshold": 0.05}),
    "80/20 Rebalance ±5%": ("spy_threshold_rebalance", {"target": 0.80, "threshold": 0.05}),
    # Periodic rebalancing
    "50/50 Monthly Rebalance": ("spy_periodic_rebalance", {"target": 0.50, "frequency": "M"}),
    "50/50 Quarterly Rebalance": ("spy_periodic_rebalance", {"target": 0.50, "frequency": "Q"}),
    "50/50 Annual Rebalance": ("spy_periodic_rebalance", {"target": 0.50, "frequency": "Y"}),
    # Buy and hold benchmarks
    "100% Buy & Hold SPY": ("spy_buy_and_hold", {"spy_weight": 1.0}),
    "50/50 No Rebalance": ("spy_buy_and_hold", {"spy_weight": 0.5}),
    # Dynamic allocation
    "MA Trend Following (200d)": (
        "spy_ma_trend",
        {"ma_period": 200, "above_weight": 0.60, "below_weight": 0.40},
    ),
    "Volatility Adjusted (20d)": (
        "spy_vol_adjusted",
        {"lookback": 20, "target_vol": 0.15, "min_weight": 0.10, "max_weight": 0.90},
    ),
    # Market timing
    "200-day MA Entry/Exit": ("spy_ma_entry_exit", {"ma_period": 200}),
    "12-Month Momentum": ("spy_momentum", {"lookback_months": 12}),
}


def slug(name):
    """Filesystem-safe identifier for a variant name."""
    cleaned = (
        name.replace("&", "and")
        .replace("%", "pct")
        .replace("±", "pm")
        .replace("/", "_")
    )
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", cleaned)
    return cleaned.strip("_").lower()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_variants.py -v`
Expected: PASS, 5 tests

- [ ] **Step 6: Write `run_local.py`**

```python
"""Run the algorithms in algorithms/ against the local LEAN engine.

Builds a throwaway config from Stock/Lean/Launcher/config.json, overriding the
algorithm, its parameters and the results directory, then invokes the Launcher
with --config. Nothing under Stock/Lean/ is modified.

Requires the local LEAN Python prerequisites — see README.md.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from variants import ALGORITHM_CLASSES, VARIANTS, slug

REPO_ROOT = Path(__file__).resolve().parent
LEAN_ROOT = REPO_ROOT.parent / "Lean"
BASE_CONFIG = LEAN_ROOT / "Launcher" / "config.json"
LAUNCHER_DIR = LEAN_ROOT / "Launcher" / "bin" / "Debug"
LAUNCHER_DLL = LAUNCHER_DIR / "QuantConnect.Lean.Launcher.dll"
RESULTS_ROOT = REPO_ROOT / "results"

REPORTED = [
    "Start Equity",
    "End Equity",
    "Compounding Annual Return",
    "Drawdown",
    "Sharpe Ratio",
    "Total Orders",
]


def strip_json_comments(text):
    """Remove // comments from LEAN's config while respecting string literals."""
    out = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < len(text) and text[index + 1] == "/":
            while index < len(text) and text[index] != "\n":
                index += 1
            continue
        out.append(char)
        index += 1
    return "".join(out)


def python_dll():
    """Locate the CPython DLL that pythonnet must load.

    Under ``uv run`` inside this project's virtualenv, sys.base_prefix points
    at the standalone x64 CPython 3.11 uv installed, and python311.dll sits in
    that directory. LEAN_PYTHON_DLL overrides for unusual installs.
    """
    override = os.environ.get("LEAN_PYTHON_DLL")
    if override:
        return override
    candidate = Path(sys.base_prefix) / "python311.dll"
    if not candidate.exists():
        raise RuntimeError(
            f"python311.dll not found at {candidate}. "
            f"Install it with 'uv python install 3.11.11' or set LEAN_PYTHON_DLL."
        )
    return str(candidate)


def build_config(module_name, parameters, results_dir):
    config = json.loads(strip_json_comments(BASE_CONFIG.read_text(encoding="utf-8")))
    config["environment"] = "backtesting"
    config["algorithm-language"] = "Python"
    config["algorithm-type-name"] = ALGORITHM_CLASSES[module_name]
    config["algorithm-location"] = str(REPO_ROOT / "algorithms" / f"{module_name}.py")
    config["data-folder"] = f"{LEAN_ROOT / 'Data'}/"
    config["results-destination-folder"] = str(results_dir)
    config["close-automatically"] = True
    # Lets LEAN resolve pandas/wrapt from this project's virtualenv.
    config["python-venv"] = str(REPO_ROOT / ".venv")
    config["parameters"] = {key: str(value) for key, value in parameters.items()}
    return config


def run_variant(name):
    module_name, parameters = VARIANTS[name]
    results_dir = RESULTS_ROOT / slug(name)
    results_dir.mkdir(parents=True, exist_ok=True)
    for stale in results_dir.glob("*-summary.json"):
        stale.unlink()

    config = build_config(module_name, parameters, results_dir)
    handle, config_path = tempfile.mkstemp(suffix=".json", text=True)
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)

    dotnet = os.environ.get("LEAN_DOTNET", "dotnet")
    env = dict(os.environ)
    env["PYTHONNET_PYDLL"] = python_dll()
    try:
        subprocess.run(
            [dotnet, str(LAUNCHER_DLL), "--config", config_path],
            cwd=str(LAUNCHER_DIR),
            stdin=subprocess.DEVNULL,
            env=env,
            check=True,
        )
    finally:
        os.unlink(config_path)

    summaries = list(results_dir.glob("*-summary.json"))
    if not summaries:
        raise RuntimeError(f"{name}: LEAN produced no summary in {results_dir}")
    return json.loads(summaries[0].read_text(encoding="utf-8"))["statistics"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", nargs="?", help="variant name to run")
    parser.add_argument("--all", action="store_true", help="run every variant")
    parser.add_argument("--list", action="store_true", help="list variant names")
    args = parser.parse_args()

    if args.list:
        for name in VARIANTS:
            print(name)
        return 0

    if args.all:
        names = list(VARIANTS)
    elif args.variant:
        if args.variant not in VARIANTS:
            print(f"Unknown variant {args.variant!r}. Use --list.", file=sys.stderr)
            return 2
        names = [args.variant]
    else:
        parser.print_help()
        return 2

    if not LAUNCHER_DLL.exists():
        print(f"LEAN Launcher not built at {LAUNCHER_DLL}", file=sys.stderr)
        return 1

    results = {}
    for name in names:
        print(f"\n=== {name} ===", flush=True)
        results[name] = run_variant(name)

    width = max(len(name) for name in results)
    print(f"\n{'Variant'.ljust(width)}  " + "  ".join(REPORTED))
    for name, statistics in results.items():
        cells = [str(statistics.get(key, "-")) for key in REPORTED]
        print(f"{name.ljust(width)}  " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Verify `run_local.py --list` works**

Run: `uv run run_local.py --list`
Expected: 16 variant names printed, exit 0. (Running an actual backtest needs the prerequisites below and is not part of this step.)

- [ ] **Step 8: Write `README.md`**

```markdown
# lean_strategies

LEAN Python ports of the SPY allocation strategies in `../sp500/strategies/`.

Each file in `algorithms/` is self-contained: its only import is
`from AlgorithmImports import *` (plus stdlib `math` where needed), and every
parameter has a default. Copy one into a QuantConnect project and it runs.

## Using these on quantconnect.com

1. Create a new Python algorithm project.
2. Paste the contents of one `algorithms/*.py` file over `main.py`.
3. Delete the `set_end_date(2021, 3, 31)` line — that bound exists only because
   the SPY data bundled with the local LEAN repo stops on 2021-03-31.
4. To run a variant other than the default, set the parameters listed in
   `variants.py` in the project's Parameters panel.

## Running locally

Prerequisites:

1. A **x64** .NET 10 SDK. An x86 SDK cannot load a 64-bit Python and will fail
   at startup.
2. Python 3.11.11 x64 — `uv python install 3.11.11` provides one.
3. A built LEAN Launcher: `dotnet build ../Lean/Launcher/QuantConnect.Lean.Launcher.csproj`

`PYTHONNET_PYDLL` does not need to be set system-wide. `run_local.py` derives
it from `sys.base_prefix` and passes it to the engine for that run only, and
points LEAN's `python-venv` setting at this project's `.venv` so pandas and
wrapt resolve.

```
uv sync
uv run run_local.py --list
uv run run_local.py "200-day MA Entry/Exit"
uv run run_local.py --all
```

Results land in `results/<slug>/` and a comparison table is printed. Set
`LEAN_DOTNET` if the x64 `dotnet` is not first on `PATH`, or `LEAN_PYTHON_DLL`
if `python311.dll` lives somewhere unusual.

## Verifying the port

`200-day MA Entry/Exit` over 2000-01-03 → 2021-03-31 should reproduce the
figures already measured from the equivalent C# algorithm:

| Statistic | Expected |
|---|---|
| End Equity | 31531.97 |
| Compounding Annual Return | 5.551% |
| Drawdown | 24.200% |
| Total Orders | 145 |

Same data, parameters and fee model, so these should match. The other six
strategies have no such anchor; compare their relative ordering against
`../sp500/results/analysis_report.md`, remembering that the Python engine
executes at the same day's close and charges no fees.

## Tests

```
uv run pytest
```

The tests drive each algorithm through a fake `AlgorithmImports`
(`tests/lean_stubs.py`) and assert on the allocation decisions. They cover
decision logic only — fills, fees and real warm-up are LEAN's job and are
verified by the anchor run above.
```

- [ ] **Step 9: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS, all tests from Tasks 1-9

- [ ] **Step 10: Commit**

```bash
git add variants.py run_local.py README.md tests/test_variants.py tests/conftest.py
git commit -m "feat: add variant registry, local runner and README"
```

---

## Self-Review

**Spec coverage.** Every section of the spec maps to a task: project structure and environment → Task 1 and Task 9; the common skeleton → Tasks 2-8 (each repeats it in full); the seven per-algorithm specifications → Tasks 2-8 one each; `variants.py` and `run_local.py` → Task 9; verification → Task 9's README and the `spy_ma_entry_exit` anchor. The three accepted semantic differences are carried into the code: warm-up via `set_warm_up` in Tasks 5-8, faithful daily rebalancing in Task 7 with the fee cost documented in the docstring, and the `ddof=1` sample deviation in Task 7 (written with `math` instead of `numpy` — deviation recorded under Global Constraints).

**Placeholder scan.** No TBD/TODO markers, no "add error handling", no "similar to Task N" — each task repeats its code in full.

**Test helper sharing.** `run_algorithm` lives in `tests/lean_stubs.py` (Task 1) and is the single entry point used by the tests in Tasks 2, 3, 5, 6, 7 and 8. Task 4 builds bars by hand and calls `feed` directly because its assertions depend on specific calendar months, which consecutive-day bars cannot express.

**Type consistency.** `algorithm.orders` is a list of `(datetime, symbol, weight)` throughout Tasks 1-9. `RollingWindow` exposes `add`, `count`, `is_ready` and `__getitem__` in the fake (Task 1) and only those members are used in Tasks 7-8. The SMA indicator exposes `is_ready` and `current.value` in the fake and only those are used in Tasks 5-6. `ALGORITHM_CLASSES` keys are module names matching the seven filenames created in Tasks 2-8, and `slug` is defined in `variants.py` and consumed by `run_local.py`.
