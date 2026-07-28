# Design: Port sp500 strategies to LEAN Python

Date: 2026-07-28

## Goal

Rewrite the seven strategy classes in `Stock/sp500/strategies/` as LEAN Python
algorithms (`QCAlgorithm`, snake_case API) so that each file can be pasted
directly into a quantconnect.com project and run unchanged, and the same files
can also run against the local LEAN engine in `Stock/Lean/`.

The existing `Stock/sp500/` project keeps its own pandas-based engine. This is a
parallel port, not a replacement.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Where the code lives | New `Stock/lean_strategies/`, its own git repo and uv project | `Stock/sp500/` pins `pandas>=3.0.2`; LEAN requires pandas 2.2.3. The environments cannot be shared. `Stock/Lean/` is an upstream repo — adding files there creates merge noise on every pull. |
| Execution target | Both QC website and local LEAN | User will iterate locally and run full-history backtests in the cloud. |
| File granularity | One self-contained file per strategy class (7 files); the 16 configured variants come from parameters | A self-contained file pastes into QC with no import wiring. Parameters map onto QC's project Parameters panel and its Optimization feature. |
| Backtest realism | LEAN defaults — next-bar fills, Interactive Brokers fee model | The Python original executes at the same day's close (`sp500/backtest.py:35`), a mild look-ahead, and charges no fees. LEAN's defaults are the trustworthy ones. |

## Prerequisites (user action required)

Local Python algorithms cannot run until these are done. Per
`Stock/Lean/Algorithm.Python/readme.md`:

1. Replace the currently installed **x86** .NET 10 SDK (`C:\Program Files (x86)\dotnet\sdk\10.0.302`) with the **x64** build. A 32-bit LEAN cannot load a 64-bit Python DLL.
2. Install **Python 3.11.11 x64**.
3. Set the `PYTHONNET_PYDLL` environment variable to that installation's `python311.dll`.

C# algorithms already work under the x86 SDK and are unaffected.

## Project structure

```
Stock/lean_strategies/
├── pyproject.toml          # python 3.11, pandas 2.2.3, wrapt 1.16.0, quantconnect-stubs
├── README.md               # prerequisites, local run instructions, QC paste instructions
├── algorithms/             # each file is standalone and paste-ready
│   ├── spy_buy_and_hold.py
│   ├── spy_threshold_rebalance.py
│   ├── spy_periodic_rebalance.py
│   ├── spy_ma_trend.py
│   ├── spy_ma_entry_exit.py
│   ├── spy_vol_adjusted.py
│   └── spy_momentum.py
├── variants.py             # the 16 named variants from sp500/main.py:14-37
├── run_local.py            # generates a temp config, invokes the LEAN Launcher
├── results/                # per-variant LEAN output, git-ignored
└── docs/superpowers/specs/ # this document
```

Files under `algorithms/` import nothing but `from AlgorithmImports import *`.
`variants.py` and `run_local.py` are local tooling and are never uploaded to QC.

## Common skeleton

Every algorithm file follows this shape:

```python
from AlgorithmImports import *


class SpyThresholdRebalance(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2000, 1, 3)
        self.set_end_date(2021, 3, 31)   # bundled local data ends here; delete this line on QC
        self.set_cash(10000)

        self.target = float(self.get_parameter("target", 0.50))
        self.threshold = float(self.get_parameter("threshold", 0.05))

        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.set_benchmark(self.spy)
```

Start date and initial capital match `sp500/main.py:43` and `sp500/main.py:52`.
Every `get_parameter` call supplies a default, so a pasted file runs on QC with
no parameters configured.

## Deliberate semantic differences from the Python original

These three were reviewed and accepted:

1. **Warm-up.** The original holds a 50/50 default through the indicator warm-up
   window (`sp500/strategies/moving_average.py:71`, `momentum.py:19`). The LEAN
   port uses `set_warm_up(n, Resolution.DAILY)` so indicators are primed from
   pre-start history and a real signal exists on the first trading day. No dead
   period.

2. **`spy_vol_adjusted` under real fees.** The original rebalances every day at
   zero cost. With the IB fee model and $10,000 of capital, fees will visibly
   erode this strategy. It is ported faithfully anyway; the poor result is real,
   not a defect. LEAN's `set_holdings` suppresses orders that round to zero
   shares, which reduces the trade count somewhat.

3. **Standard deviation convention.** pandas `.std()` defaults to `ddof=1`;
   LEAN's built-in `StandardDeviation` indicator uses the population formula.
   To match the original, `spy_vol_adjusted` keeps daily returns in a
   `RollingWindow` and computes `numpy.std(returns, ddof=1) * numpy.sqrt(252)`
   rather than using the built-in indicator. Likewise `spy_momentum` keeps the
   original's "one month = 21 trading days" convention
   (`sp500/strategies/momentum.py:48`) instead of calendar months.

## Per-algorithm specification

### `spy_buy_and_hold.py` — `SpyBuyAndHold`
Source: `strategies/buy_and_hold.py`. Parameter: `spy_weight` (1.0).

On the first bar containing SPY, `set_holdings(spy, spy_weight)` and set an
`_initialized` flag. Never trade again — the allocation drifts freely.

### `spy_threshold_rebalance.py` — `SpyThresholdRebalance`
Source: `strategies/rebalance_50_50.py`. Parameters: `target` (0.50),
`threshold` (0.05).

First bar: `set_holdings(spy, target)`. Every subsequent bar compute
`w = portfolio[spy].holdings_value / portfolio.total_portfolio_value`; when
`abs(w - target) >= threshold`, rebalance back to `target`. The `>=` matches
`rebalance_50_50.py:38`.

### `spy_periodic_rebalance.py` — `SpyPeriodicRebalance`
Source: `strategies/rebalance_periodic.py`. Parameters: `target` (0.50),
`frequency` (`"M"`).

Period key mirrors `_period_key()`: `M` → `(year, month)`, `Q` →
`(year, (month - 1) // 3)`, `Y` → `(year,)`. On each bar, if the key differs
from the stored one, store it and `set_holdings(spy, target)`. The first bar
naturally triggers the initial allocation. Reject any other `frequency` value
with a clear error in `initialize`.

### `spy_ma_trend.py` — `SpyMaTrend`
Source: `strategies/moving_average.py` class `MovingAverageTrend`. Parameters:
`ma_period` (200), `above_weight` (0.60), `below_weight` (0.40).

`self.sma(spy, ma_period, Resolution.DAILY)` plus `set_warm_up(ma_period, Resolution.DAILY)`.
Target is `above_weight` when close >= SMA, else `below_weight`. Trade only when
the target changes, plus once on the first post-warm-up bar to establish the
position. Between flips the portfolio drifts without correction, matching the
original.

### `spy_ma_entry_exit.py` — `SpyMaEntryExit`
Source: `strategies/moving_average.py` class `MovingAverageEntryExit`.
Parameter: `ma_period` (200).

Same SMA and warm-up setup. `_in_market` starts `False`. Each post-warm-up bar,
`should_be_in = close >= sma`; when it differs from `_in_market`, update the
flag and `set_holdings(spy, 1.0 if should_be_in else 0.0)`. Starting `_in_market`
at `False` means the algorithm begins flat and acts on the first real signal —
this is what the already-validated C# port does.

### `spy_vol_adjusted.py` — `SpyVolAdjusted`
Source: `strategies/volatility_adjusted.py`. Parameters: `lookback` (20),
`target_vol` (0.15), `min_weight` (0.10), `max_weight` (0.90).

Keep a `RollingWindow[float]` of the last `lookback` daily returns, computed
from the previous close. `set_warm_up(lookback + 1, Resolution.DAILY)`. Once the
window is full: `vol = numpy.std(returns, ddof=1) * numpy.sqrt(252)`; skip the
bar if `vol == 0`; otherwise
`w = clip(target_vol / vol, min_weight, max_weight)` and `set_holdings(spy, w)`
every bar. `lookback` returns require `lookback + 1` prices, matching
`volatility_adjusted.py:48-53`.

### `spy_momentum.py` — `SpyMomentum`
Source: `strategies/momentum.py`. Parameter: `lookback_months` (12).

`required_days = lookback_months * 21`. Keep a `RollingWindow[float]` of that
many closes; `set_warm_up(required_days, Resolution.DAILY)`. Evaluate only when
`(year, month)` differs from the stored key. `should_be_in = close > oldest`
where `oldest` is the window's last element, matching
`prices_history.iloc[-required_days]`. `_in_market` starts `False`; on a change,
`set_holdings(spy, 1.0 if should_be_in else 0.0)`.

## `variants.py`

Maps each of the 16 names in `sp500/main.py:14-37` to `(module_name, parameters)`:

| Name | Module | Parameters |
|---|---|---|
| 50/50 Rebalance ±1% | `spy_threshold_rebalance` | target 0.50, threshold 0.01 |
| 50/50 Rebalance ±5% | `spy_threshold_rebalance` | target 0.50, threshold 0.05 |
| 50/50 Rebalance ±10% | `spy_threshold_rebalance` | target 0.50, threshold 0.10 |
| 50/50 Rebalance ±20% | `spy_threshold_rebalance` | target 0.50, threshold 0.20 |
| 60/40 Rebalance ±5% | `spy_threshold_rebalance` | target 0.60, threshold 0.05 |
| 70/30 Rebalance ±5% | `spy_threshold_rebalance` | target 0.70, threshold 0.05 |
| 80/20 Rebalance ±5% | `spy_threshold_rebalance` | target 0.80, threshold 0.05 |
| 50/50 Monthly Rebalance | `spy_periodic_rebalance` | target 0.50, frequency M |
| 50/50 Quarterly Rebalance | `spy_periodic_rebalance` | target 0.50, frequency Q |
| 50/50 Annual Rebalance | `spy_periodic_rebalance` | target 0.50, frequency Y |
| 100% Buy & Hold SPY | `spy_buy_and_hold` | spy_weight 1.0 |
| 50/50 No Rebalance | `spy_buy_and_hold` | spy_weight 0.5 |
| MA Trend Following (200d) | `spy_ma_trend` | ma_period 200, above 0.60, below 0.40 |
| Volatility Adjusted (20d) | `spy_vol_adjusted` | lookback 20, target_vol 0.15, min 0.10, max 0.90 |
| 200-day MA Entry/Exit | `spy_ma_entry_exit` | ma_period 200 |
| 12-Month Momentum | `spy_momentum` | lookback_months 12 |

## `run_local.py`

Usage: `uv run run_local.py "<variant name>"` or `uv run run_local.py --all`.

For each selected variant:

1. Read `Stock/Lean/Launcher/config.json` as the base configuration.
2. Override `algorithm-language` to `Python`, `algorithm-location` to the
   absolute path of the algorithm file, `parameters` to the variant's values
   (all stringified — LEAN parameters are strings), and
   `results-destination-folder` to a per-variant output directory
   (`Stock/lean_strategies/results/<slug>/`).
3. Write the merged config to a temp file.
4. Invoke `dotnet <Lean>/Launcher/bin/Debug/QuantConnect.Lean.Launcher.dll --config <temp file>`.
5. Read the resulting `*-summary.json` and collect the headline statistics.

Finally print a comparison table across all runs. Nothing under `Stock/Lean/` is
modified — the `--config` flag (`Lean/Configuration/Config.cs:59`,
`Lean/Configuration/LeanArgumentParser.cs:37`) and `results-destination-folder`
(`Lean/Common/Globals.cs:100`) are both existing engine features.

## Verification

`spy_ma_entry_exit.py` run with `ma_period=200` over 2000-01-03 → 2021-03-31 must
reproduce the already-measured C# result: **end equity $31,531.97, CAGR 5.551%,
drawdown 24.200%, 145 orders, $145 fees**. Same data, same parameters, same fee
model, so the numbers should match. This is the anchor proving the port is
faithful.

The other six have no such anchor. They are checked by confirming their relative
ordering and rough magnitudes are consistent with `sp500/results/analysis_report.md`,
after accounting for the different execution timing and fees.

## Out of scope

- Extending market data beyond the bundled SPY daily history (ends 2021-03-31). Full-history runs happen on QC.
- Charts and markdown reports equivalent to `sp500/reporting.py`. `run_local.py` prints a table; LEAN already writes its own result JSON.
- Any change to `Stock/sp500/` or `Stock/Lean/`.
