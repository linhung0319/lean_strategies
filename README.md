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
`../sp500/results/analysis_report.md`, remembering that the original pandas
backtester executes at the same day's close and charges no fees.

## Converting your own data

The LEAN repo bundles SPY and about twenty other tickers, all stopping on
2021-03-31. Anything else — a different symbol, or a more recent date —
has to be downloaded and written in LEAN's own on-disk format. `leandata/`
does that.

### Quickstart

```
uv sync --group data                                  # once: installs yfinance

uv run --group data convert_data.py convert --source yfinance --ticker NVDA \
    --exchange-code Q --start 2015-01-02 --end 2026-08-07

uv run convert_data.py check                          # confirm the folder is usable
uv run run_local.py "NVDA 200-day MA Entry/Exit" --data-folder data
```

`../Lean/Data` is never written to. Everything lands in `data/`, which is
half committed and half ignored — see
[What a data folder must contain](#what-a-data-folder-must-contain).

`yfinance` sits in a non-default dependency group on purpose. `run_local.py`
points LEAN's `python-venv` at this same `.venv`, so anything installed here
also lands on the embedded interpreter's path, and the backtests have no
business carrying `curl_cffi`, `lxml` and `protobuf`.

### The pipeline

```
  yfinance          csv           <your next source>
      └───────────────┴───────────────┘
                      │   sources/<name>.py        vendor format -> canonical
                      ▼
              SecurityHistory                      RAW OHLCV + dividends + splits,
                      │                            naive exchange-local time
                      │   lean/writer.py           canonical -> LEAN bytes
                      ▼
       data/equity/usa/{daily,factor_files,map_files}/<ticker>.*
                      +
       data/{market-hours,symbol-properties,alternative}/     mirrored from ../Lean/Data
                      │   run_local.py --data-folder data
                      ▼
                LEAN backtest
```

Three layers, deliberately kept apart: `model.py` defines the canonical
`SecurityHistory`, `sources/` turns a vendor download into one, `lean/` turns
one into bytes. `lean/` never imports `sources/` and vice versa.

### What a data folder must contain

Pointing `--data-folder` at a directory **replaces** `../Lean/Data` wholesale
— it is not layered on top. So the folder needs the reference data LEAN reads
regardless of which symbols you asked for. `convert` handles all of this, but
it is worth knowing what it did and which parts go stale.

| Path | Where it comes from | In git? | Time-sensitive? | Do you need to do anything? |
|---|---|---|---|---|
| `equity/usa/daily/<ticker>.zip` | **generated** per symbol | no | yes — one row per session | re-run `convert` with a later `--end` |
| `equity/usa/factor_files/<ticker>.csv` | **generated** per symbol | no | yes — dividends and splits | same |
| `equity/usa/map_files/<ticker>.csv` | **generated** per symbol | no | no | same |
| `market-hours/` | **copied** from `../Lean/Data` | yes | yes — holiday calendar | **no.** Holidays run to 2027-12-24 and upstream actively maintains the file |
| `symbol-properties/` | **copied** from `../Lean/Data` | yes | no — no date column at all | **no.** The `usa,[*],equity` wildcard covers every US equity |
| `alternative/interest-rate/usa/` | **copied, then topped up** | yes | yes — FOMC rate changes | **yes, eventually.** See below |

New tickers need no registration anywhere: `market-hours` and
`symbol-properties` both match through `[*]` wildcards.

The git split follows what the file *is*, not where it sits. Price data is one
`convert` command away and grows with every symbol and every re-download, so
it stays out. The reference data is ~3.9 MB, static, shared by every symbol,
and in one case hand-maintained, so it is committed: a fresh checkout can then
run a backtest without a Lean clone present, and an upstream change to the
holiday table shows up as a reviewable diff instead of silently moving
results. `tests/fixtures/` is the deliberate exception — it holds downloaded
prices, but a fixed two symbols that never grow, and the cross-validation test
has to run offline.

`check` prints exactly what a folder holds:

```
$ uv run convert_data.py check
data folder .../lean_strategies/data
  NVDA       2916 rows  2015-01-02 -> 2026-08-07
  SPY        5849 rows  1998-01-02 -> 2021-03-31
  risk-free rate curve 2003-01-09 -> 2025-12-11, held flat after that
```

#### The one file that needs topping up

`alternative/interest-rate/usa/interest-rate.csv` is the risk-free curve. It
feeds Sharpe Ratio, Sortino Ratio, Alpha, Treynor Ratio and Probabilistic
Sharpe Ratio (`Lean/Common/Statistics/PortfolioStatistics.cs:293-312`).
Returns, drawdown and order counts do not depend on it.

It is the awkward one for two reasons:

1. **Missing it fails silently.** `InterestRateProvider` falls back to a flat
   1% (`Lean/Common/Data/InterestRateProvider.cs:40`) rather than erroring, so
   a folder without it still produces a plausible-looking Sharpe.
2. **LEAN's copy stops at `2023-07-27,5.5`** and upstream has not touched it
   since 2023-08-19, so `git pull` will not fix it. Past that date
   `GetInterestRate` carries the last rate forward forever
   (`InterestRateProvider.cs:68-78`).

`convert` therefore appends the later FOMC changes from the version-controlled
`leandata/reference/interest-rate-usa.csv`. Only rows dated after LEAN's own
last entry are added, so it is idempotent and never contradicts the bundled
data. See `leandata/reference/README.md` for the source and for why the
column is the upper bound of the federal funds target range.

Measured effect, the same NVDA 200-day run with and without the top-up:

| Statistic | topped up | LEAN's 2023 file |
|---|---|---|
| End Equity / CAGR / Drawdown / Orders | identical | identical |
| Sharpe Ratio | 1.358 | 1.352 |
| Sortino Ratio | 1.534 | 1.527 |
| Probabilistic Sharpe Ratio | 66.024% | 65.067% |

Only two of eleven years are affected there; a backtest concentrated in 2024
onwards would move considerably more.

**To extend it further**, add one row per FOMC change to
`leandata/reference/interest-rate-usa.csv`: the effective date, then the new
upper bound of the target range, then re-run `convert`. Edit the patch rather
than `data/`'s copy: the patch is the one that carries a source and a
rationale, and it is what a fresh checkout replays from. (Direct edits to
`data/` do survive later `convert` runs — only `--refresh-overlay` replaces
them — but nothing records where the numbers came from.)

The curve is a step function, so holding the last rate past its date is
correct, not stale. `check` and `convert` only warn when the last change is
more than 400 days before the data ends — the FOMC meets eight times a year,
so a longer hold suggests a missed change rather than a genuine one.

### Commands

```
convert    download a symbol and write it into a data folder
check      report what a data folder holds and whether it is complete
verify     diff a generated symbol against another data folder, row by row
snapshot   save a source's raw output as a replayable CSV pair
```

```
uv run --group data convert_data.py convert --source yfinance --ticker QQQ \
    --exchange-code Q --start 2010-01-01 --end 2024-12-31
uv run convert_data.py check
uv run convert_data.py verify --ticker SPY --against ../Lean/Data
uv run --group data convert_data.py snapshot --ticker SPY \
    --start 1998-01-02 --end 2021-03-31
```

`--exchange-code` sets the third column of the map file: `Q` for NASDAQ, `N`
for NYSE, `P` for NYSE Arca. It defaults to `P`, which suits most ETFs but
not all of them — QQQ is NASDAQ-listed, as `Lean/Data/equity/usa/map_files/`
will confirm for anything LEAN already ships.

`verify` exits non-zero when it finds differences, so it composes in a script;
the other commands exit non-zero only on a real failure.

`data/` **accumulates** symbols rather than being replaced, so converting a
second ticker leaves the first in place and both share one copy of the
reference data.

`verify` is the tool to reach for when the numbers disagree: it prints a
per-row table of price-factor and reference-price differences, then the
largest close differences, so you can tell whether the cause is the bars, the
factor file or the calendar.

### Worked example: a stock LEAN does not ship

```
uv run --group data convert_data.py convert --source yfinance --ticker NVDA \
    --exchange-code Q --start 2015-01-02 --end 2026-08-07
uv run run_local.py "NVDA 200-day MA Entry/Exit" --data-folder data
```

`algorithms/stock_ma_entry_exit.py` and `stock_buy_and_hold.py` take the
ticker and the date window as parameters, so they run against any converted
symbol. Register a combination in `variants.py` and `run_local.py` picks it
up.

Measured over 2015-01-02 → 2026-08-07 on converted NVDA data:

| Statistic | Buy & Hold | 200d MA | 50d MA |
|---|---|---|---|
| End Equity | 4,607,555 | 2,846,077 | 429,848 |
| Compounding Annual Return | 69.637% | 62.739% | 38.277% |
| Drawdown | 66.300% | 49.200% | 37.900% |
| Sharpe Ratio | 1.335 | 1.358 | 0.925 |
| Total Orders | 1 | 31 | 187 |

NVDA also exercises the part of the adapter most likely to be wrong on a
stock: it split 4-for-1 in 2021 and 10-for-1 in 2024. The converted bars show
the real price cliffs (2021-07-19 closes at 751.19, the next session at
186.12) and the factor file carries split factors of 0.025, then 0.1, then 1
— because LEAN wants raw prices plus a factor file, not Yahoo's pre-adjusted
series.

### Adding a data source

A new vendor is a new module in `leandata/sources/` plus one `register(...)`
line in `leandata/sources/__init__.py`. Nothing in `leandata/lean/`,
`convert_data.py` or `compare_datasets.py` changes.

`sources/csv_source.py` is that path demonstrated end to end — converting a
snapshot through it produces byte-identical LEAN files to converting the same
data through `yfinance`. Two things the adapter owns, both easy to get wrong:

- **Raw prices.** LEAN stores as-traded prices and reconstructs adjusted
  series from the factor file. Yahoo's OHLCV is already split-adjusted even
  with `auto_adjust=False`, so `YFinanceSource` un-adjusts the splits back
  out. A source that hands over adjusted prices would double-adjust.
- **Naive exchange-local time.** Convert the zone, then drop it. Localising a
  UTC-stamped feed without converting shifts pre-open bars onto the wrong day.

`SecurityHistory` validates both on construction, so an adapter that gets the
schema, ordering or price invariants wrong fails immediately rather than
producing quietly wrong files.

### Comparing a converted dataset against the bundled one

```
uv run compare_datasets.py
```

Runs `200-day MA Entry/Exit` over both `../Lean/Data` and `./data`, prints the
statistics side by side, and asserts the bundled column still reproduces the
anchor above. It refuses to run if the two datasets end on different days:
LEAN anchors `Adjusted` prices at the end of the factor file, so a longer
download rescales every adjusted price by a constant — returns and signals
survive that, share counts and fees do not.

Measured result for SPY, 1998-01-02 → 2021-03-31:

| Statistic | bundled | yfinance |
|---|---|---|
| End Equity | 31531.97 | 28947.21 |
| Compounding Annual Return | 5.551% | 5.127% |
| Drawdown | 24.200% | 26.600% |
| Sharpe Ratio | 0.236 | 0.205 |
| Total Orders | 145 | 155 |

The conversion is faithful; the two *feeds* are not. Both files cover the
same 5849 trading days and produce 96 factor rows on identical dates, and
AAPL's recent factor rows come out byte-identical to LEAN's. But SPY closes
disagree by 0.146% on average before 2009, tapering to 0.0000% from 2017
(QC's own bundled data has artifacts too — a high of 148.88 on 2000-12-11
against an open of 137.38). A 200-day crossover is knife-edge sensitive to
that: the two datasets disagree about being in or out of the market on 20 of
5345 days, which is exactly the 10 extra round trips LEAN reports.

Use the bundled data for anything that has to match a published number, and
converted data for symbols and periods LEAN does not ship. Do not treat the
two as cross-checks on each other — their inputs differ.

One harmless log line to expect: LEAN probes for
`equity/usa/hour/<ticker>.zip` and records a failed data request when a
converted folder has only daily bars. It falls back to daily and the run
completes normally.

## Tests

```
uv run pytest                              # offline
uv run --group data pytest -m network      # the one test that calls Yahoo
```

The algorithm tests drive each strategy through a fake `AlgorithmImports`
(`tests/lean_stubs.py`) and assert on the allocation decisions. They cover
decision logic only — fills, fees and real warm-up are LEAN's job and are
verified by the anchor run above.

The conversion tests add `tests/test_factor_cross_validation.py`, which
rebuilds SPY and AAPL from committed snapshots and checks them against the
bundled LEAN files. It skips itself if `../Lean` is absent.
