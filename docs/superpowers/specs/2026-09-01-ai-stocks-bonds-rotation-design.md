# Design: Replicate the QuantConnect AI Stocks/Bonds Rotation algorithm locally

## Goal

Run QuantConnect's `AIStocksBondsRotationAlgorithm` leaderboard example on the
local LEAN engine against data this repo downloads itself, so its statistics
can be set beside the published cloud figures.

The published run reports, over an undisclosed five-year window:

| Statistic | QuantConnect |
|---|---|
| Total Orders | 207 |
| Compounding Annual Return | 26.262% |
| Drawdown | 31.900% |
| Net Profit | 220.696% |
| Sharpe Ratio | 0.77 |
| Total Fees | $4043.42 |
| Portfolio Turnover | 4.28% |

The algorithm needs three kinds of market input: US equities, a crypto pair,
and three FRED macro series. Only the first is something this repo can already
convert. The other two each need a new path through `leandata`.

## Why this algorithm and not the futures one

The first candidate, `InverseVolatilityFuturesAlgorithm`, was rejected after
investigation. `add_future(ticker)` takes LEAN's defaults, which are
`DataNormalizationMode.BackwardsRatio` and `DataMappingMode.OpenInterest`
(`Lean/Common/Extensions.cs:3966-3977`, `UniverseSettings.cs:104`). That means
the indicator is fed a back-adjusted continuous series with no roll gaps, and
the roll date is chosen from per-contract open interest. yfinance publishes
neither: its `=F` tickers are a single unadjusted front-month quote per root,
with no contract chain and no open interest, and it has no VIX future at all.
Reproducing that algorithm would require fabricating a contract chain, which
changes what is being measured. Per-contract futures data is obtainable free
(Databento's signup credit, or Interactive Brokers), but that is a separate
project.

This algorithm's inputs are all genuinely obtainable, and two of the three
kinds come from the same origin QuantConnect itself uses: Bitfinex for the
crypto pair, and FRED for the macro series. Only the equities come from a
different vendor than QuantConnect's, and the README already measures that
difference at 0.0000% over the window in question.

## Decisions

- **Backtest window: 2021-09-01 to 2026-08-31.** This is what
  `set_start_date(self.end_date - timedelta(5*365))` means when run today.
  QuantConnect's window cannot be recovered from the published figures. The
  only date-like field, `Lowest Capacity Asset BND TRO5ZARLX6JP`, is a
  security identifier carrying no date, and 220.696% over five years gives
  exactly the reported 26.262% CAGR for any five-year window. The two runs are
  therefore not the same period, and the comparison is of character, not of
  matching numbers.
- **Fetch window: 2016-01-01 to 2026-08-31.** `lookback_years` defaults to 4,
  so the first `_rebalance` in September 2021 calls `self.history(...,
  timedelta(4*365))` and reaches back to September 2017. Fetching from 2016
  leaves margin.
- **BTCUSD comes from Bitfinex's own public API, not yfinance.** The algorithm
  asks for `Market.BITFINEX`. Bitfinex publishes free, unauthenticated daily
  candles (verified: HTTP 200, history reaching before 2016), so the local run
  uses prices from the same exchange QuantConnect does. Yahoo's `BTC-USD` is a
  multi-exchange composite and would introduce a difference that the decision
  tree could amplify.
- **FRED series come from `fredgraph.csv`**, free and unauthenticated
  (verified: HTTP 200 for `VIXCLS`, `T10Y3M`, `DFF`). This is the same origin
  QuantConnect's FRED dataset draws from.
- **New asset classes are handled by dispatching on `SymbolSpec.security_type`,
  not by splitting the canonical model.** `SecurityHistory` already carries
  `market` and `security_type` and already defaults dividends and splits to
  empty, so a crypto history is a valid one. What differs between asset
  classes is confined to a new `AssetProfile`. The alternative, parallel
  `EquityHistory` and `CryptoHistory` types, would rewrite `model.py`,
  `writer.py` and every existing test to accommodate one symbol.
- **FRED does not become a `SecurityHistory`.** It is one value per date, not
  OHLCV. It gets its own narrow type, `ValueSeries`, and its own writer.
- **FRED still goes through `leandata`**, rather than a standalone fetch
  script, so that `convert_data.py check` reports it, it carries provenance,
  and it is covered by tests. The README's central promise is that `check`
  tells you everything a data folder holds; three series outside that ledger
  would eventually be the cause of a silently wrong backtest.
- **`scikit-learn` goes in the default dependency group**, unlike `yfinance`.
  `run_local.py` points LEAN's `python-venv` at this `.venv`, and the
  algorithm imports sklearn at backtest time, so it has to be there.
- **The two new sources use stdlib `urllib.request`.** Both are a single GET
  returning CSV or JSON; adding an HTTP dependency to the shared venv is not
  warranted.

## Facts established from the LEAN source

These drive the design and were each confirmed by reading LEAN, not assumed.

| Fact | Source |
|---|---|
| Crypto daily zip is `btcusd_trade.zip`; equity daily is `spy.zip` with no suffix | `Lean/Common/Util/LeanData.cs:791-816` |
| Crypto prices are **not** scaled by 10000; equity prices are | `Lean/Common/Data/Market/TradeBar.cs:432-443` (`LineParseNoScale`) against `:381-383` (`useScaleFactor: true`) |
| Crypto volume is fractional | `Lean/Data/crypto/coinbase/daily/btcusd_trade.zip` carries `20141201 00:00,300,370,300,370,0.05655554` |
| Bitfinex data and exchange timezone are both UTC | `Lean/Data/market-hours/market-hours-database.json`, entry `Crypto-bitfinex-[*]` |
| `bitfinex,BTCUSD` is already in symbol-properties | `Lean/Data/symbol-properties/symbol-properties-database.csv` |
| `DailyPreciseEndTime` and `SeedInitialPrices` both exist | `Lean/Common/AlgorithmSettings.cs:156,182` |
| **There is no `Fred` type in this LEAN repo** | Searched; it lives in the separate `Lean.DataSource.FRED` repository |
| `TotalReturn` normalization is split-adjusted price plus accumulated dividends | `Lean/Common/Data/Auxiliary/CorporateFactorProvider.cs:62-65`, `Lean/Common/Extensions.cs:3850-3851` |

The last one matters: the algorithm requests `DataNormalizationMode.TOTAL_RETURN`
for its labels, which needs the dividend and split information the existing
factor-file generator already produces. No new work there.

## Data availability, as measured

| Input | Source | Verified |
|---|---|---|
| SPY, GLD, BND daily plus corporate actions | yfinance | 1506 rows 2020-09-01 to 2026-08-31; SPY 24 dividends, BND 72, GLD 0 |
| BTCUSD daily | `api-pub.bitfinex.com/v2/candles/trade:1D:tBTCUSD/hist` | HTTP 200, daily candles before 2016 |
| VIXCLS, T10Y3M, DFF | `fred.stlouisfed.org/graph/fredgraph.csv?id=` | HTTP 200 for all three |
| market-hours, symbol-properties | `../Lean/Data`, existing overlay | `Crypto-bitfinex-[*]` wildcard, `bitfinex,BTCUSD` row |

## Project structure

```
leandata/
  model.py              + ValueSeries
  lean/
    profiles.py         NEW  AssetProfile, EQUITY, CRYPTO, profile_for()
    formatting.py       + raw_price(), fractional_volume()
    bars.py             _render_rows and DailyBarLayout take a profile
    writer.py           writes factor/map only when the profile says to
    customdata.py       NEW  ValueSeries -> alternative/fred/<key>.csv
  sources/
    bitfinex.py         NEW  BitfinexSource -> SecurityHistory (crypto)
    fred.py             NEW  FredSource     -> ValueSeries
    __init__.py         + two register() lines
  cli.py                + --security-type/--market, + convert-series,
                          check reports crypto and FRED
algorithms/
  ai_stocks_bonds_rotation.py   NEW
tests/
  test_profiles.py                NEW
  test_crypto_lean_files.py       NEW
  test_value_series.py            NEW
  test_bitfinex_source.py         NEW
  test_fred_source.py             NEW
  test_ai_rotation.py             NEW
  bitfinex_fakes.py, fred_fakes.py  NEW
variants.py             + one entry
pyproject.toml          + scikit-learn in [project.dependencies]
```

## `lean/profiles.py`

```python
@dataclass(frozen=True)
class AssetProfile:
    security_type: str
    timezone: str                    # "America/New_York" | "UTC"
    price_scaled: bool               # equity True, crypto False
    volume_integral: bool            # equity True, crypto False
    zip_suffix: str                  # "" | "_trade"
    writes_corporate_actions: bool   # equity True, crypto False
```

Two instances, `EQUITY` and `CRYPTO`, and `profile_for(security_type)` raising
on anything else. Every asset-class difference the writer needs lives here, so
adding one later is a new instance rather than a new branch in four files.

`timezone` is on the profile because the existing `model.EXCHANGE_TIMEZONE`
constant hardcodes `America/New_York` and Bitfinex is UTC. `SecurityHistory`
validation only requires tz-naive timestamps, so nothing there changes; the
profile tells the source what to convert to, and tells `check` what to report.

## `model.ValueSeries`

```python
@dataclass(frozen=True)
class ValueSeries:
    key: str                     # "vixcls"
    values: pd.Series            # DatetimeIndex (tz-naive) -> float64
    provenance: Provenance | None = None
```

Validation: `DatetimeIndex`, tz-naive, strictly increasing, unique, all finite.

**It must not require positive values.** Measured over the fetch window,
`T10Y3M` ranges from -1.89 to 2.27, is negative on 738 rows and exactly 0 on
11. Reusing `_validate_events`, which demands `> 0`, would reject real data on
its first run.

## `sources/bitfinex.py`

`GET https://api-pub.bitfinex.com/v2/candles/trade:1D:tBTCUSD/hist`
with `start`, `end` as millisecond epochs, plus `limit` and `sort=1`, paging
forward until the window is covered.

The response row order is:

```
[MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]
```

**Close is third, before high and low.** This was confirmed empirically, not
read off the docs: over 3889 real rows, reading the columns as
`[MTS, O, C, H, L, V]` satisfies LEAN's OHLC invariants on every row, while
reading them as `[MTS, O, H, L, C, V]` violates them on 3879. Validation would
not have caught this on its own — a swapped pair still satisfies
`high >= max(open, close)` on a day the two are near each other, which is why
`test_bitfinex_source.py` asserts the field mapping explicitly against a fixed
fake response rather than relying on the model's checks.

One request with `limit=10000` covers the whole fetch window (3889 rows,
2016-01-01 to 2026-08-31, already ascending under `sort=1`), so paging exists
only as a guard, not as a routine code path.

Timestamps are UTC midnight, converted to tz-naive UTC rather than to New York.
Dividends and splits stay empty.

## `sources/fred.py`

`GET https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>&cosd=<start>&coed=<end>`

Two columns, `observation_date` and the series id, LF terminated. A missing
observation is written as an **empty field**, not as the `.` that FRED's older
text format uses: `fredgraph.csv` emits `2016-01-18,` for a market holiday.
Measured over 2016-01-01 to 2026-08-31: `VIXCLS` has 69 empty rows out of
2780, `T10Y3M` has 115 out of 2781, `DFF` has none out of 3893 (it publishes
on weekends too). Empty rows are dropped rather than filled, because the
algorithm's own `.dropna()` expects gaps and filling them would invent
observations.

The same measurement confirms why `ValueSeries` must accept non-positive
values rather than reuse `_validate_events`: `T10Y3M` ranges from -1.89 to
2.27, is negative on 738 of those rows, and is exactly 0 on 11 of them.

## `algorithms/ai_stocks_bonds_rotation.py`

The QuantConnect source is kept verbatim except for two changes.

**1. A local `Fred` fallback.** This LEAN repo has no `Fred` type. The file
defines one only if the name is absent, so the same file still runs unmodified
when pasted into a QuantConnect project:

```python
try:
    Fred
except NameError:
    class Fred(PythonData):
        def get_source(self, config, date, is_live_mode):
            path = os.path.join(
                Globals.data_folder, "alternative", "fred",
                f"{config.symbol.value.lower()}.csv")
            return SubscriptionDataSource(path, SubscriptionTransportMedium.LOCAL_FILE)

        def reader(self, config, line, date, is_live_mode):
            ...   # "YYYYMMDD,value" -> instance with .time and .value
```

This preserves the README's standing promise that an `algorithms/*.py` file
can be pasted into a QuantConnect project and run.

**2. Fixed dates.** `set_start_date(2021, 9, 1)` and `set_end_date(2026, 8, 31)`
replace the relative start, matching how every other algorithm in this repo is
written. The README already documents deleting `set_end_date` before pasting
into QuantConnect.

Everything else stays as written: `daily_precise_end_time`,
`seed_initial_prices`, `DecisionTreeRegressor(max_depth=12, random_state=1)`,
`StandardScaler`, the monthly schedule, the 1.5x gross weighting and the
Bitcoin weight cap. The point of the exercise is that the decision logic is
identical and only the data underneath differs.

## Testing

- `test_profiles.py` covers the crypto zip name, unscaled prices, fractional
  volume, and the absence of factor and map files; equity behaviour unchanged.
- `test_crypto_lean_files.py` checks that a written crypto zip has the same
  shape as `Lean/Data/crypto/coinbase/daily/btcusd_trade.zip`. It skips if
  `../Lean` is absent, following `test_factor_cross_validation.py`.
- `test_value_series.py` covers negative and zero values being accepted, and
  unsorted, duplicated or non-finite input being rejected.
- `test_bitfinex_source.py` covers the `[MTS, OPEN, CLOSE, HIGH, LOW, VOL]`
  field order, paging, and UTC handling. Fake responses live in
  `bitfinex_fakes.py`, matching how `yfinance_fakes.py` is used today.
- `test_fred_source.py` covers `.` rows being dropped and negative values
  being preserved.
- `test_ai_rotation.py` drives the decision logic through `tests/lean_stubs.py`:
  weights sum to 1.5, the Bitcoin cap redistributes to equities, and a
  negative prediction drops the symbol.
- Two `network`-marked tests, one per new source.

**The existing test suite must pass untouched.** Equity output being
byte-identical before and after is this design's acceptance condition.

## Verification

1. `uv run pytest` gives a green offline suite, including the pre-existing
   tests.
2. `uv run convert_data.py check` reports the three equities, BTCUSD and the
   three FRED series.
3. `uv run run_local.py "AI Stocks/Bonds Rotation" --data-folder data`
   completes and produces statistics.
4. The result table is set beside the QuantConnect figures with the known
   divergences stated.

## Known divergences from the QuantConnect run

| Item | Expected impact |
|---|---|
| Different five-year window | **Large.** Their end date is unrecoverable, so the numbers are not directly comparable |
| Daily bars against LEAN's default minute resolution | Medium; changes when scheduled-event orders fill |
| Local `Fred` implementation | Small; the underlying FRED data is the same |
| sklearn version differences | Small; could change decision-tree tie-breaking |
| Raw Bitfinex API against QuantConnect's cleaned Bitfinex data | Small; same exchange |
| Yahoo against QuantConnect for SPY, GLD, BND | Small; the README measures a 0.0000% close difference from 2017, and the whole window is after 2017 |

A caveat worth stating plainly: `DecisionTreeRegressor(max_depth=12)` fitted on
roughly a thousand samples is far more sensitive to small input differences
than a moving-average crossover. The README already records that a 0.146%
close difference produced ten extra round trips in `spy_ma_entry_exit`. A tree
can respond more sharply than that. Closeness of inputs does not guarantee
closeness of outputs, and a large divergence here is not by itself evidence of
a conversion bug.

## Out of scope

- Minute or hour resolution for any asset class.
- Futures of any kind, and the `InverseVolatilityFuturesAlgorithm` replication.
- Sweeping multiple backtest end dates to locate QuantConnect's window. Worth
  doing later; it is a separate piece of work resting on this one.
- Any change to the existing equity conversion path.
