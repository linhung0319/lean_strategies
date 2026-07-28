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
