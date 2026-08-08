"""Run one strategy against two LEAN data folders and diff the results.

The point is to check a converted dataset against LEAN's bundled SPY, which
is the only reference with known-good numbers. Usage:

    uv run compare_datasets.py
    uv run compare_datasets.py --variant "200-day MA Entry/Exit" \\
        --dataset bundled=../Lean/Data --dataset yfinance=data
"""

import argparse
import sys
from pathlib import Path

import run_local
from leandata.lean import overlay
from run_local import REPORTED, RESULTS_ROOT, run_variant
from variants import VARIANTS

DEFAULT_VARIANT = "200-day MA Entry/Exit"
REPO_ROOT = Path(__file__).resolve().parent

# From README.md: the numbers the bundled data has to keep reproducing. If
# this side drifts, the comparison says nothing and you want to know first.
ANCHOR = {
    "End Equity": "31531.97",
    "Compounding Annual Return": "5.551%",
    "Drawdown": "24.200%",
    "Sharpe Ratio": "0.236",
    "Total Orders": "145",
    "Total Fees": "$145.00",
}

COMPARED = [*REPORTED, "Net Profit", "Total Fees"]
# Share counts scale with the price level and volume comes straight from the
# vendor, so these differ by construction rather than by error.
INFORMATIONAL = [
    "Total Fees",
    "Estimated Strategy Capacity",
    "Lowest Capacity Asset",
    "Portfolio Turnover",
]


def parse_dataset(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected label=path, got {text!r}")
    label, path = text.split("=", 1)
    return label.strip(), Path(path.strip())


def preflight(datasets, ticker: str, allow_range_mismatch: bool) -> int:
    print("=== datasets ===")
    ends = {}
    for label, path in datasets:
        info = overlay.describe(path)
        print(f"\n[{label}]")
        print(info.describe())
        if info.missing_reference:
            print(f"  ERROR reference data missing; LEAN would fail mid-run", file=sys.stderr)
            return 1
        match = [entry for entry in info.series if entry.ticker == ticker.upper()]
        if not match:
            print(f"  ERROR no daily {ticker} data in {path}", file=sys.stderr)
            return 1
        ends[label] = match[0].last_date

    if len(set(ends.values())) > 1:
        message = (
            f"\nthe datasets end on different days: "
            + ", ".join(f"{label} {day}" for label, day in ends.items())
            + "\nLEAN anchors Adjusted prices at the end of the factor file, so a longer\n"
            "dataset rescales every adjusted price by a constant. Returns and signals\n"
            "survive that, but share counts, fees and end equity do not. Re-convert with\n"
            f"--end {min(ends.values())}, or pass --allow-range-mismatch to proceed anyway."
        )
        if not allow_range_mismatch:
            print(message, file=sys.stderr)
            return 1
        print(message.replace("Re-convert", "Proceeding anyway; would normally re-convert"))
    return 0


def compare_bars(datasets, ticker: str) -> None:
    if len(datasets) != 2:
        return
    (left_label, left_path), (right_label, right_path) = datasets
    left = {row[0]: row for row in overlay.read_daily_csv(_daily_zip(left_path, ticker))}
    right = {row[0]: row for row in overlay.read_daily_csv(_daily_zip(right_path, ticker))}

    print(f"\n=== input data: {left_label} vs {right_label} ===")
    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    print(f"  trading days   {len(left)} vs {len(right)}; "
          f"only in {left_label}: {len(only_left)}, only in {right_label}: {len(only_right)}")

    shared = sorted(set(left) & set(right))
    if not shared:
        return
    relative = [abs(left[day][4] - right[day][4]) / right[day][4] for day in shared]
    worst = max(zip(relative, shared))
    material = sum(1 for value in relative if value > 0.001)
    print(f"  close          mean |delta| {sum(relative) / len(relative) * 100:.4f}%, "
          f"max {worst[0] * 100:.4f}% on {worst[1]}")
    print(f"  closes differing by more than 0.1%: {material} of {len(shared)}")

    left_factors = _read_factors(left_path, ticker)
    right_factors = _read_factors(right_path, ticker)
    shared_factors = sorted(set(left_factors) & set(right_factors))
    print(f"  factor rows    {len(left_factors)} vs {len(right_factors)}, "
          f"{len(shared_factors)} shared dates")
    if shared_factors:
        drift = max(abs(left_factors[day] - right_factors[day]) for day in shared_factors)
        print(f"  max |price factor delta| {drift:.3e}")


def report(results, variant: str) -> None:
    labels = list(results)
    width = max(len(key) for key in COMPARED + INFORMATIONAL)
    column = max(max(len(label) for label in labels), 16)

    print(f"\n=== {variant} ===")
    header = f"{'statistic'.ljust(width)}  " + "  ".join(label.rjust(column) for label in labels)
    print(header)
    print("-" * len(header))
    for key in COMPARED:
        cells = [str(results[label].get(key, "-")).rjust(column) for label in labels]
        print(f"{key.ljust(width)}  " + "  ".join(cells))

    print("\ninformational -- expected to differ (share counts and vendor volume)")
    for key in INFORMATIONAL:
        if not any(key in results[label] for label in labels):
            continue
        cells = [str(results[label].get(key, "-")).rjust(column) for label in labels]
        print(f"{key.ljust(width)}  " + "  ".join(cells))


def check_anchor(statistics) -> bool:
    print("\n=== anchor check (bundled data vs README) ===")
    ok = True
    for key, expected in ANCHOR.items():
        actual = statistics.get(key, "-")
        passed = str(actual) == expected
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {key:<28} expected {expected:>12}  got {actual:>12}")
    return ok


def _daily_zip(root: Path, ticker: str) -> Path:
    return root / "equity" / "usa" / "daily" / f"{ticker.lower()}.zip"


def _read_factors(root: Path, ticker: str) -> dict:
    path = root / "equity" / "usa" / "factor_files" / f"{ticker.lower()}.csv"
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_bytes().decode("ascii").splitlines():
        if line.strip():
            day, price_factor = line.split(",")[:2]
            rows[day] = float(price_factor)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default=DEFAULT_VARIANT, choices=list(VARIANTS))
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        type=parse_dataset,
        metavar="LABEL=PATH",
        help="repeatable; defaults to bundled=../Lean/Data and yfinance=./data",
    )
    parser.add_argument("--allow-range-mismatch", action="store_true")
    parser.add_argument("--skip-backtests", action="store_true", help="only diff the input data")
    args = parser.parse_args(argv)

    datasets = args.datasets or [
        ("bundled", run_local.DEFAULT_DATA_FOLDER),
        ("yfinance", REPO_ROOT / "data"),
    ]

    if preflight(datasets, args.ticker, args.allow_range_mismatch):
        return 1
    compare_bars(datasets, args.ticker)
    if args.skip_backtests:
        return 0

    if not run_local.LAUNCHER_DLL.exists():
        print(f"\nLEAN Launcher not built at {run_local.LAUNCHER_DLL}", file=sys.stderr)
        return 1

    results = {}
    for label, path in datasets:
        print(f"\n=== running {args.variant} on {label} ===", flush=True)
        results[label] = run_variant(args.variant, data_folder=path, label=label)
        print(f"    results in {RESULTS_ROOT / '<slug>__' }{label}")

    report(results, args.variant)

    if "bundled" in results and args.variant == DEFAULT_VARIANT:
        if not check_anchor(results["bundled"]):
            print("\nthe bundled run no longer matches the README anchor; "
                  "the comparison above is not trustworthy", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
