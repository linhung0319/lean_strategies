"""Command line for the data conversion pipeline.

    convert   download a symbol and write it into a LEAN data folder
    universe  download a whole index and write an ETF constituents universe
    verify    diff a generated symbol against another LEAN data folder
    check     report whether a data folder is complete enough for a run
    snapshot  save a source's raw output as a replayable CSV pair
"""

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

from .constituents import DEFAULT_PAUSE_SECONDS, build_constituent_universe
from .errors import LeanDataError
from .lean import overlay
from .lean.writer import LeanDataWriter
from .model import Resolution
from .snapshot import write_snapshot
from .sources import available, available_membership, get_membership_source, get_source
from .sources.base import FetchRequest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data"
# A constituents build re-downloads the ETF over its own window, which would
# overwrite the SPY that compare_datasets.py pins to 2021-03-31. Its own
# folder keeps that comparison reproducible.
DEFAULT_CONSTITUENTS_OUT = REPO_ROOT / "data-sp500"
DEFAULT_LEAN_DATA = REPO_ROOT.parent / "Lean" / "Data"
DEFAULT_FIXTURES = REPO_ROOT / "tests" / "fixtures"


def parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def parse_option(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected key=value, got {text!r}")
    key, value = text.split("=", 1)
    return key.strip(), value.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="convert_data.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_fetch_arguments(target):
        target.add_argument("--source", default="yfinance", choices=available())
        target.add_argument("--ticker", required=True)
        target.add_argument("--resolution", default="daily", choices=[item.value for item in Resolution])
        target.add_argument("--start", type=parse_date, help="inclusive, YYYY-MM-DD")
        target.add_argument("--end", type=parse_date, help="inclusive, YYYY-MM-DD")
        target.add_argument(
            "--source-option",
            dest="source_options",
            action="append",
            type=parse_option,
            default=[],
            metavar="KEY=VALUE",
            help="passed to the source constructor; repeatable",
        )

    convert = commands.add_parser("convert", help="download and write LEAN data files")
    add_fetch_arguments(convert)
    convert.add_argument("--out", type=Path, default=DEFAULT_OUT)
    convert.add_argument("--lean-data", type=Path, default=DEFAULT_LEAN_DATA)
    convert.add_argument("--exchange-code", default="P", help="map file column 3: P, Q or N")
    convert.add_argument("--market", default="usa")
    convert.add_argument("--no-overlay", action="store_true", help="skip mirroring LEAN's reference data")
    convert.add_argument("--refresh-overlay", action="store_true")

    universe = commands.add_parser(
        "universe", help="download an index and write an ETF constituents universe"
    )
    universe.add_argument("--etf", default="SPY", help="the ETF the universe files are named after")
    universe.add_argument("--source", default="yfinance", choices=available())
    universe.add_argument("--membership", default="sp500-github", choices=available_membership())
    universe.add_argument(
        "--membership-file",
        type=Path,
        help="replay a saved membership CSV instead of downloading; pins the roster",
    )
    universe.add_argument("--start", type=parse_date, required=True, help="inclusive, YYYY-MM-DD")
    universe.add_argument("--end", type=parse_date, required=True, help="inclusive, YYYY-MM-DD")
    universe.add_argument("--out", type=Path, default=DEFAULT_CONSTITUENTS_OUT)
    universe.add_argument("--lean-data", type=Path, default=DEFAULT_LEAN_DATA)
    universe.add_argument(
        "--cache",
        type=Path,
        help="snapshot cache directory; defaults to <out>/.cache",
    )
    universe.add_argument("--market", default="usa")
    universe.add_argument("--etf-exchange-code", default="P")
    universe.add_argument("--exchange-code", default="Q", help="map file column 3 for constituents")
    universe.add_argument(
        "--pause", type=float, default=DEFAULT_PAUSE_SECONDS,
        help="seconds to wait between vendor requests; cached symbols never wait",
    )
    universe.add_argument("--refresh", action="store_true", help="ignore the snapshot cache")
    universe.add_argument("--no-overlay", action="store_true")
    universe.add_argument("--refresh-overlay", action="store_true")

    verify = commands.add_parser("verify", help="diff generated files against another data folder")
    verify.add_argument("--ticker", required=True)
    verify.add_argument("--out", type=Path, default=DEFAULT_OUT)
    verify.add_argument("--against", type=Path, default=DEFAULT_LEAN_DATA)
    verify.add_argument("--market", default="usa")
    verify.add_argument("--rows", type=int, default=12, help="how many differing rows to print")

    check = commands.add_parser("check", help="report on a data folder's completeness")
    check.add_argument("--out", type=Path, default=DEFAULT_OUT)

    snapshot = commands.add_parser("snapshot", help="save a source's output as replayable CSV")
    add_fetch_arguments(snapshot)
    snapshot.add_argument("--out", type=Path, default=DEFAULT_FIXTURES)
    snapshot.add_argument("--stem", help="file stem; defaults to <ticker>_<source>_<resolution>")

    return parser


def _fetch(args):
    options = dict(args.source_options)
    source = get_source(args.source, **options)
    request = FetchRequest(
        ticker=args.ticker,
        resolution=Resolution(args.resolution),
        start=args.start,
        end=args.end,
        market=getattr(args, "market", "usa"),
        exchange_code=getattr(args, "exchange_code", "P"),
    )
    history = source.fetch(request)
    for warning in getattr(source, "warnings", []):
        print(f"warning   {warning}")
    return history


def command_convert(args) -> int:
    history = _fetch(args)
    report = LeanDataWriter(args.out).write(history)
    print(report.describe())

    if not args.no_overlay:
        copied = overlay.ensure_overlay(args.out, args.lean_data, refresh=args.refresh_overlay)
        for path in copied:
            print(f"mirrored  {path}")
        for row in overlay.extend_interest_rates(args.out):
            print(f"rate      appended {row}")
        rates = overlay.interest_rate_range(args.out)
        if rates:
            print(f"rate      curve {rates[0]} -> {rates[1]}, held flat after that")
            print(overlay.stale_rate_warning(rates[1], history.last_date) or "", end="")
    missing = overlay.missing_reference_data(args.out)
    if missing:
        print(f"WARNING   {args.out} is missing reference data: "
              f"{', '.join(str(path) for path in missing)}", file=sys.stderr)
        return 1
    return 0


def command_universe(args) -> int:
    membership_options = {"path": args.membership_file} if args.membership_file else {}
    membership_source = get_membership_source(args.membership, **membership_options)
    membership = membership_source.fetch_membership()

    cache = args.cache or (args.out / ".cache")
    membership_file = None
    if not args.membership_file:
        # Pin the roster next to the data it produced, so the build can be
        # replayed after upstream rewrites the file in place.
        membership_file = membership_source.save(cache / "membership" / f"{args.membership}.csv")
        print(f"pinned    {membership_file}")

    report = build_constituent_universe(
        etf_ticker=args.etf,
        membership=membership,
        source=get_source(args.source),
        out=args.out,
        start=args.start,
        end=args.end,
        cache=cache,
        market=args.market,
        etf_exchange_code=args.etf_exchange_code,
        constituent_exchange_code=args.exchange_code,
        refresh=args.refresh,
        pause_seconds=args.pause,
    )
    report.membership_file = membership_file or args.membership_file
    print(report.describe())

    if not args.no_overlay:
        copied = overlay.ensure_overlay(args.out, args.lean_data, refresh=args.refresh_overlay)
        for path in copied:
            print(f"mirrored  {path}")
        for row in overlay.extend_interest_rates(args.out):
            print(f"rate      appended {row}")
        rates = overlay.interest_rate_range(args.out)
        if rates:
            print(f"rate      curve {rates[0]} -> {rates[1]}, held flat after that")
            print(overlay.stale_rate_warning(rates[1], args.end) or "", end="")

    missing = overlay.missing_reference_data(args.out)
    if missing:
        print(f"WARNING   {args.out} is missing reference data: "
              f"{', '.join(str(path) for path in missing)}", file=sys.stderr)
        return 1
    return 0


def command_check(args) -> int:
    info = overlay.describe(args.out)
    print(info.describe())
    rates = overlay.interest_rate_range(args.out)
    if rates:
        print(f"  risk-free rate curve {rates[0]} -> {rates[1]}, held flat after that")
        if info.series:
            print(overlay.stale_rate_warning(rates[1], max(e.last_date for e in info.series)) or "", end="")
    return 1 if info.missing_reference else 0


def command_snapshot(args) -> int:
    history = _fetch(args)
    stem = args.stem or f"{history.symbol.key}_{args.source}_{args.resolution}"
    bars_path, actions_path = write_snapshot(history, args.out, stem)
    print(f"bars      {len(history.bars)} rows  {history.first_date} -> {history.last_date}")
    print(f"wrote     {bars_path}")
    print(f"wrote     {actions_path}")
    return 0


def command_verify(args) -> int:
    key = args.ticker.lower()
    generated = args.out / "equity" / args.market
    bundled = args.against / "equity" / args.market

    status = 0
    status |= _verify_factor_file(generated / "factor_files" / f"{key}.csv",
                                  bundled / "factor_files" / f"{key}.csv", args.rows)
    status |= _verify_daily_bars(generated / "daily" / f"{key}.zip",
                                 bundled / "daily" / f"{key}.zip", args.rows)
    return status


def _verify_factor_file(generated: Path, reference: Path, limit: int) -> int:
    print(f"\n=== factor file ===\n  generated {generated}\n  reference {reference}")
    if not generated.exists() or not reference.exists():
        print("  SKIP one side is missing")
        return 1
    if generated.read_bytes() == reference.read_bytes():
        print("  IDENTICAL (byte for byte)")
        return 0

    left = _read_factor_rows(generated)
    right = _read_factor_rows(reference)
    print(f"  rows      generated {len(left)}, reference {len(right)}")
    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    if only_left:
        print(f"  dates only in generated ({len(only_left)}): {_sample(only_left, limit)}")
    if only_right:
        print(f"  dates only in reference ({len(only_right)}): {_sample(only_right, limit)}")

    shared = sorted(set(left) & set(right))
    differing = [
        day for day in shared
        if left[day][0] != right[day][0] or left[day][1] != right[day][1] or left[day][2] != right[day][2]
    ]
    if not differing:
        print("  every shared row agrees")
    else:
        worst = max(abs(left[day][0] - right[day][0]) for day in differing)
        print(f"  differing rows {len(differing)} of {len(shared)}, max |price factor delta| {worst:.3e}")
        print(f"  {'date':<10} {'pf gen':>12} {'pf ref':>12} {'delta':>11}   "
              f"{'sf gen':>10} {'sf ref':>10}   {'ref gen':>10} {'ref ref':>10}")
        for day in differing[:limit]:
            gen, ref = left[day], right[day]
            print(f"  {day:<10} {gen[0]:>12.7f} {ref[0]:>12.7f} {gen[0] - ref[0]:>11.2e}   "
                  f"{gen[1]:>10.8g} {ref[1]:>10.8g}   {gen[2]:>10.4f} {ref[2]:>10.4f}")
        if len(differing) > limit:
            print(f"  ... {len(differing) - limit} more")
    return 1


def _verify_daily_bars(generated: Path, reference: Path, limit: int) -> int:
    print(f"\n=== daily bars ===\n  generated {generated}\n  reference {reference}")
    if not generated.exists() or not reference.exists():
        print("  SKIP one side is missing")
        return 1

    left = {row[0]: row for row in overlay.read_daily_csv(generated)}
    right = {row[0]: row for row in overlay.read_daily_csv(reference)}
    print(f"  rows      generated {len(left)} ({min(left)} -> {max(left)}), "
          f"reference {len(right)} ({min(right)} -> {max(right)})")

    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    if only_left:
        print(f"  days only in generated ({len(only_left)}): {_sample(only_left, limit)}")
    if only_right:
        print(f"  days only in reference ({len(only_right)}): {_sample(only_right, limit)}")

    shared = sorted(set(left) & set(right))
    if not shared:
        print("  no overlapping days")
        return 1
    deltas = sorted(
        ((abs(left[day][4] - right[day][4]), day) for day in shared), reverse=True
    )
    mean = sum(delta for delta, _ in deltas) / len(deltas)
    print(f"  close     max |delta| {deltas[0][0]:.4f} on {deltas[0][1]}, mean {mean:.4f}")
    material = [day for delta, day in deltas if right[day][4] and delta / right[day][4] > 0.001]
    print(f"  closes differing by more than 0.1%: {len(material)}")
    for delta, day in deltas[:limit]:
        if delta == 0:
            break
        print(f"    {day}  generated {left[day][4]:>10.4f}  reference {right[day][4]:>10.4f}  delta {delta:>8.4f}")
    return 0 if not only_left and not only_right and deltas[0][0] == 0 else 1


def _read_factor_rows(path: Path) -> dict:
    rows = {}
    for line in path.read_bytes().decode("ascii").splitlines():
        if not line.strip():
            continue
        day, price_factor, split_factor, reference_price = line.split(",")[:4]
        rows[day] = (float(price_factor), float(split_factor), float(reference_price))
    return rows


def _sample(items, limit: int) -> str:
    shown = ", ".join(str(item) for item in items[:limit])
    return shown + (f", ... (+{len(items) - limit})" if len(items) > limit else "")


COMMANDS = {
    "convert": command_convert,
    "universe": command_universe,
    "verify": command_verify,
    "check": command_check,
    "snapshot": command_snapshot,
}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (LeanDataError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
