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
DEFAULT_DATA_FOLDER = LEAN_ROOT / "Data"

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


def build_config(module_name, parameters, results_dir, data_folder=None):
    config = json.loads(strip_json_comments(BASE_CONFIG.read_text(encoding="utf-8")))
    config["environment"] = "backtesting"
    config["algorithm-language"] = "Python"
    config["algorithm-type-name"] = ALGORITHM_CLASSES[module_name]
    config["algorithm-location"] = str(REPO_ROOT / "algorithms" / f"{module_name}.py")
    # Absolute: the launcher runs with cwd=Lean/Launcher/bin/Debug, so a
    # relative --data-folder would resolve against that instead of here.
    folder = Path(data_folder).resolve() if data_folder else DEFAULT_DATA_FOLDER
    config["data-folder"] = f"{folder}/"
    config["results-destination-folder"] = str(results_dir)
    config["close-automatically"] = True
    # Lets LEAN resolve pandas/wrapt from this project's virtualenv.
    config["python-venv"] = str(REPO_ROOT / ".venv")
    config["parameters"] = {key: str(value) for key, value in parameters.items()}
    return config


def run_variant(name, data_folder=None, label=None):
    """Run one variant. `data_folder` overrides LEAN's bundled data, and
    `label` keeps the results of two datasets side by side."""
    module_name, parameters = VARIANTS[name]
    results_dir = RESULTS_ROOT / (slug(name) if label is None else f"{slug(name)}__{label}")
    results_dir.mkdir(parents=True, exist_ok=True)
    for stale in results_dir.glob("*-summary.json"):
        stale.unlink()

    config = build_config(module_name, parameters, results_dir, data_folder)
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
    parser.add_argument(
        "--data-folder",
        help="override LEAN's data-folder, e.g. a folder built by convert_data.py",
    )
    parser.add_argument("--label", help="suffix for results/<slug>__<label>/")
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

    if args.data_folder:
        # A data folder missing market-hours or symbol-properties fails deep
        # into LEAN's startup with an unhelpful stack trace; catch it here.
        from leandata.lean.overlay import missing_reference_data

        missing = missing_reference_data(args.data_folder)
        if missing:
            print(
                f"{args.data_folder} is missing reference data: "
                f"{', '.join(str(path) for path in missing)}. "
                f"Run 'uv run convert_data.py convert ...' or copy it from {DEFAULT_DATA_FOLDER}.",
                file=sys.stderr,
            )
            return 1

    results = {}
    for name in names:
        print(f"\n=== {name} ===", flush=True)
        results[name] = run_variant(name, data_folder=args.data_folder, label=args.label)

    width = max(len(name) for name in results)
    print(f"\n{'Variant'.ljust(width)}  " + "  ".join(REPORTED))
    for name, statistics in results.items():
        cells = [str(statistics.get(key, "-")) for key in REPORTED]
        print(f"{name.ljust(width)}  " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
