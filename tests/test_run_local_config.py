"""run_local.py gained a data-folder override; these guard the old behaviour."""

import json
from pathlib import Path

import run_local
from run_local import build_config, strip_json_comments

JSONC = """
{
  // a comment
  "environment": "backtesting",
  "data-folder": "../../../Data/",  // another one
  "algorithm-location": "QuantConnect.Algorithm.CSharp.dll",
  "note": "a // inside a string must survive"
}
"""


def test_strip_json_comments_leaves_string_literals_alone():
    parsed = json.loads(strip_json_comments(JSONC))
    assert parsed["note"] == "a // inside a string must survive"
    assert parsed["data-folder"] == "../../../Data/"


def config_for(**options):
    return build_config("spy_ma_entry_exit", {"ma_period": 200}, Path("results/x"), **options)


def test_the_default_data_folder_is_still_the_lean_clone():
    # The regression that would silently break every existing invocation.
    config = config_for()
    assert config["data-folder"] == f"{run_local.LEAN_ROOT / 'Data'}/"


def test_an_override_replaces_the_data_folder_with_a_trailing_slash():
    config = config_for(data_folder=Path("C:/tmp/overlay"))
    assert config["data-folder"].endswith("/")
    assert config["data-folder"].rstrip("/").endswith("overlay")


def test_the_algorithm_and_parameters_are_wired_the_same_way():
    config = config_for()
    assert config["algorithm-type-name"] == "SpyMaEntryExit"
    assert config["algorithm-language"] == "Python"
    assert config["parameters"] == {"ma_period": "200"}  # LEAN wants strings
    assert config["python-venv"] == str(run_local.REPO_ROOT / ".venv")


def test_results_directory_is_suffixed_only_when_labelled(monkeypatch, tmp_path):
    recorded = {}

    def fake_run(command, **kwargs):
        recorded["cwd"] = kwargs.get("cwd")
        results_dir = json.loads(Path(command[3]).read_text(encoding="utf-8"))["results-destination-folder"]
        recorded["results_dir"] = Path(results_dir)
        recorded["results_dir"].mkdir(parents=True, exist_ok=True)
        (recorded["results_dir"] / "X-summary.json").write_text('{"statistics": {"Total Orders": "7"}}')

        class Completed:
            returncode = 0

        return Completed()

    monkeypatch.setattr(run_local.subprocess, "run", fake_run)
    monkeypatch.setattr(run_local, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(run_local, "python_dll", lambda: "python311.dll")

    statistics = run_local.run_variant("200-day MA Entry/Exit", label="yfinance")
    assert statistics == {"Total Orders": "7"}
    assert recorded["results_dir"] == tmp_path / "200_day_ma_entry_exit__yfinance"

    run_local.run_variant("200-day MA Entry/Exit")
    assert recorded["results_dir"] == tmp_path / "200_day_ma_entry_exit"


def test_a_relative_data_folder_is_made_absolute():
    # The launcher runs with cwd=Lean/Launcher/bin/Debug, so "--data-folder data"
    # would otherwise send LEAN looking three directories away from the repo.
    config = config_for(data_folder="data")
    folder = Path(config["data-folder"].rstrip("/"))
    assert folder.is_absolute()
    assert folder == (Path.cwd() / "data").resolve()
