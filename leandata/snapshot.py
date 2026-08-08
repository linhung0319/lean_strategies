"""A plain-CSV serialisation of the canonical model.

Snapshots exist so that a download can be replayed without touching the
network: the test fixtures are snapshots, and ``sources.csv_source`` reads
them back. Because the round trip is lossless, converting from a snapshot
produces byte-identical LEAN files to converting from the live source.

Two files per security:

``<stem>.csv``          ``date,open,high,low,close,volume`` -- RAW prices
``<stem>_actions.csv``  ``ex_date,dividend,ratio`` -- 0 means "no leg"
"""

import csv
from pathlib import Path

import pandas as pd

from .model import BAR_COLUMNS, SecurityHistory, empty_dividends, empty_splits

BARS_HEADER = ["date", *BAR_COLUMNS]
ACTIONS_HEADER = ["ex_date", "dividend", "ratio"]
DATE_FORMAT = "%Y-%m-%d"


def snapshot_paths(directory: Path, stem: str) -> tuple[Path, Path]:
    directory = Path(directory)
    return directory / f"{stem}.csv", directory / f"{stem}_actions.csv"


def write_snapshot(history: SecurityHistory, directory: Path, stem: str) -> tuple[Path, Path]:
    bars_path, actions_path = snapshot_paths(directory, stem)
    bars_path.parent.mkdir(parents=True, exist_ok=True)

    with bars_path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream)
        writer.writerow(BARS_HEADER)
        for timestamp, row in zip(history.bars.index, history.bars[list(BAR_COLUMNS)].itertuples(index=False)):
            writer.writerow(
                [timestamp.strftime(DATE_FORMAT), row.open, row.high, row.low, row.close, row.volume]
            )

    actions = _merge_actions(history.dividends, history.splits)
    with actions_path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream)
        writer.writerow(ACTIONS_HEADER)
        for day, dividend, ratio in actions:
            writer.writerow([day.strftime(DATE_FORMAT), dividend, ratio])

    return bars_path, actions_path


def read_snapshot(bars_path: Path, actions_path: Path | None = None):
    """Return (bars, dividends, splits) frames matching the model contract."""
    bars_path = Path(bars_path)
    frame = pd.read_csv(bars_path)
    missing = [column for column in BARS_HEADER if column not in frame.columns]
    if missing:
        raise ValueError(f"{bars_path} is missing column(s): {', '.join(missing)}")
    index = pd.DatetimeIndex(pd.to_datetime(frame["date"]), name="time")
    bars = frame[list(BAR_COLUMNS)].astype("float64")
    bars.index = index

    dividends, splits = empty_dividends(), empty_splits()
    if actions_path is None:
        actions_path = bars_path.with_name(f"{bars_path.stem}_actions.csv")
    actions_path = Path(actions_path)
    if actions_path.exists():
        actions = pd.read_csv(actions_path)
        if len(actions):
            days = pd.DatetimeIndex(pd.to_datetime(actions["ex_date"]))
            paid = actions["dividend"].astype("float64") > 0
            if paid.any():
                dividends = pd.DataFrame(
                    {"amount": actions.loc[paid.to_numpy(), "dividend"].astype("float64").to_numpy()},
                    index=pd.DatetimeIndex(days[paid.to_numpy()], name="ex_date"),
                )
            split = actions["ratio"].astype("float64") > 0
            if split.any():
                splits = pd.DataFrame(
                    {"ratio": actions.loc[split.to_numpy(), "ratio"].astype("float64").to_numpy()},
                    index=pd.DatetimeIndex(days[split.to_numpy()], name="ex_date"),
                )
    return bars, dividends, splits


def _merge_actions(dividends: pd.DataFrame, splits: pd.DataFrame):
    days = sorted(set(dividends.index) | set(splits.index))
    for day in days:
        dividend = float(dividends["amount"].get(day, 0.0))
        ratio = float(splits["ratio"].get(day, 0.0))
        yield day, dividend, ratio
