"""Translating one ticker spelling into another.

Three spellings of the same company show up in this pipeline and none of them
agree on class shares:

=========  =========  =======================================================
index      ``BRK.B``  what the index provider publishes
Yahoo      ``BRK-B``  what the vendor's API answers to
LEAN       ``BRKB``   what a data folder can hold
=========  =========  =======================================================

There is a fourth mismatch that is not about spelling at all: a company that
has been renamed is in the roster under its old ticker and at the vendor
under its new one. ``load_renames`` reads the version-controlled table that
resolves those; see ``leandata/reference/README.md`` for what qualifies as
one.

LEAN is the constrained one. A ``SecurityIdentifier`` may not contain a space
or a pipe (``SecurityIdentifier.cs:44``) and LEAN's own bundled SPY universe
file spells Berkshire's SID ``BRKB R735QTJ8XC9X`` even though its ticker
column says ``BRK.B``. Rather than reproduce that split personality -- which
would need a ticker-to-SID exception everywhere a symbol is written -- this
pipeline drops the separator once, on the way in, and is consistent about
``BRKB`` from then on. The cost is cosmetic: backtest output says ``BRKB``.
"""

import csv
import re
from pathlib import Path
from typing import Final

# Everything a provider might use to separate a share class from its root.
_CLASS_SEPARATORS: Final = re.compile(r"[.\-/]")

RENAMES_FILE: Final = Path(__file__).resolve().parent / "reference" / "ticker-renames-usa.csv"


def lean_ticker(index_ticker: str) -> str:
    """``BRK.B`` -> ``BRKB``: the spelling used for LEAN files and SIDs."""
    return _CLASS_SEPARATORS.sub("", index_ticker.strip().upper())


def yahoo_ticker(index_ticker: str) -> str:
    """``BRK.B`` -> ``BRK-B``: the spelling Yahoo's API answers to."""
    return _CLASS_SEPARATORS.sub("-", index_ticker.strip().upper())


def load_renames(path: Path = RENAMES_FILE) -> dict[str, str]:
    """``{old ticker: new ticker}`` from the version-controlled rename table.

    Missing file means no renames rather than an error: the table is a
    correction, and a build without it is degraded but not wrong.
    """
    path = Path(path)
    if not path.exists():
        return {}
    renames: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            old = (row.get("old") or "").strip().upper()
            new = (row.get("new") or "").strip().upper()
            if old and new:
                renames[old] = new
    return renames


def resolve(ticker: str, renames: dict[str, str] | None = None) -> str:
    """Follow a rename chain to the ticker the company trades under today."""
    renames = renames or {}
    seen = {ticker}
    while ticker in renames and renames[ticker] not in seen:
        ticker = renames[ticker]
        seen.add(ticker)
    return ticker
