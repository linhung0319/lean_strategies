"""Adapter over a snapshot CSV pair.

This is the second adapter, and it exists to keep the first one honest: if
adding a source really is "one module plus one register call", then this
module is the proof. It also lets the conversion be replayed offline and
gives the test suite a real registry path instead of a stand-in.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from ..model import Provenance, Resolution, SecurityHistory, SymbolSpec
from ..snapshot import read_snapshot
from .base import BaseSource, FetchRequest


class CsvSource(BaseSource):
    """Reads bars and corporate actions from a snapshot written by the CLI."""

    name: ClassVar[str] = "csv"

    def __init__(self, path: str | Path, actions: str | Path | None = None) -> None:
        self.path = Path(path)
        self.actions = Path(actions) if actions else None

    def supported_resolutions(self) -> frozenset[Resolution]:
        # A snapshot carries whatever resolution was captured; the file itself
        # does not say, so trust the caller.
        return frozenset(Resolution)

    def fetch(self, request: FetchRequest) -> SecurityHistory:
        if not self.path.exists():
            raise FileNotFoundError(f"snapshot not found at {self.path}")
        bars, dividends, splits = read_snapshot(self.path, self.actions)
        self._check_not_empty(bars, request)

        history = SecurityHistory(
            symbol=SymbolSpec(
                ticker=request.ticker,
                market=request.market,
                security_type=request.security_type,
                exchange_code=request.exchange_code,
            ),
            resolution=request.resolution,
            bars=bars,
            dividends=dividends,
            splits=splits,
            provenance=Provenance(
                source=self.name,
                fetched_at=datetime.now(timezone.utc),
                source_version="1",
                options={"path": str(self.path)},
            ),
        )
        return history.clip(request.start, request.end)
