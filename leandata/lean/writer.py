"""Writes a SecurityHistory into a LEAN data folder.

Source-agnostic by construction: this module imports the canonical model and
the LEAN format helpers, and nothing from ``leandata.sources``.
"""

import json
import os
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath

from ..errors import LeanDataError
from ..model import SecurityHistory, SymbolSpec
from .bars import BarFile, layout_for
from .factors import build_factor_rows, render_factor_file
from .mapfiles import build_map_rows, render_map_file


@dataclass(frozen=True)
class WriteReport:
    bar_files: tuple[Path, ...]
    factor_file: Path
    map_file: Path
    bar_count: int
    first_date: date
    last_date: date
    factor_rows: int
    warnings: tuple[str, ...]

    def describe(self) -> str:
        lines = [
            f"bars      {self.bar_count} rows  {self.first_date} -> {self.last_date}",
            f"factors   {self.factor_rows} rows  {self.factor_file}",
            f"map       {self.map_file}",
        ]
        lines.extend(f"data      {path}" for path in self.bar_files)
        lines.extend(f"warning   {message}" for message in self.warnings)
        return "\n".join(lines)


class LeanDataWriter:
    """Writes bar, factor and map files for one security into ``root``."""

    def __init__(self, root: Path, *, overwrite: bool = True) -> None:
        self.root = Path(root)
        self.overwrite = overwrite

    def write(self, history: SecurityHistory) -> WriteReport:
        layout = layout_for(history.resolution)

        bar_paths = []
        for entry in layout.entries(history.symbol, history.bars):
            bar_paths.append(self._write_zip(entry))

        factor_rows, warnings = build_factor_rows(
            history.bars, history.dividends, history.splits
        )
        factor_path = self._write_bytes(
            self._auxiliary_path(history.symbol, "factor_files"),
            render_factor_file(factor_rows),
        )
        map_path = self._write_bytes(
            self._auxiliary_path(history.symbol, "map_files"),
            render_map_file(build_map_rows(history.symbol, history.first_date)),
        )
        if history.provenance is not None:
            self._write_provenance(history)

        return WriteReport(
            bar_files=tuple(bar_paths),
            factor_file=factor_path,
            map_file=map_path,
            bar_count=len(history.bars),
            first_date=history.first_date,
            last_date=history.last_date,
            factor_rows=len(factor_rows),
            warnings=tuple(warnings),
        )

    def _auxiliary_path(self, symbol: SymbolSpec, folder: str) -> Path:
        relative = PurePosixPath(symbol.security_type, symbol.market, folder, f"{symbol.key}.csv")
        return self.root / relative

    def _write_zip(self, entry: BarFile) -> Path:
        target = self.root / entry.zip_path
        self._guard(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write beside the target and rename, so an interrupted run never
        # leaves a truncated archive that LEAN will fail to open.
        temporary = target.with_name(target.name + ".tmp")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(entry.entry_name, entry.payload)
        os.replace(temporary, target)
        return target

    def _write_bytes(self, target: Path, payload: bytes) -> Path:
        self._guard(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        # Binary mode throughout: factor and map files are CRLF terminated and
        # text mode on Windows would turn every "\r\n" into "\r\r\n".
        temporary.write_bytes(payload)
        os.replace(temporary, target)
        return target

    def _write_provenance(self, history: SecurityHistory) -> None:
        target = (
            self.root
            / ".provenance"
            / f"{history.symbol.key}-{history.resolution.value}.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = history.provenance.to_dict()
        payload.update(
            {
                "ticker": history.symbol.ticker,
                "resolution": history.resolution.value,
                "first_date": history.first_date.isoformat(),
                "last_date": history.last_date.isoformat(),
                "bar_count": len(history.bars),
            }
        )
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _guard(self, target: Path) -> None:
        if target.exists() and not self.overwrite:
            raise LeanDataError(f"{target} already exists and overwrite=False")
