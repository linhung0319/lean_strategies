"""LEAN's on-disk format: bar zips, factor files, map files, data folders.

Every module here consumes the canonical model and produces bytes. None of
them import ``leandata.sources`` -- that separation is what keeps a new data
source from rippling through the writer.
"""

from .bars import BAR_LAYOUTS, BarFile, BarLayout, DailyBarLayout, layout_for
from .factors import FactorRow, build_factor_rows, render_factor_file
from .mapfiles import MapRow, build_map_rows, render_map_file
from .overlay import (
    MIRRORED,
    describe,
    ensure_overlay,
    extend_interest_rates,
    interest_rate_range,
    missing_reference_data,
    read_daily_csv,
    stale_rate_warning,
)
from .writer import LeanDataWriter, WriteReport

__all__ = [
    "BAR_LAYOUTS",
    "MIRRORED",
    "BarFile",
    "BarLayout",
    "DailyBarLayout",
    "FactorRow",
    "LeanDataWriter",
    "MapRow",
    "WriteReport",
    "build_factor_rows",
    "build_map_rows",
    "describe",
    "ensure_overlay",
    "extend_interest_rates",
    "interest_rate_range",
    "layout_for",
    "missing_reference_data",
    "read_daily_csv",
    "stale_rate_warning",
    "render_factor_file",
    "render_map_file",
]
