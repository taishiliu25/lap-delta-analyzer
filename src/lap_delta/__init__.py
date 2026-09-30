"""Lap Delta Analyzer — distance-domain lap time delta analysis for motorsport telemetry.

Import the pure-analysis layer eagerly (only needs numpy/scipy/pandas). Adapters
(``fastf1``) and visualization (``plotly``) are imported explicitly by callers so
that importing this package stays cheap and dependency-light.
"""

from __future__ import annotations

from . import align, catalog, coaching, corners, laps, minisectors, sectors, units
from .schema import (
    CANONICAL_COLUMNS,
    Lap,
    LapMeta,
    LapSet,
    SchemaError,
    format_laptime,
    validate_schema,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "Lap",
    "LapMeta",
    "LapSet",
    "SchemaError",
    "format_laptime",
    "validate_schema",
    "align",
    "catalog",
    "coaching",
    "corners",
    "laps",
    "minisectors",
    "sectors",
    "units",
]

__version__ = "0.1.0"
