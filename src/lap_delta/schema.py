"""Canonical telemetry schema shared by every data source and analysis module.

The whole point of this module is the *contract*: every adapter (FastF1 today,
iRacing tomorrow) must emit a :class:`Lap` whose ``samples`` DataFrame conforms to
:data:`CANONICAL_COLUMNS` with normalized units. Downstream code (laps, align,
corners, coaching, viz) only ever sees this schema and never a source-specific quirk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Canonical columns and their logical dtypes. Order is meaningful for display.
CANONICAL_COLUMNS: dict[str, str] = {
    "distance_m": "float",  # distance from start/finish, meters, monotonic 0 -> track length
    "time_s": "float",      # cumulative lap time from lap start, seconds, monotonic
    "speed_kph": "float",   # km/h
    "throttle": "float",    # 0..1
    "brake": "float",       # 0..1 (FastF1 supplies only {0, 1}; iRacing continuous)
    "gear": "int",          # selected gear
    "rpm": "float",         # engine RPM
    "x": "float",           # planar track position X (map)
    "y": "float",           # planar track position Y (map)
}

#: Columns that must never contain NaN (position may be missing on some sources/laps).
REQUIRED_NON_NULL = ("distance_m", "time_s", "speed_kph", "throttle", "brake")


class SchemaError(ValueError):
    """Raised when a Lap's samples violate the canonical schema."""


@dataclass
class LapMeta:
    """Per-lap metadata carried alongside the samples DataFrame."""

    source: str  # "fastf1" | "iracing"
    driver: str = ""
    team: str = ""
    year: int | None = None
    event: str = ""
    session: str = ""
    track: str = ""
    lap_number: int | None = None
    lap_time_s: float | None = None
    is_valid: bool = True
    is_out_lap: bool = False
    is_in_lap: bool = False
    is_pit: bool = False
    compound: str | None = None
    sample_count: int | None = None
    sample_rate_hz: float | None = None
    # Official timing sectors (seconds) — populated by sources that report them (FastF1).
    # Optional so other sources / analysis code never depend on their presence.
    sector1_s: float | None = None
    sector2_s: float | None = None
    sector3_s: float | None = None
    team_color: str | None = None  # hex color for driver-specific plotting, e.g. "#3671C6"

    @property
    def label(self) -> str:
        """Short human label, e.g. ``VER L12 (1:20.123)``."""
        parts = [self.driver or self.source]
        if self.lap_number is not None:
            parts.append(f"L{self.lap_number}")
        if self.lap_time_s:
            parts.append(f"({format_laptime(self.lap_time_s)})")
        return " ".join(parts)


@dataclass(eq=False)
class Lap:
    """A single lap: canonical samples DataFrame + metadata.

    ``eq=False`` keeps laps as identity objects (a Lap wraps an unhashable DataFrame),
    so they can live in sets / dict keys.
    """

    samples: pd.DataFrame
    meta: LapMeta

    def __post_init__(self) -> None:
        if self.meta.sample_count is None:
            self.meta.sample_count = len(self.samples)


#: A collection of laps (kept as a plain list for simplicity).
LapSet = list


def format_laptime(seconds: float | None) -> str:
    """Format seconds as ``M:SS.mmm`` (or ``--:--.---`` for missing)."""
    if seconds is None or not np.isfinite(seconds):
        return "--:--.---"
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes}:{rem:06.3f}"


def validate_schema(lap: Lap | pd.DataFrame, *, require_position: bool = False) -> bool:
    """Validate a Lap (or bare DataFrame) against the canonical schema.

    Checks presence of all canonical columns, absence of NaN in required columns,
    unit ranges (throttle/brake in [0, 1], non-negative speed), and that
    ``distance_m`` is monotonic non-decreasing (a small backward tolerance absorbs
    numerical noise). Raises :class:`SchemaError` on the first violation.
    """
    df = lap.samples if isinstance(lap, Lap) else lap
    if not isinstance(df, pd.DataFrame):
        raise SchemaError("samples must be a pandas DataFrame")

    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing canonical columns: {missing}")

    if len(df) < 2:
        raise SchemaError("a lap needs at least 2 samples")

    for col in REQUIRED_NON_NULL:
        if df[col].isna().any():
            raise SchemaError(f"NaN values in required column '{col}'")

    if require_position:
        for col in ("x", "y"):
            if df[col].isna().any():
                raise SchemaError(f"NaN values in position column '{col}'")

    throttle = df["throttle"].to_numpy(dtype=float)
    if (throttle < -0.01).any() or (throttle > 1.01).any():
        raise SchemaError("throttle must be within [0, 1]")

    brake = df["brake"].to_numpy(dtype=float)
    if (brake < -0.01).any() or (brake > 1.01).any():
        raise SchemaError("brake must be within [0, 1]")

    if (df["speed_kph"].to_numpy(dtype=float) < -0.1).any():
        raise SchemaError("speed_kph must be non-negative")

    dist = df["distance_m"].to_numpy(dtype=float)
    if np.any(np.diff(dist) < -1.0):  # allow <1 m backward blips from noise
        raise SchemaError("distance_m must be monotonic non-decreasing")

    return True
