"""Distance-domain alignment and cumulative time delta (M3).

Two laps are never sampled at the same points in time or space, so we resample both
onto a common distance grid and compare cumulative lap time at each distance. The
delta at distance d is ``t_cmp(d) - t_ref(d)``; a rising curve means the comparison
lap is losing time. This is the standard "time delta vs distance" used in race
engineering, and it makes differing sample rates between sources irrelevant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import Lap

DEFAULT_GRID_POINTS = 1000

#: Channels interpolated onto the grid (everything except distance, which *is* the grid).
_INTERP_CHANNELS = ("time_s", "speed_kph", "throttle", "brake", "gear", "rpm", "x", "y")


def _samples(lap: Lap | pd.DataFrame) -> pd.DataFrame:
    return lap.samples if isinstance(lap, Lap) else lap


def make_distance_grid(*laps: Lap | pd.DataFrame, n: int = DEFAULT_GRID_POINTS) -> np.ndarray:
    """Common distance grid spanning the overlap of all laps (no extrapolation).

    Start = max of per-lap first distances; end = min of per-lap last distances.
    """
    dfs = [_samples(lap) for lap in laps]
    start = max(float(df["distance_m"].iloc[0]) for df in dfs)
    end = min(float(df["distance_m"].iloc[-1]) for df in dfs)
    if not end > start:
        raise ValueError("laps do not overlap in distance")
    return np.linspace(start, end, n)


def resample_to_grid(lap: Lap | pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Interpolate a lap's channels onto ``grid`` (indexed by distance).

    Distance is clamped monotonic and de-duplicated so ``np.interp`` sees a strictly
    increasing x-axis.
    """
    df = _samples(lap)
    dist = np.maximum.accumulate(df["distance_m"].to_numpy(dtype=float))
    dist_u, keep = np.unique(dist, return_index=True)

    out: dict[str, np.ndarray] = {"distance_m": np.asarray(grid, dtype=float)}
    for ch in _INTERP_CHANNELS:
        if ch in df.columns:
            values = df[ch].to_numpy(dtype=float)[keep]
            out[ch] = np.interp(grid, dist_u, values)
    return pd.DataFrame(out)


def _rescale_distance(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """Map a lap's distance onto ``[lo, hi]`` by fraction of lap completed."""
    d = np.maximum.accumulate(df["distance_m"].to_numpy(dtype=float))
    scaled = df.copy()
    scaled["distance_m"] = (d - d[0]) / (d[-1] - d[0]) * (hi - lo) + lo
    return scaled


def compute_delta(
    comparison: Lap | pd.DataFrame,
    reference: Lap | pd.DataFrame,
    grid: np.ndarray | None = None,
    n: int = DEFAULT_GRID_POINTS,
) -> pd.DataFrame:
    """Resample both laps onto a common grid and compute cumulative time delta.

    Laps are aligned by **fraction of lap completed** (both cover the same track), so
    equal grid positions compare the same point on track despite small telemetry
    distance differences — and the delta endpoint equals the lap-time gap. The grid is
    expressed in meters on the *reference* lap's distance scale.

    Returns a DataFrame with ``distance_m``, ``delta_s`` (cmp - ref), and each channel
    suffixed ``_cmp`` / ``_ref`` for overlay plotting.
    """
    ref_df = _samples(reference)
    rd = np.maximum.accumulate(ref_df["distance_m"].to_numpy(dtype=float))
    lo, hi = float(rd[0]), float(rd[-1])
    if grid is None:
        grid = np.linspace(lo, hi, n)

    ref_res = resample_to_grid(reference, grid)
    cmp_res = resample_to_grid(_rescale_distance(_samples(comparison), lo, hi), grid)

    out = pd.DataFrame({"distance_m": grid})
    for ch in _INTERP_CHANNELS:
        if ch in cmp_res.columns:
            out[f"{ch}_cmp"] = cmp_res[ch].to_numpy()
        if ch in ref_res.columns:
            out[f"{ch}_ref"] = ref_res[ch].to_numpy()

    out["delta_s"] = out["time_s_cmp"] - out["time_s_ref"]
    return out


def total_delta(delta_df: pd.DataFrame) -> float:
    """Cumulative time delta at the end of the lap (seconds; positive = slower)."""
    return float(delta_df["delta_s"].iloc[-1])
