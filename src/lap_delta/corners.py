"""Corner detection and per-corner metrics (M4).

FastF1 has no lateral-G channel, so corners are found as **speed minima** (apexes),
optionally reinforced by track **curvature** derived from the x/y trace. Each corner is
bounded by the midpoints between neighbouring apexes, and we extract the brake point,
apex (minimum) speed, throttle-on point, and - once a delta trace is available - the
time gained/lost through the corner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .schema import Lap


@dataclass
class Corner:
    number: int
    apex_dist: float
    apex_speed_kph: float
    entry_dist: float
    exit_dist: float
    brake_point_dist: float = float("nan")   # first braking before apex
    throttle_on_dist: float = float("nan")   # first sustained throttle after apex
    mean_curvature: float = float("nan")
    time_delta_s: float = float("nan")        # cmp - ref across [entry, exit]


def _samples(lap: Lap | pd.DataFrame) -> pd.DataFrame:
    return lap.samples if isinstance(lap, Lap) else lap


def curvature_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Local curvature |x'y'' - y'x''| / (x'^2 + y'^2)^1.5 along the path."""
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    num = np.abs(dx * ddy - dy * ddx)
    den = np.power(dx * dx + dy * dy, 1.5) + 1e-9
    return num / den


def segment_metrics(df: pd.DataFrame, entry: float, exit_: float) -> dict:
    """Brake point / apex speed / throttle-on within a distance window.

    Shared by :func:`detect_corners` and the coaching layer so a corner detected on
    the reference lap can be measured identically on the comparison lap.
    """
    seg = df[(df["distance_m"] >= entry) & (df["distance_m"] <= exit_)]
    if len(seg) < 2:
        return {"apex_dist": np.nan, "min_speed": np.nan,
                "brake_point": np.nan, "throttle_on": np.nan}

    apex_idx = seg["speed_kph"].idxmin()
    apex_dist = float(df.loc[apex_idx, "distance_m"])
    min_speed = float(df.loc[apex_idx, "speed_kph"])

    pre = seg[seg["distance_m"] <= apex_dist]
    braked = pre[pre["brake"] > 0.1]
    brake_point = float(braked["distance_m"].iloc[0]) if len(braked) else np.nan

    post = seg[seg["distance_m"] >= apex_dist]
    on_throttle = post[post["throttle"] > 0.5]
    throttle_on = float(on_throttle["distance_m"].iloc[0]) if len(on_throttle) else np.nan

    return {"apex_dist": apex_dist, "min_speed": min_speed,
            "brake_point": brake_point, "throttle_on": throttle_on}


def detect_corners(
    lap: Lap | pd.DataFrame,
    *,
    prominence_frac: float = 0.06,
    min_separation_frac: float = 0.02,
) -> list[Corner]:
    """Detect corners as prominent speed minima, bounded by midpoints between apexes.

    ``prominence_frac`` and ``min_separation_frac`` are fractions of the speed range and
    the lap length, exposed so per-track sensitivity can be tuned.
    """
    df = _samples(lap).reset_index(drop=True)
    dist = df["distance_m"].to_numpy(dtype=float)
    speed = df["speed_kph"].to_numpy(dtype=float)
    n = len(speed)
    if n < 5:
        return []

    speed_range = float(speed.max() - speed.min())
    if speed_range <= 0:
        return []

    prominence = prominence_frac * speed_range
    min_sep_samples = max(3, int(min_separation_frac * n))

    minima, _ = find_peaks(-speed, prominence=prominence, distance=min_sep_samples)
    if len(minima) == 0:
        return []
    minima = np.sort(minima)

    has_xy = df["x"].notna().all() and df["y"].notna().all()
    curv = curvature_xy(df["x"].to_numpy(float), df["y"].to_numpy(float)) if has_xy else None

    corners: list[Corner] = []
    n_c = len(minima)
    for k, apex_i in enumerate(minima):
        apex_dist = float(dist[apex_i])
        # Bound each corner by the midpoints to neighbouring apexes — robust to flat
        # straights where no clean speed maximum exists to delimit the corner.
        entry = 0.5 * (float(dist[minima[k - 1]]) + apex_dist) if k > 0 else float(dist[0])
        exit_ = 0.5 * (apex_dist + float(dist[minima[k + 1]])) if k < n_c - 1 else float(dist[-1])

        m = segment_metrics(df, entry, exit_)
        mean_curv = float("nan")
        if curv is not None:
            mask = (dist >= entry) & (dist <= exit_)
            if mask.any():
                mean_curv = float(np.nanmean(curv[mask]))

        corners.append(Corner(
            number=k + 1,
            apex_dist=apex_dist,
            apex_speed_kph=float(speed[apex_i]),
            entry_dist=entry,
            exit_dist=exit_,
            brake_point_dist=m["brake_point"],
            throttle_on_dist=m["throttle_on"],
            mean_curvature=mean_curv,
        ))
    return corners


def annotate_time_loss(corners: list[Corner], delta_df: pd.DataFrame) -> list[Corner]:
    """Fill each corner's ``time_delta_s`` = delta at exit − delta at entry.

    Positive means the comparison lap lost time through that corner.
    """
    d = delta_df["distance_m"].to_numpy(dtype=float)
    delta = delta_df["delta_s"].to_numpy(dtype=float)
    for c in corners:
        d_entry = float(np.interp(c.entry_dist, d, delta))
        d_exit = float(np.interp(c.exit_dist, d, delta))
        c.time_delta_s = d_exit - d_entry
    return corners
