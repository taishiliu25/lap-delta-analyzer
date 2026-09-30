"""Rule-based, plain-English coaching hints (M5).

Corners are detected on the *reference* lap, then measured identically on the
*comparison* lap (same distance windows). Simple, transparent rules turn the metric
differences into driver-facing suggestions. These are heuristics - framed as
suggestions, not ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import units
from .corners import Corner, annotate_time_loss, detect_corners, segment_metrics
from .schema import Lap

# Thresholds (metric) above which a difference is worth mentioning.
_SPEED_KPH = 2.0
_DIST_M = 8.0
_TIME_S = 0.03


def _speed_txt(delta_kph: float, imperial: bool) -> float:
    return abs(float(units.speed(delta_kph, imperial)))


def _dist_txt(delta_m: float, imperial: bool) -> float:
    return abs(float(units.distance(delta_m, imperial)))


def _hint(corner: Corner, ref: dict, cmp: dict, imperial: bool,
          ref_code: str, cmp_code: str) -> str:
    """Plain-English, driver-named hint for one corner (``cmp`` relative to ``ref``)."""
    su, du = units.speed_unit(imperial), units.distance_unit(imperial)
    hints: list[str] = []

    dv = cmp["min_speed"] - ref["min_speed"]
    if np.isfinite(dv) and dv < -_SPEED_KPH:
        hints.append(f"{cmp_code} carries {_speed_txt(dv, imperial):.0f} {su} less apex speed "
                     f"than {ref_code} — carry more entry speed")
    elif np.isfinite(dv) and dv > _SPEED_KPH:
        hints.append(f"{cmp_code} carries {_speed_txt(dv, imperial):.0f} {su} more apex speed "
                     f"than {ref_code}")

    dbp = cmp["brake_point"] - ref["brake_point"]
    if np.isfinite(dbp) and dbp < -_DIST_M:
        hints.append(f"{cmp_code} brakes {_dist_txt(dbp, imperial):.0f} {du} earlier than "
                     f"{ref_code} — try braking later")
    elif np.isfinite(dbp) and dbp > _DIST_M:
        hints.append(f"{cmp_code} brakes {_dist_txt(dbp, imperial):.0f} {du} later than {ref_code}")

    dto = cmp["throttle_on"] - ref["throttle_on"]
    if np.isfinite(dto) and dto > _DIST_M:
        hints.append(f"{cmp_code} gets to throttle {_dist_txt(dto, imperial):.0f} {du} later "
                     f"than {ref_code} — get on power sooner")
    elif np.isfinite(dto) and dto < -_DIST_M:
        hints.append(f"{cmp_code} gets to throttle {_dist_txt(dto, imperial):.0f} {du} earlier "
                     f"than {ref_code} — good")

    if not hints:
        return f"{cmp_code} matched {ref_code} closely"
    return "; ".join(hints)


def coaching_table(
    reference: Lap,
    comparison: Lap,
    delta_df: pd.DataFrame,
    *,
    imperial: bool = False,
    ref_code: str | None = None,
    cmp_code: str | None = None,
) -> pd.DataFrame:
    """Ranked per-corner comparison with coaching hints (worst time loss first).

    Hints name the drivers explicitly; ``ref_code``/``cmp_code`` default to each lap's driver.
    """
    ref_code = ref_code or reference.meta.driver or "REF"
    cmp_code = cmp_code or comparison.meta.driver or "CMP"
    corners = annotate_time_loss(detect_corners(reference), delta_df)
    ref_df, cmp_df = reference.samples, comparison.samples
    su, du = units.speed_unit(imperial), units.distance_unit(imperial)

    rows = []
    for c in corners:
        ref_m = segment_metrics(ref_df, c.entry_dist, c.exit_dist)
        cmp_m = segment_metrics(cmp_df, c.entry_dist, c.exit_dist)
        rows.append({
            "Corner": c.number,
            f"Apex @ ({du})": round(float(units.distance(c.apex_dist, imperial)), 0),
            "Time delta (s)": round(c.time_delta_s, 3),
            f"{ref_code} apex ({su})": round(float(units.speed(ref_m["min_speed"], imperial)), 1),
            f"{cmp_code} apex ({su})": round(float(units.speed(cmp_m["min_speed"], imperial)), 1),
            "Coaching hint": _hint(c, ref_m, cmp_m, imperial, ref_code, cmp_code),
        })

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("Time delta (s)", ascending=False).reset_index(drop=True)
    return table
