"""Lap selection and reference picking (M2).

Operates purely on the canonical :class:`Lap`/``LapMeta`` — so the same logic serves
FastF1 today and iRacing tomorrow.
"""

from __future__ import annotations

from .schema import Lap

#: Minimum sample count for a lap to be considered analysable.
MIN_SAMPLES = 10


def is_valid_lap(lap: Lap) -> bool:
    """A lap is valid if it's a complete, non-pit, non-deleted flying lap with a time."""
    m = lap.meta
    return bool(
        m.is_valid
        and not m.is_pit
        and not m.is_out_lap
        and not m.is_in_lap
        and m.lap_time_s is not None
        and m.lap_time_s > 0
        and len(lap.samples) >= MIN_SAMPLES
    )


def valid_laps(laps: list[Lap]) -> list[Lap]:
    """Filter to valid flying laps, dropping out/in/pit/deleted/incomplete laps."""
    return [lap for lap in laps if is_valid_lap(lap)]


def pick_reference(laps: list[Lap]) -> Lap:
    """Return the fastest valid lap (the delta reference)."""
    candidates = valid_laps(laps)
    if not candidates:
        raise ValueError("no valid laps to pick a reference from")
    return min(candidates, key=lambda lap: lap.meta.lap_time_s)


def fastest_per_driver(laps: list[Lap]) -> dict[str, Lap]:
    """Map each driver to their fastest valid lap."""
    best: dict[str, Lap] = {}
    for lap in valid_laps(laps):
        drv = lap.meta.driver
        if drv not in best or lap.meta.lap_time_s < best[drv].meta.lap_time_s:
            best[drv] = lap
    return best


def drivers_in(laps: list[Lap]) -> list[str]:
    """Distinct driver codes present, ordered by fastest valid lap time."""
    best = fastest_per_driver(laps)
    return sorted(best, key=lambda d: best[d].meta.lap_time_s)
