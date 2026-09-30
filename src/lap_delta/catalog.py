"""Pure parsing helpers that turn FastF1 catalog DataFrames into dropdown options.

These take already-fetched pandas DataFrames (event schedule, session results, laps table)
and return plain Python structures for the UI. Keeping them free of any network / FastF1
import makes the cascading-dropdown logic unit-testable with synthetic frames.
"""

from __future__ import annotations

import pandas as pd

from .schema import format_laptime

#: FastF1 session names -> the short identifiers used elsewhere in the app.
_SESSION_ABBR = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Qualifying": "Q",
    "Sprint Qualifying": "SQ",
    "Sprint Shootout": "SS",
    "Sprint": "S",
    "Race": "R",
}


def _isna(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None


def event_options(schedule_df: pd.DataFrame) -> list[dict]:
    """Real race weekends for a year (testing events excluded), ordered by round.

    Each option: ``{"round": int, "name": str, "event_format": str, "label": str}``.
    """
    options: list[dict] = []
    for _, row in schedule_df.iterrows():
        fmt = str(row.get("EventFormat", "") or "")
        rnd = row.get("RoundNumber")
        if fmt == "testing" or _isna(rnd) or int(rnd) == 0:
            continue
        name = str(row.get("EventName", "") or "")
        options.append({
            "round": int(rnd),
            "name": name,
            "event_format": fmt,
            "label": f"R{int(rnd)} — {name}",
        })
    options.sort(key=lambda o: o["round"])
    return options


def session_options(event_row) -> list[tuple[str, str]]:
    """Sessions available for an event as ``(label, identifier)`` pairs.

    ``label`` is the full FastF1 name (e.g. "Qualifying"); ``identifier`` is the short code
    (``Q``, ``FP1``, ``S`` …) accepted by ``get_session``. Falls back to the full name when a
    session name isn't in the known map.
    """
    getter = event_row.get if hasattr(event_row, "get") else event_row.__getitem__
    out: list[tuple[str, str]] = []
    for i in range(1, 6):
        try:
            name = getter(f"Session{i}")
        except (KeyError, IndexError):
            name = None
        if _isna(name) or not str(name):
            continue
        name = str(name)
        out.append((name, _SESSION_ABBR.get(name, name)))
    return out


def driver_options(results_df: pd.DataFrame) -> list[dict]:
    """Drivers who took part, as ``{"code", "name", "team", "color"}`` (results order)."""
    out: list[dict] = []
    for _, row in results_df.iterrows():
        code = row.get("Abbreviation")
        if _isna(code) or not str(code):
            continue
        color = row.get("TeamColor")
        color = str(color) if isinstance(color, str) and color else None
        if color and not color.startswith("#"):
            color = f"#{color}"
        out.append({
            "code": str(code),
            "name": str(row.get("FullName", code) or code),
            "team": str(row.get("TeamName", "") or ""),
            "color": color,
        })
    return out


def lap_options(laps_df: pd.DataFrame, driver: str) -> list[dict]:
    """A driver's timed laps as ``{"lap_number", "lap_time_s", "label"}``, fastest first.

    Filters to laps with a valid ``LapTime`` and (when present) excludes deleted / inaccurate
    laps, mirroring the canonical validity rules.
    """
    sub = laps_df[laps_df["Driver"] == driver]
    rows: list[dict] = []
    for _, lap in sub.iterrows():
        lt = lap.get("LapTime")
        if _isna(lt):
            continue
        if "Deleted" in sub.columns and bool(lap.get("Deleted", False)):
            continue
        if "IsAccurate" in sub.columns and not bool(lap.get("IsAccurate", True)):
            continue
        lap_time_s = float(lt.total_seconds())
        num = lap.get("LapNumber")
        lap_number = None if _isna(num) else int(num)
        rows.append({
            "lap_number": lap_number,
            "lap_time_s": lap_time_s,
            "label": f"L{lap_number} — {format_laptime(lap_time_s)}",
        })
    rows.sort(key=lambda r: r["lap_time_s"])
    return rows
