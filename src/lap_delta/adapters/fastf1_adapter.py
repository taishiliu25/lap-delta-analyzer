"""FastF1 adapter (M1): real Formula 1 telemetry -> canonical :class:`Lap` list.

``fastf1`` is imported lazily inside methods so the pure-analysis layer and its tests
never require it (or a network connection) to be present.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import Lap, LapMeta
from .base import TelemetryAdapter

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("cache")


class FastF1Adapter(TelemetryAdapter):
    """Load F1 car + position telemetry by (year, event, session, drivers)."""

    source = "fastf1"

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR):
        import fastf1  # lazy

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(self.cache_dir))

    def load(
        self,
        year: int,
        event: str | int,
        session: str = "Q",
        drivers: list[str] | str | None = None,
        laps: str = "all",
    ) -> list[Lap]:
        """Return canonical laps for the given session.

        ``drivers`` filters to specific driver codes (e.g. ``["VER", "PER"]``); ``laps``
        is ``"all"`` or ``"quick"`` (drop slow in/out laps via FastF1's quicklap filter).
        """
        ses = self._open_session(year, event, session, telemetry=True)

        if isinstance(drivers, str):
            drivers = [drivers]

        event_name = str(getattr(ses.event, "EventName", event))
        track = str(getattr(ses.event, "Location", ""))
        colors = self._team_colors(ses)

        out: list[Lap] = []
        for drv, lap_row in self._iter_laps(ses, drivers, laps):
            lap = self._build_lap(lap_row, drv, year, event_name, session, track,
                                  team_color=colors.get(str(drv)))
            if lap is not None:
                out.append(lap)
        return out

    # -- catalog (lightweight, for cascading dropdowns) ----------------------------

    def schedule(self, year: int):
        """Full event schedule for a year (rounds, names, formats, session names)."""
        import fastf1  # lazy

        return fastf1.get_event_schedule(int(year))

    def session_summary(self, year: int, event: str | int, session: str = "Q"):
        """Load laps + results WITHOUT telemetry — cheap driver/lap/sector metadata.

        Returns ``(results_df, laps_df)`` where ``results_df`` lists the drivers who ran
        (``Abbreviation``, ``FullName``, ``TeamName``, ``TeamColor``) and ``laps_df`` is the
        per-lap timing table (``Driver``, ``LapNumber``, ``LapTime``, ``Sector*Time`` …).
        """
        ses = self._open_session(year, event, session, telemetry=False)
        return ses.results, ses.laps

    def load_selected(
        self, year: int, event: str | int, session: str, selections: list[tuple[str, int]]
    ) -> list[Lap]:
        """Build canonical laps for specific ``(driver, lap_number)`` pairs only.

        Telemetry for the session is loaded once (and cached by FastF1); we then build just
        the requested laps rather than every lap in the session.
        """
        ses = self._open_session(year, event, session, telemetry=True)
        event_name = str(getattr(ses.event, "EventName", event))
        track = str(getattr(ses.event, "Location", ""))
        colors = self._team_colors(ses)

        out: list[Lap] = []
        for driver, lap_number in selections:
            sub = ses.laps.pick_drivers(driver) if hasattr(ses.laps, "pick_drivers") \
                else ses.laps.pick_driver(driver)
            match = sub[sub["LapNumber"] == lap_number]
            if match.empty:
                log.warning("no lap %s for %s", lap_number, driver)
                continue
            lap = self._build_lap(match.iloc[0], driver, year, event_name, session, track,
                                  team_color=colors.get(str(driver)))
            if lap is not None:
                out.append(lap)
        return out

    def circuit_corners(
        self, year: int, event: str | int, session: str = "Q"
    ) -> list[dict]:
        """Official numbered corners as ``[{"number", "label", "distance_m"}, …]``.

        Uses FastF1's ``get_circuit_info()`` (MultiViewer API, cached), whose corner
        ``Distance`` is measured from the start/finish line — the same convention as our
        canonical ``distance_m`` — so callers can place labels by interpolating a lap's x/y at
        each corner distance. Returns ``[]`` if circuit data is unavailable (offline / older
        seasons) so callers can fall back to detected corners.
        """
        try:
            ses = self._open_session(year, event, session, telemetry=True)
            info = ses.get_circuit_info()
            corners = info.corners
        except Exception as exc:  # noqa: BLE001 — circuit data is best-effort
            log.warning("circuit info unavailable: %s", exc)
            return []

        out: list[dict] = []
        for _, row in corners.iterrows():
            dist = row.get("Distance")
            if dist is None or pd.isna(dist):
                continue
            number = int(row.get("Number", 0))
            letter = row.get("Letter") or ""
            out.append({"number": number, "label": f"{number}{letter}",
                        "distance_m": float(dist)})
        out.sort(key=lambda c: c["distance_m"])
        return out

    # -- internals -----------------------------------------------------------------

    def _open_session(self, year, event, session, *, telemetry: bool):
        """Open and load a FastF1 session (``telemetry=False`` for cheap metadata loads)."""
        import fastf1  # lazy — also ensures the cache is configured

        ses = fastf1.get_session(year, event, session)
        ses.load(laps=True, telemetry=telemetry, weather=False, messages=False)
        return ses

    @staticmethod
    def _team_colors(ses) -> dict[str, str]:
        """Map driver code -> team hex color from session results (best-effort)."""
        colors: dict[str, str] = {}
        try:
            results = ses.results
        except Exception:  # noqa: BLE001 — results may be unavailable for some sessions
            return colors
        for _, row in results.iterrows():
            code, col = row.get("Abbreviation"), row.get("TeamColor")
            if isinstance(code, str) and isinstance(col, str) and col:
                colors[code] = col if col.startswith("#") else f"#{col}"
        return colors

    @staticmethod
    def _iter_laps(session, drivers, laps):
        all_laps = session.laps
        driver_list = drivers or list(all_laps["Driver"].unique())
        for drv in driver_list:
            if hasattr(all_laps, "pick_drivers"):
                sub = all_laps.pick_drivers(drv)
            else:  # pre-3.4 fallback
                sub = all_laps.pick_driver(drv)
            if laps == "quick" and hasattr(sub, "pick_quicklaps"):
                sub = sub.pick_quicklaps()
            for _, lap_row in sub.iterlaps():
                yield drv, lap_row

    def _build_lap(
        self, lap_row, driver, year, event, session, track, team_color=None
    ) -> Lap | None:
        try:
            tel = lap_row.get_telemetry()
        except Exception as exc:  # noqa: BLE001 — telemetry gaps are expected; skip lap
            log.warning("skipping %s lap %s: %s", driver, lap_row.get("LapNumber"), exc)
            return None

        if tel is None or len(tel) < 2:
            return None
        if "Distance" not in tel.columns:
            tel = tel.add_distance()

        df = self._map_channels(tel)
        if df is None or len(df) < 2:
            return None

        lap_time = lap_row.get("LapTime")
        lap_time_s = None if pd.isna(lap_time) else float(lap_time.total_seconds())

        # Anchor telemetry time to the official lap time so the delta endpoint matches
        # the timing-screen gap (telemetry time-span and timing can differ by tens of ms).
        last_t = float(df["time_s"].iloc[-1])
        if lap_time_s and last_t > 0:
            df["time_s"] = df["time_s"] * (lap_time_s / last_t)

        pit_in, pit_out = lap_row.get("PitInTime"), lap_row.get("PitOutTime")
        is_accurate = bool(lap_row.get("IsAccurate", True))
        deleted = bool(lap_row.get("Deleted", False))

        dt = np.diff(df["time_s"].to_numpy())
        sample_rate = float(1.0 / np.median(dt)) if len(dt) and np.median(dt) > 0 else None

        def _sector(name: str) -> float | None:
            v = lap_row.get(name)
            return None if v is None or pd.isna(v) else float(v.total_seconds())

        meta = LapMeta(
            source=self.source,
            driver=str(driver),
            team=str(lap_row.get("Team", "")),
            year=int(year),
            event=str(event),
            session=str(session),
            track=str(track),
            lap_number=None if pd.isna(lap_row.get("LapNumber")) else int(lap_row.get("LapNumber")),
            lap_time_s=lap_time_s,
            is_valid=is_accurate and not deleted and lap_time_s is not None,
            is_out_lap=not pd.isna(pit_out),
            is_in_lap=not pd.isna(pit_in),
            is_pit=(not pd.isna(pit_in)) or (not pd.isna(pit_out)),
            compound=None if pd.isna(lap_row.get("Compound")) else str(lap_row.get("Compound")),
            sample_count=len(df),
            sample_rate_hz=sample_rate,
            sector1_s=_sector("Sector1Time"),
            sector2_s=_sector("Sector2Time"),
            sector3_s=_sector("Sector3Time"),
            team_color=team_color,
        )
        return Lap(samples=df, meta=meta)

    @staticmethod
    def _map_channels(tel: pd.DataFrame) -> pd.DataFrame | None:
        distance = np.asarray(tel["Distance"], dtype=float)
        distance = np.maximum.accumulate(distance)  # guarantee monotonic for interpolation

        time_s = (tel["Time"] - tel["Time"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)

        throttle = np.clip(np.asarray(tel["Throttle"], dtype=float) / 100.0, 0.0, 1.0)

        brake = np.asarray(tel["Brake"], dtype=float)
        if np.nanmax(brake) > 1.0:  # some seasons encode 0..100 instead of bool
            brake = brake / 100.0
        brake = np.clip(brake, 0.0, 1.0)

        gear = np.nan_to_num(np.asarray(tel.get("nGear", 0), dtype=float)).astype(int)

        def col(name):
            return np.asarray(tel[name], dtype=float) if name in tel.columns else np.full(len(tel), np.nan)

        return pd.DataFrame({
            "distance_m": distance,
            "time_s": time_s,
            "speed_kph": np.asarray(tel["Speed"], dtype=float),
            "throttle": throttle,
            "brake": brake,
            "gear": gear,
            "rpm": col("RPM"),
            "x": col("X"),
            "y": col("Y"),
        })
