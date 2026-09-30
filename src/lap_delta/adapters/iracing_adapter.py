"""iRacing ``.ibt`` adapter (M7): sim telemetry -> canonical :class:`Lap` list.

``pyirsdk`` (imported as ``irsdk``) reads ``.ibt`` files via its ``IBT`` class. It is
imported **lazily inside** :meth:`IracingIbtAdapter.load` so importing ``lap_delta`` stays
cheap and the analysis tests never require it (mirroring the FastF1 adapter).

Two source-specific quirks live here and nowhere else:

* ``IBT`` exposes channels (``get_all``) but **does not parse the session-info YAML**, so we
  read that block straight from the file header offsets it *does* expose
  (``_header.session_info_offset/len`` over ``_shared_mem``) to recover track / car / driver.
* iRacing ships one continuous stream, not per-lap files, so we split it into laps at
  start/finish crossings (``LapDistPct`` wraparound).

Channel mapping (raw -> canonical, normalized at this boundary):

    LapDist                     -> distance_m   (np.maximum.accumulate: guaranteed monotonic)
    SessionTime - segment_start -> time_s
    Speed (m/s) * 3.6           -> speed_kph
    Throttle                    -> throttle      (already 0..1)
    Brake                       -> brake         (continuous pressure 0..1 — unlike FastF1)
    Gear                        -> gear
    RPM                         -> rpm
    Lat / Lon                   -> x, y          (local equirectangular projection)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..schema import Lap, LapMeta
from .base import TelemetryAdapter

log = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_000.0
MIN_SEGMENT_SAMPLES = 50
WRAP_THRESHOLD = -0.5  # a LapDistPct backward jump this large marks a start/finish crossing
#: A flying lap never drops near-stationary; below this (km/h, robust 2nd-percentile) the
#: segment is a standing start, pit crawl, spin or stop — not a clean lap.
MIN_FLYING_SPEED_KPH = 15.0

#: Raw iRacing channels consumed here. Everything downstream sees only the canonical schema.
_REQUIRED = ("SessionTime", "LapDist", "LapDistPct", "Speed", "Throttle", "Brake")
_OPTIONAL = ("Gear", "RPM", "Lap", "Lat", "Lon", "OnPitRoad")
_CHANNELS = _REQUIRED + _OPTIONAL


# --- pure helpers (unit-tested directly against synthetic arrays) -------------------

def split_lap_segments(pct, *, min_len: int = MIN_SEGMENT_SAMPLES) -> list[slice]:
    """Split a continuous ``LapDistPct`` stream into per-lap slices at S/F wraparounds.

    A lap boundary is where the fraction-of-lap jumps backward (``≈1.0 → ≈0.0``). Segments
    shorter than ``min_len`` samples (capture fragments) are dropped.
    """
    pct = np.asarray(pct, dtype=float)
    n = len(pct)
    if n == 0:
        return []
    cuts = np.where(np.diff(pct) < WRAP_THRESHOLD)[0] + 1
    bounds = [0, *cuts.tolist(), n]
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b - a >= min_len]


def project_xy(lat_deg, lon_deg) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular projection of lat/lon (degrees) to local meters.

    Longitude is scaled by ``cos(mean latitude)`` so x and y share one metric scale; the
    origin is the first sample. Accurate to well under a meter over a few-km lap — plenty
    for a track map.
    """
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    if len(lat) == 0:
        return np.array([]), np.array([])
    lat0 = np.radians(np.nanmean(lat))
    x = EARTH_RADIUS_M * np.radians(lon) * np.cos(lat0)
    y = EARTH_RADIUS_M * np.radians(lat)
    return x - x[0], y - y[0]


def build_lap_frame(channels: dict[str, np.ndarray], sl: slice) -> pd.DataFrame | None:
    """Map one segment's raw channels to a canonical ``samples`` DataFrame (or ``None``).

    Returns ``None`` if a required channel is absent or the segment is too short to analyse.
    """
    if any(channels.get(name) is None for name in _REQUIRED):
        return None

    def req(name):
        return np.asarray(channels[name][sl], dtype=float)

    def opt(name, default=np.nan):
        arr = channels.get(name)
        return np.asarray(arr[sl], dtype=float) if arr is not None else None

    dist = req("LapDist")
    if len(dist) < 2:
        return None
    time = req("SessionTime")

    lat, lon = channels.get("Lat"), channels.get("Lon")
    if lat is not None and lon is not None:
        x, y = project_xy(np.asarray(lat)[sl], np.asarray(lon)[sl])
    else:
        x = y = np.full(len(dist), np.nan)

    gear = opt("Gear")
    gear = np.nan_to_num(gear).astype(int) if gear is not None else np.zeros(len(dist), dtype=int)
    rpm = opt("RPM")

    return pd.DataFrame({
        "distance_m": np.maximum.accumulate(dist),  # LapDist is per-lap; force monotonic
        "time_s": time - time[0],
        "speed_kph": req("Speed") * 3.6,
        "throttle": np.clip(req("Throttle"), 0.0, 1.0),
        "brake": np.clip(req("Brake"), 0.0, 1.0),   # stays continuous (iRacing's advantage)
        "gear": gear,
        "rpm": rpm if rpm is not None else np.full(len(dist), np.nan),
        "x": x,
        "y": y,
    })


def segment_is_valid(channels: dict[str, np.ndarray], sl: slice, track_len_m: float,
                     lap_time_s: float | None) -> tuple[bool, bool]:
    """Return ``(is_valid, is_pit)`` for a segment.

    A valid flying lap (a) covers ~the whole track (LapDist reaches ≥95 % of the layout
    length from a low start), (b) never drops near-stationary — its 2nd-percentile speed
    stays above :data:`MIN_FLYING_SPEED_KPH`, which rejects standing starts / pit crawls /
    spins that ``LapDist`` coverage alone can't (a lone spurious ``LapDist=0`` sample makes a
    standing-start segment look "complete") — (c) never touches pit road, and (d) has a
    positive lap time. So capture fragments, out/in laps and formation laps are all excluded.
    """
    on_pit = channels.get("OnPitRoad")
    is_pit = bool(np.asarray(on_pit[sl]).astype(bool).any()) if on_pit is not None else False

    dist = np.asarray(channels["LapDist"][sl], dtype=float)
    covers_full = bool(
        track_len_m > 0
        and np.percentile(dist, 2) <= 0.05 * track_len_m
        and dist.max() >= 0.95 * track_len_m
    )
    speed_kph = np.asarray(channels["Speed"][sl], dtype=float) * 3.6
    keeps_moving = bool(len(speed_kph) and np.percentile(speed_kph, 2) >= MIN_FLYING_SPEED_KPH)

    n = sl.stop - sl.start
    is_valid = bool(
        covers_full and keeps_moving and not is_pit and n >= MIN_SEGMENT_SAMPLES
        and lap_time_s is not None and lap_time_s > 0
    )
    return is_valid, is_pit


# --- adapter ------------------------------------------------------------------------

class IracingIbtAdapter(TelemetryAdapter):
    """Load iRacing ``.ibt`` telemetry and split it into canonical laps."""

    source = "iracing"

    def load(self, ibt_path: str, laps: str = "all", **kwargs) -> list[Lap]:  # noqa: ARG002
        """Return canonical laps from an ``.ibt`` file (oldest first).

        ``laps="valid"`` returns only complete, non-pit flying laps (via the shared
        :mod:`lap_delta.laps` rules); ``"all"`` (default) returns every split segment so the
        UI can present them and let the user choose.
        """
        try:
            import irsdk  # lazy: heavy + Windows-oriented; not needed for analysis/tests
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Reading iRacing .ibt files needs 'pyirsdk'. Install it with "
                "`pip install pyirsdk` (or `pip install -e \".[iracing]\"`)."
            ) from exc

        ibt = irsdk.IBT()
        ibt.open(str(ibt_path))
        try:
            names = set(ibt.var_headers_names or ())
            channels = {ch: np.asarray(ibt.get_all(ch)) for ch in _CHANNELS if ch in names}
            labels = self._session_labels(ibt, irsdk)
        finally:
            ibt.close()

        missing = [c for c in _REQUIRED if c not in channels]
        if missing:
            raise ValueError(f"{ibt_path}: .ibt is missing required channels {missing}")

        track_len_m = float(np.nanmax(channels["LapDist"]))
        out: list[Lap] = []
        for sl in split_lap_segments(channels["LapDistPct"]):
            lap = self._build_lap(channels, sl, labels, track_len_m)
            if lap is not None:
                out.append(lap)

        if laps == "valid":
            from ..laps import valid_laps
            return valid_laps(out)
        return out

    # -- internals -----------------------------------------------------------------

    def _build_lap(self, channels, sl, labels, track_len_m) -> Lap | None:
        df = build_lap_frame(channels, sl)
        if df is None or len(df) < 2:
            return None

        lap_time_s = float(df["time_s"].iloc[-1])
        is_valid, is_pit = segment_is_valid(channels, sl, track_len_m, lap_time_s)

        lap_arr = channels.get("Lap")
        lap_number = int(np.median(np.asarray(lap_arr[sl]))) if lap_arr is not None else None

        dt = np.diff(df["time_s"].to_numpy())
        sample_rate = float(1.0 / np.median(dt)) if len(dt) and np.median(dt) > 0 else None

        meta = LapMeta(
            source=self.source,
            driver=labels.get("driver", ""),
            team=labels.get("car", ""),  # the car is the most useful "team"-like label here
            event=labels.get("session", ""),
            session=labels.get("session", ""),
            track=labels.get("track", ""),
            lap_number=lap_number,
            lap_time_s=lap_time_s,
            is_valid=is_valid,
            is_pit=is_pit,
            sample_count=len(df),
            sample_rate_hz=sample_rate,
            compound=labels.get("car"),
        )
        try:
            return self.validate(Lap(samples=df, meta=meta))
        except Exception as exc:  # noqa: BLE001 - a malformed fragment shouldn't sink the load
            log.warning("skipping .ibt segment %s: %s", sl, exc)
            return None

    @staticmethod
    def _session_labels(ibt, irsdk) -> dict[str, str]:
        """Recover track / car / driver from the session-info YAML the IBT reader skips."""
        try:
            header, mem = ibt._header, ibt._shared_mem
            raw = mem[header.session_info_offset:
                      header.session_info_offset + header.session_info_len]
            utf8_bom = b"\xef\xbb\xbf"
            enc = "utf-8" if raw[:len(utf8_bom)] == utf8_bom else "cp1252"
            text = raw.rstrip(b"\x00").decode(enc, errors="replace")
            info = irsdk.yaml.load(text, Loader=irsdk.CustomYamlSafeLoader)
        except Exception as exc:  # noqa: BLE001 - metadata is best-effort
            log.warning("could not read .ibt session info: %s", exc)
            return {}

        weekend = info.get("WeekendInfo", {}) or {}
        driver_info = info.get("DriverInfo", {}) or {}
        drivers = driver_info.get("Drivers", []) or []
        idx = driver_info.get("DriverCarIdx", 0)
        me = next((d for d in drivers if d.get("CarIdx") == idx), drivers[0] if drivers else {})
        sessions = (info.get("SessionInfo", {}) or {}).get("Sessions", []) or []
        session = str(sessions[-1].get("SessionType", "")) if sessions else ""
        return {
            "track": str(weekend.get("TrackDisplayName", "") or ""),
            "driver": str(me.get("UserName", "") or ""),
            "car": str(me.get("CarScreenName", "") or ""),
            "session": session,
        }
