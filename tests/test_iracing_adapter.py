"""iRacing adapter pure-helper tests (M7) — synthetic arrays, no pyirsdk / no .ibt file.

These run in the default offline suite. They exercise the source-specific logic (stream
lap-splitting, lat/lon projection, channel mapping, validity) that turns a raw iRacing
capture into canonical laps — the same way the analysis modules are tested on synthetic laps.
"""

from __future__ import annotations

import numpy as np
import pytest

from lap_delta.adapters.iracing_adapter import (
    MIN_FLYING_SPEED_KPH,
    build_lap_frame,
    project_xy,
    segment_is_valid,
    split_lap_segments,
)
from lap_delta.laps import is_valid_lap
from lap_delta.schema import Lap as _Lap
from lap_delta.schema import LapMeta, validate_schema

# --- split_lap_segments ------------------------------------------------------------

def _sawtooth(*lengths):
    """Concatenate ramps 0->~1, each a 'lap'; the joins are S/F wraparounds (big drops)."""
    return np.concatenate([np.linspace(0.0, 0.999, n) for n in lengths])


def test_split_on_wraparound_counts_laps():
    pct = _sawtooth(120, 120, 120)
    segs = split_lap_segments(pct, min_len=50)
    assert len(segs) == 3
    assert [s.stop - s.start for s in segs] == [120, 120, 120]
    # slices tile the stream with no gaps/overlaps
    assert segs[0].start == 0 and segs[-1].stop == len(pct)


def test_split_drops_short_fragments():
    pct = _sawtooth(120, 30)  # a full lap then a 30-sample tail fragment
    assert len(split_lap_segments(pct, min_len=50)) == 1
    assert len(split_lap_segments(pct, min_len=20)) == 2


def test_split_no_wraparound_is_single_segment():
    assert len(split_lap_segments(np.linspace(0, 0.6, 200), min_len=50)) == 1
    assert split_lap_segments(np.array([]), min_len=50) == []


# --- project_xy --------------------------------------------------------------------

def test_project_xy_origin_and_scale():
    lat = np.array([47.20, 47.201, 47.202])   # ~Red Bull Ring latitude
    lon = np.array([14.76, 14.761, 14.762])
    x, y = project_xy(lat, lon)
    assert x[0] == 0.0 and y[0] == 0.0                     # origin at first sample
    # 0.001 deg latitude ~ 111 m; longitude shrunk by cos(lat)
    assert np.isclose(y[1], np.radians(0.001) * 6_371_000.0, rtol=1e-6)
    assert x[-1] > 0 and y[-1] > 0
    assert not np.isnan(x).any()


def test_project_xy_degenerate_is_zero_no_nan():
    lat = np.full(5, 47.2)
    lon = np.full(5, 14.76)
    x, y = project_xy(lat, lon)
    assert np.allclose(x, 0.0) and np.allclose(y, 0.0)


# --- build_lap_frame ---------------------------------------------------------------

def _flying_channels(n=400, length=4000.0):
    """A synthetic single-lap capture: full distance, always moving, off pit road."""
    dist = np.linspace(0.0, length, n)
    return {
        "SessionTime": 100.0 + np.linspace(0.0, 90.0, n),   # nonzero t0 -> must be rebased
        "LapDist": dist,
        "LapDistPct": dist / length,
        "Speed": np.full(n, 40.0),                           # m/s -> 144 km/h
        "Throttle": np.linspace(0.2, 1.0, n),
        "Brake": np.linspace(0.8, 0.0, n),                   # continuous pressure
        "Gear": np.clip((np.linspace(1, 6, n)).astype(int), 1, 6),
        "RPM": np.full(n, 9000.0),
        "Lat": 47.20 + dist * 1e-6,
        "Lon": 14.76 + dist * 1e-6,
        "OnPitRoad": np.zeros(n, dtype=bool),
    }


def test_build_lap_frame_maps_units_and_passes_schema():
    ch = _flying_channels()
    df = build_lap_frame(ch, slice(0, len(ch["LapDist"])))
    validate_schema(_Lap(df, LapMeta(source="iracing")), require_position=True)

    assert np.isclose(df["time_s"].iloc[0], 0.0)             # SessionTime rebased to lap start
    assert np.isclose(df["speed_kph"].iloc[0], 40.0 * 3.6)   # m/s -> km/h
    assert 0.0 < df["brake"].iloc[0] < 1.0                   # brake stayed continuous
    assert np.all(np.diff(df["distance_m"]) >= 0)            # monotonic distance


def test_build_lap_frame_requires_core_channels():
    ch = _flying_channels()
    del ch["Speed"]
    assert build_lap_frame(ch, slice(0, 400)) is None


def test_build_lap_frame_without_position_is_nan_xy():
    ch = _flying_channels()
    del ch["Lat"]
    df = build_lap_frame(ch, slice(0, 400))
    assert df["x"].isna().all()
    validate_schema(_Lap(df, LapMeta(source="iracing")))     # still valid without position


# --- segment_is_valid --------------------------------------------------------------

def test_valid_flying_lap():
    ch = _flying_channels()
    sl = slice(0, len(ch["LapDist"]))
    is_valid, is_pit = segment_is_valid(ch, sl, track_len_m=4000.0, lap_time_s=90.0)
    assert is_valid and not is_pit


def test_standing_start_rejected_despite_full_distance():
    """A lone spurious LapDist=0 makes a standing start look 'complete'; the speed gate
    rejects it because the car sits near-stationary at the grid."""
    ch = _flying_channels()
    n = len(ch["LapDist"])
    ch["Speed"][: n // 4] = 0.0                              # stationary on the grid
    ch["LapDist"][0] = 0.0                                   # spurious low sample
    ch["LapDist"][1:] = np.linspace(2800.0, 4000.0, n - 1)   # actually starts 2/3 around
    is_valid, _ = segment_is_valid(ch, slice(0, n), track_len_m=4000.0, lap_time_s=70.0)
    assert not is_valid


def test_pit_segment_flagged_and_invalid():
    ch = _flying_channels()
    ch["OnPitRoad"][:] = True
    is_valid, is_pit = segment_is_valid(ch, slice(0, 400), track_len_m=4000.0, lap_time_s=90.0)
    assert is_pit and not is_valid


def test_partial_lap_rejected():
    ch = _flying_channels()
    ch["LapDist"] = np.linspace(0.0, 2000.0, len(ch["LapDist"]))  # only half the track
    is_valid, _ = segment_is_valid(ch, slice(0, 400), track_len_m=4000.0, lap_time_s=45.0)
    assert not is_valid


def test_valid_segment_feeds_canonical_lap_filter():
    """A segment marked valid + its frame together satisfy the shared laps.is_valid_lap."""
    ch = _flying_channels()
    sl = slice(0, len(ch["LapDist"]))
    df = build_lap_frame(ch, sl)
    is_valid, is_pit = segment_is_valid(ch, sl, 4000.0, float(df["time_s"].iloc[-1]))
    lap = _Lap(df, LapMeta(source="iracing", lap_time_s=float(df["time_s"].iloc[-1]),
                           is_valid=is_valid, is_pit=is_pit))
    assert is_valid_lap(lap)


def test_min_flying_speed_threshold_is_sane():
    # Below any real racing corner, above a pit/standing crawl.
    assert 5.0 < MIN_FLYING_SPEED_KPH < 40.0


@pytest.mark.parametrize("bad", [np.array([0.5]), np.array([])])
def test_split_handles_tiny_input(bad):
    assert split_lap_segments(bad, min_len=50) == []
