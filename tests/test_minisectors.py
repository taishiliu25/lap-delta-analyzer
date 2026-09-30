"""Adaptive mini-sector boundaries + SQLite store (offline, synthetic laps)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lap_delta.minisectors import (
    compute_edges_frac,
    get_or_compute,
    load_edges,
    save_edges,
    track_key,
)
from lap_delta.schema import Lap, LapMeta


def _profile_lap(*, fast_first=True, length=4000.0, track="TestTrack") -> Lap:
    """A lap that is fast on one half and slow on the other (for equal-time slicing tests)."""
    n = 1200
    d = np.linspace(0.0, length, n)
    fast, slow = 320.0, 90.0
    first, second = (fast, slow) if fast_first else (slow, fast)
    speed = np.where(d < length / 2, first, second)
    v = speed / 3.6
    t = np.cumsum(np.gradient(d) / v)
    t -= t[0]
    df = pd.DataFrame({
        "distance_m": d, "time_s": t, "speed_kph": speed,
        "throttle": np.ones(n), "brake": np.zeros(n), "gear": np.full(n, 6),
        "rpm": np.full(n, 11000.0), "x": np.zeros(n), "y": np.zeros(n),
    })
    return Lap(df, LapMeta(source="synthetic", track=track, lap_time_s=float(t[-1])))


def test_edges_span_full_lap_and_target_count():
    lap = _profile_lap()
    edges = compute_edges_frac(lap, target_seconds=5.0)
    assert edges[0] == 0.0 and edges[-1] == 1.0
    assert np.all(np.diff(edges) > 0)                      # strictly increasing
    expected_n = round(lap.meta.lap_time_s / 5.0)
    assert len(edges) - 1 == expected_n


def test_mini_sectors_longer_on_fast_half():
    # Equal-time slicing ⇒ the fast (first) half covers more distance per sector than the slow.
    edges = compute_edges_frac(_profile_lap(fast_first=True), target_seconds=5.0)
    assert (edges[1] - edges[0]) > (edges[-1] - edges[-2])


def test_store_round_trip(tmp_path):
    db = tmp_path / "ms.db"
    edges = np.array([0.0, 0.4, 0.75, 1.0])
    save_edges("K|1000", edges, track="K", length_m=1000.0, target_s=5.0, db_path=db)
    loaded = load_edges("K|1000", db_path=db)
    assert loaded is not None and np.allclose(loaded, edges)
    assert load_edges("missing", db_path=db) is None


def test_get_or_compute_caches_and_is_stable(tmp_path):
    db = tmp_path / "ms.db"
    lap = _profile_lap()
    edges1, n1, cached1 = get_or_compute(lap, db_path=db)
    assert cached1 is False
    edges2, n2, cached2 = get_or_compute(lap, db_path=db)
    assert cached2 is True                                 # served from the DB the 2nd time
    assert n1 == n2 and np.allclose(edges1, edges2)


def test_track_key_changes_with_layout_length():
    assert track_key(_profile_lap(length=4000.0)) != track_key(_profile_lap(length=5000.0))
