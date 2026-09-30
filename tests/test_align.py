"""Distance-domain alignment and delta math (M3) — the correctness core."""

from __future__ import annotations

import numpy as np

from lap_delta.align import compute_delta, make_distance_grid, resample_to_grid, total_delta


def test_reference_vs_itself_is_zero(ref_lap):
    delta_df = compute_delta(ref_lap, ref_lap)
    assert np.abs(delta_df["delta_s"].to_numpy()).max() < 1e-6


def test_end_delta_equals_laptime_difference(ref_lap, slow_lap):
    delta_df = compute_delta(slow_lap, ref_lap)
    expected = slow_lap.meta.lap_time_s - ref_lap.meta.lap_time_s
    assert abs(total_delta(delta_df) - expected) < 1e-2
    # A uniformly slower lap should never be ahead at any point.
    assert delta_df["delta_s"].min() >= -1e-6


def test_grid_is_monotonic_increasing(ref_lap, slow_lap):
    grid = make_distance_grid(ref_lap, slow_lap, n=500)
    assert np.all(np.diff(grid) > 0)
    assert len(grid) == 500


def test_resample_preserves_channels(ref_lap):
    grid = make_distance_grid(ref_lap, n=200)
    out = resample_to_grid(ref_lap, grid)
    for col in ("time_s", "speed_kph", "throttle", "brake", "x", "y"):
        assert col in out.columns
    assert len(out) == 200


def test_resample_preserves_lap_time(ref_lap):
    # Interpolating onto the full-span grid must preserve the total lap time at the endpoint.
    grid = make_distance_grid(ref_lap, n=800)
    out = resample_to_grid(ref_lap, grid)
    assert abs(float(out["time_s"].iloc[-1]) - ref_lap.meta.lap_time_s) < 1e-3


def test_uniform_5pct_slower_delta_grows_steadily(make_lap):
    ref = make_lap(driver="VER")
    slow = make_lap(driver="PER", speed_scale=0.95)     # 5% slower everywhere
    delta_df = compute_delta(slow, ref)

    delta = delta_df["delta_s"].to_numpy()
    assert np.all(np.diff(delta) >= -1e-9)              # monotonic non-decreasing
    assert delta[0] == 0.0
    expected = slow.meta.lap_time_s - ref.meta.lap_time_s
    assert abs(total_delta(delta_df) - expected) < 1e-2
    assert expected > 0
