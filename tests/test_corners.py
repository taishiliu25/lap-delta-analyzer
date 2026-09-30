"""Corner detection and per-corner metrics (M4)."""

from __future__ import annotations

import numpy as np

from lap_delta.align import compute_delta
from lap_delta.corners import annotate_time_loss, detect_corners


def test_detects_expected_corner_count(ref_lap):
    corners = detect_corners(ref_lap)
    assert len(corners) == 2


def test_three_dips_found_at_right_distances(make_lap):
    apexes = ((800.0, 90.0), (1600.0, 70.0), (2400.0, 110.0))
    lap = make_lap(corners=apexes)
    corners = detect_corners(lap)

    assert len(corners) == 3
    found = sorted(c.apex_dist for c in corners)
    for got, (want, _) in zip(found, apexes, strict=True):
        assert abs(got - want) < 120.0


def test_apex_location_and_speed(ref_lap):
    corners = detect_corners(ref_lap)
    apex_dists = sorted(c.apex_dist for c in corners)
    assert abs(apex_dists[0] - 1000.0) < 150.0
    assert abs(apex_dists[1] - 2000.0) < 150.0

    by_dist = sorted(corners, key=lambda c: c.apex_dist)
    assert abs(by_dist[0].apex_speed_kph - 80.0) < 6.0
    assert abs(by_dist[1].apex_speed_kph - 120.0) < 6.0


def test_brake_before_apex_throttle_after(ref_lap):
    for c in detect_corners(ref_lap):
        assert c.brake_point_dist < c.apex_dist
        assert c.throttle_on_dist >= c.apex_dist - 1e-6


def test_time_loss_annotation_is_finite_and_sums(ref_lap, slow_lap):
    delta_df = compute_delta(slow_lap, ref_lap)
    corners = annotate_time_loss(detect_corners(ref_lap), delta_df)
    assert all(np.isfinite(c.time_delta_s) for c in corners)
    # Slower lap loses time in corners; per-corner losses are all non-negative here.
    assert all(c.time_delta_s >= -1e-3 for c in corners)
