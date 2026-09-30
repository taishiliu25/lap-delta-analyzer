"""Sector analysis: official 3 sectors + mini-sector reconciliation."""

from __future__ import annotations

import numpy as np

from lap_delta.align import compute_delta, total_delta
from lap_delta.sectors import main_sectors, mini_sectors


def test_mini_sectors_times_sum_to_lap_time(ref_lap, slow_lap):
    delta_df = compute_delta(slow_lap, ref_lap)
    mini = mini_sectors(delta_df, np.linspace(0.0, 1.0, 21))
    assert len(mini) == 20
    # Reference mini-sector times sum to the reference lap time across the grid.
    assert abs(mini["ref_s"].sum() - ref_lap.meta.lap_time_s) < 1e-2


def test_mini_sector_deltas_sum_to_total_delta(ref_lap, slow_lap):
    delta_df = compute_delta(slow_lap, ref_lap)
    mini = mini_sectors(delta_df, np.linspace(0.0, 1.0, 26))
    assert abs(mini["delta_s"].sum() - total_delta(delta_df)) < 5e-3


def test_uniformly_slower_lap_loses_every_mini_sector(make_lap):
    ref = make_lap(driver="VER")
    slow = make_lap(driver="PER", speed_scale=0.95)
    mini = mini_sectors(compute_delta(slow, ref), np.linspace(0.0, 1.0, 16))
    assert (mini["faster"] == "REF").all()          # comparison slower everywhere
    assert (mini["delta_s"] >= -1e-9).all()


def test_mini_sectors_respect_variable_edges(ref_lap, slow_lap):
    # Non-uniform edges (denser early) are honored: first mini-sector is the shortest.
    edges = np.array([0.0, 0.05, 0.2, 0.5, 1.0])
    mini = mini_sectors(compute_delta(slow_lap, ref_lap), edges)
    assert len(mini) == 4
    spans = mini["dist_end"].to_numpy() - mini["dist_start"].to_numpy()
    assert spans[0] == min(spans)


def test_main_sectors_from_official_metadata(make_lap):
    ref = make_lap(driver="VER")
    cmp = make_lap(driver="PER")
    ref.meta.sector1_s, ref.meta.sector2_s, ref.meta.sector3_s = 25.0, 30.0, 20.0
    cmp.meta.sector1_s, cmp.meta.sector2_s, cmp.meta.sector3_s = 25.5, 29.0, 21.0

    df = main_sectors(ref, cmp)
    assert list(df["sector"]) == [1, 2, 3]
    assert list(df["delta_s"]) == [0.5, -1.0, 1.0]
    assert list(df["faster"]) == ["REF", "CMP", "REF"]


def test_main_sectors_fallback_to_equal_thirds(ref_lap, slow_lap):
    # No official sector metadata -> equal-distance thirds; sectors sum to the lap time.
    df = main_sectors(ref_lap, slow_lap)
    assert len(df) == 3
    assert abs(df["ref_s"].sum() - ref_lap.meta.lap_time_s) < 1e-2
    assert np.isfinite(df["delta_s"].to_numpy()).all()
