"""Smoke tests for the plotly figure builders (offline, synthetic data)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from lap_delta import viz
from lap_delta.align import compute_delta
from lap_delta.sectors import main_sector_edges, main_sectors, mini_sectors


@pytest.fixture
def delta_df(ref_lap, slow_lap):
    return compute_delta(slow_lap, ref_lap)


def _edges(n):
    return np.linspace(0.0, 1.0, n + 1)


def test_traces_figure(delta_df):
    fig = viz.traces_figure(delta_df, "VER L1", "PER L2", ref_color="#111", cmp_color="#222")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 6                       # speed/throttle/brake x2 + delta


def test_delta_gain_loss_figure_labels_drivers(delta_df):
    fig = viz.delta_gain_loss_figure(delta_df, "VER", "PER")
    assert isinstance(fig, go.Figure)
    assert "PER" in fig.layout.title.text and "VER" in fig.layout.title.text


def test_gear_rpm_figure(delta_df):
    fig = viz.gear_rpm_figure(delta_df, "VER", "PER")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 4


def test_minisector_dominance_map_has_numbers_boundaries_and_start_finish(delta_df):
    fig = viz.minisector_dominance_map(delta_df, mini_sectors(delta_df, _edges(12)), "VER", "PER")
    assert isinstance(fig, go.Figure)
    names = {tr.name for tr in fig.data}
    assert "mini #" in names          # per-mini-sector number labels
    assert "boundaries" in names      # gate lines slicing the map into mini-sectors
    assert "Start/Finish" in names    # S/F marker


def test_sector_dominance_map_matches_minisector_format(ref_lap, slow_lap, delta_df):
    main_df = main_sectors(ref_lap, slow_lap)  # 3 official-sector rows (thirds fallback here)
    edges = main_sector_edges(ref_lap, delta_df)
    sector_df = main_df.assign(dist_start=edges[:-1], dist_end=edges[1:])
    fig = viz.sector_dominance_map(delta_df, sector_df, "VER", "PER")
    names = {tr.name for tr in fig.data}
    assert "sector #" in names        # numbered like the mini map
    assert "boundaries" in names      # same gate slicing
    assert "Start/Finish" in names
    assert "dominance" in fig.layout.title.text.lower()


def test_teammate_colors_are_split(delta_df):
    # Same input color for both (teammates) -> comparison lightened so the two read apart.
    rc, cc = viz._colors("#3671C6", "#3671C6")
    assert rc.lower() != cc.lower()
    fig = viz.traces_figure(delta_df, "VER", "PER", ref_color="#3671C6", cmp_color="#3671C6")
    speed_colors = {tr.line.color for tr in fig.data if tr.line.color}
    assert len(speed_colors) >= 2     # not a single indistinguishable color


def test_track_map_shows_corner_numbers_and_start_finish(delta_df):
    d = delta_df["distance_m"].to_numpy()
    corners = [("1", float(d[len(d) // 4])), ("2", float(d[len(d) // 2]))]
    fig = viz.track_map_figure(delta_df, corners=corners)
    names = {tr.name for tr in fig.data}
    assert "Corner" in names and "Start/Finish" in names


def test_maps_degrade_without_position(delta_df):
    no_pos = delta_df.drop(columns=["x_cmp", "y_cmp"])
    tm = viz.track_map_figure(no_pos, corners=[("1", 500.0)])
    dom = viz.minisector_dominance_map(no_pos, mini_sectors(delta_df, _edges(8)), "VER", "PER")
    assert "unavailable" in tm.layout.title.text.lower()
    assert "unavailable" in dom.layout.title.text.lower()
