"""FastF1 adapter end-to-end (M1) — network-gated.

Real telemetry hits the FastF1 API and downloads data, so this is skipped by default.
Enable it with:  LAPDELTA_E2E=1 pytest -m network
"""

from __future__ import annotations

import os

import pytest

from lap_delta.schema import validate_schema

pytestmark = pytest.mark.network

RUN_E2E = os.environ.get("LAPDELTA_E2E") == "1"


@pytest.mark.skipif(not RUN_E2E, reason="set LAPDELTA_E2E=1 to run the FastF1 network test")
def test_loads_monza_2023_qualifying(tmp_path):
    pytest.importorskip("fastf1")
    from lap_delta.adapters import FastF1Adapter

    laps = FastF1Adapter(cache_dir=tmp_path).load(
        year=2023, event="Monza", session="Q", drivers=["VER", "PER"],
    )
    assert laps, "expected at least one lap"

    drivers = {lap.meta.driver for lap in laps}
    assert {"VER", "PER"} <= drivers

    for lap in laps:
        validate_schema(lap)
        s = lap.samples
        assert s["speed_kph"].between(0, 400).all()
        assert s["throttle"].between(0, 1).all()
        assert s["brake"].between(0, 1).all()
        assert float(s["distance_m"].iloc[-1]) > 4000  # Monza lap ~5.79 km


@pytest.mark.slow
@pytest.mark.skipif(not RUN_E2E, reason="set LAPDELTA_E2E=1 to run the FastF1 network test")
def test_silverstone_delta_matches_laptime_gap(tmp_path):
    """End-to-end: real Silverstone 2023 Q, VER vs NOR fastest laps.

    The cumulative delta at the finish must match the real lap-time gap (telemetry time is
    anchored to the official lap time, so these should agree to a few tens of ms).
    """
    pytest.importorskip("fastf1")
    from lap_delta.adapters import FastF1Adapter
    from lap_delta.align import compute_delta, total_delta
    from lap_delta.laps import fastest_per_driver

    laps = FastF1Adapter(cache_dir=tmp_path).load(
        year=2023, event="Silverstone", session="Q", drivers=["VER", "NOR"],
    )
    best = fastest_per_driver(laps)
    assert {"VER", "NOR"} <= set(best)

    reference, comparison = best["VER"], best["NOR"]
    delta_df = compute_delta(comparison, reference)

    gap = comparison.meta.lap_time_s - reference.meta.lap_time_s
    assert abs(total_delta(delta_df) - gap) < 0.1

    # Official sector times are carried into LapMeta and roughly sum to the lap time.
    for lap in (reference, comparison):
        secs = [lap.meta.sector1_s, lap.meta.sector2_s, lap.meta.sector3_s]
        assert all(s is not None and s > 0 for s in secs)
        assert abs(sum(secs) - lap.meta.lap_time_s) < 0.5

    # Adaptive mini-sectors: boundaries compute + cache; deltas reconcile with the overall gap.
    from lap_delta.minisectors import get_or_compute
    from lap_delta.sectors import mini_sectors

    db = tmp_path / "ms.db"
    edges, n, cached = get_or_compute(reference, db_path=db)
    assert cached is False and n >= 5 and edges[0] == 0.0 and edges[-1] == 1.0
    assert get_or_compute(reference, db_path=db)[2] is True      # second call served from DB

    mini = mini_sectors(delta_df, edges)
    assert abs(mini["delta_s"].sum() - total_delta(delta_df)) < 0.05

    # Official corner numbers are available for the track map.
    corners = FastF1Adapter(cache_dir=tmp_path).circuit_corners(2023, "Silverstone", "Q")
    assert corners and all(c["distance_m"] > 0 for c in corners)
    assert {c["number"] for c in corners} >= {1, 2, 3}


@pytest.mark.slow
@pytest.mark.skipif(not RUN_E2E, reason="set LAPDELTA_E2E=1 to run the FastF1 network test")
def test_catalog_and_selected_load(tmp_path):
    """Light catalog methods list real events/drivers; load_selected builds specific laps."""
    pytest.importorskip("fastf1")
    from lap_delta.adapters import FastF1Adapter
    from lap_delta.catalog import driver_options, event_options

    adapter = FastF1Adapter(cache_dir=tmp_path)
    events = event_options(adapter.schedule(2023))
    assert any("Silverstone" in e["name"] or "British" in e["name"] for e in events)

    results, laps_df = adapter.session_summary(2023, "Silverstone", "Q")
    drivers = {d["code"] for d in driver_options(results)}
    assert {"VER", "NOR"} <= drivers

    ver_laps = laps_df[laps_df["Driver"] == "VER"]
    lap_no = int(ver_laps["LapNumber"].iloc[0])
    built = adapter.load_selected(2023, "Silverstone", "Q", [("VER", lap_no)])
    assert len(built) == 1
    assert built[0].meta.driver == "VER" and built[0].meta.lap_number == lap_no
