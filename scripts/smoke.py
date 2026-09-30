"""Full-pipeline smoke test on real FastF1 data (no Streamlit needed).

    python scripts/smoke.py

Loads 2023 Monza Qualifying (VER vs PER), validates the schema, computes the delta,
detects corners, prints the coaching table, and confirms both Plotly figures build.
Downloads + caches telemetry on first run (needs internet once).
"""

from __future__ import annotations

from lap_delta.adapters import FastF1Adapter
from lap_delta.align import compute_delta, total_delta
from lap_delta.coaching import coaching_table
from lap_delta.corners import annotate_time_loss, detect_corners
from lap_delta.laps import fastest_per_driver
from lap_delta.schema import format_laptime, validate_schema
from lap_delta.viz import traces_figure, track_map_figure


def main() -> None:
    laps = FastF1Adapter().load(year=2023, event="Monza", session="Q", drivers=["VER", "PER"])
    print(f"Loaded {len(laps)} laps")
    for lap in laps:
        validate_schema(lap)

    best = fastest_per_driver(laps)
    ref, cmp = best["VER"], best["PER"]
    print(f"VER best {format_laptime(ref.meta.lap_time_s)} | "
          f"PER best {format_laptime(cmp.meta.lap_time_s)}")

    delta = compute_delta(cmp, ref)
    print(f"Total delta (PER - VER): {total_delta(delta):+.3f} s over "
          f"{delta['distance_m'].iloc[-1]:.0f} m")

    corners = annotate_time_loss(detect_corners(ref), delta)
    print(f"Corners detected: {len(corners)}")

    table = coaching_table(ref, cmp, delta)
    print("\nCorner-by-corner (worst first):")
    print(table.to_string(index=False))

    fig_traces = traces_figure(delta, ref.meta.label, cmp.meta.label)
    fig_map = track_map_figure(delta)
    print(f"\nFigures built: {len(fig_traces.data)} trace-series, "
          f"{len(fig_map.data)} map-series")
    print("OK - full pipeline ran on real telemetry.")


if __name__ == "__main__":
    main()
