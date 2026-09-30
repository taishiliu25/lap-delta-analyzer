"""Sector analysis: official 3 sectors + equal-distance mini-sectors.

Two complementary views of *where* the time goes:

* **Main sectors** — the official S1/S2/S3 timing gates (from :class:`LapMeta`, as reported
  by FastF1). When a source doesn't supply them, we fall back to splitting the lap into three
  equal-distance parts so the view still works for any telemetry.
* **Mini-sectors** — the lap's distance grid split into ``n`` equal slices; per slice we time
  each lap and mark the faster driver. This is the classic F1 "mini-sector dominance" split
  and it reconciles exactly with the cumulative delta.

The ``faster`` column uses ``"REF"`` / ``"CMP"`` (reference vs comparison); mapping those to
driver codes is left to the presentation layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import Lap


def _thirds_from_samples(df: pd.DataFrame) -> list[float]:
    """Sector times from equal-distance thirds (fallback when timing sectors are absent)."""
    d = df["distance_m"].to_numpy(dtype=float)
    t = df["time_s"].to_numpy(dtype=float)
    lo, hi = float(d[0]), float(d[-1])
    b1, b2 = lo + (hi - lo) / 3.0, lo + 2.0 * (hi - lo) / 3.0
    tb = np.interp([lo, b1, b2, hi], d, t)
    return [float(tb[1] - tb[0]), float(tb[2] - tb[1]), float(tb[3] - tb[2])]


def _sector_times(lap: Lap) -> list[float]:
    """Official S1/S2/S3 seconds from metadata, else the equal-thirds fallback."""
    m = lap.meta
    official = [m.sector1_s, m.sector2_s, m.sector3_s]
    if all(v is not None and np.isfinite(v) for v in official):
        return [float(v) for v in official]
    return _thirds_from_samples(lap.samples)


def main_sectors(reference: Lap, comparison: Lap) -> pd.DataFrame:
    """Per-sector (1/2/3) reference vs comparison times, delta, and faster driver.

    ``delta_s`` is ``cmp - ref`` (positive = comparison slower in that sector).
    """
    ref_t = _sector_times(reference)
    cmp_t = _sector_times(comparison)
    rows = []
    for i in range(3):
        delta = cmp_t[i] - ref_t[i]
        rows.append({
            "sector": i + 1,
            "ref_s": round(ref_t[i], 3),
            "cmp_s": round(cmp_t[i], 3),
            "delta_s": round(delta, 3),
            "faster": "CMP" if delta < 0 else "REF",
        })
    return pd.DataFrame(rows)


def main_sector_edges(reference: Lap, delta_df: pd.DataFrame) -> np.ndarray:
    """The 4 distance edges (on the ``delta_df`` grid) bounding the 3 official sectors.

    The S1/S2 boundaries are located from the reference lap's official sector *times*
    (find the distance where its cumulative time crosses S1 and S1+S2); with no timing gates
    the lap is split into equal-distance thirds. Returned on the delta grid so the boundaries
    line up with :func:`lap_delta.viz.sector_dominance_map`.
    """
    d = delta_df["distance_m"].to_numpy(dtype=float)
    lo, hi = float(d[0]), float(d[-1])
    m = reference.meta
    official = [m.sector1_s, m.sector2_s, m.sector3_s]
    if all(v is not None and np.isfinite(v) for v in official):
        rd = reference.samples["distance_m"].to_numpy(dtype=float)
        rt = reference.samples["time_s"].to_numpy(dtype=float)
        span = rd[-1] - rd[0]
        if span > 0:
            f1 = (float(np.interp(official[0], rt, rd)) - rd[0]) / span
            f2 = (float(np.interp(official[0] + official[1], rt, rd)) - rd[0]) / span
            return np.array([lo, lo + f1 * (hi - lo), lo + f2 * (hi - lo), hi])
    return np.array([lo, lo + (hi - lo) / 3.0, lo + 2.0 * (hi - lo) / 3.0, hi])


def mini_sectors(delta_df: pd.DataFrame, edges_frac) -> pd.DataFrame:
    """Time each lap through mini-sectors defined by ``edges_frac`` and mark the faster driver.

    ``edges_frac`` are lap-distance fractions in ``[0, 1]`` (including the 0.0 and 1.0
    endpoints), as produced by :func:`lap_delta.minisectors.get_or_compute`; they are scaled
    onto the reference distance grid so the same track boundaries apply to any pair of laps.
    Requires a ``delta_df`` from :func:`lap_delta.align.compute_delta` (has ``time_s_ref`` /
    ``time_s_cmp``). Returns one row per mini-sector with ``dist_start``, ``dist_end``,
    ``ref_s``, ``cmp_s``, ``delta_s`` (cmp - ref) and ``faster``; ``delta_s`` sums to the
    end-of-lap cumulative delta.
    """
    d = delta_df["distance_m"].to_numpy(dtype=float)
    t_ref = delta_df["time_s_ref"].to_numpy(dtype=float)
    t_cmp = delta_df["time_s_cmp"].to_numpy(dtype=float)
    frac = np.asarray(edges_frac, dtype=float)
    edges = d[0] + frac * (d[-1] - d[0])
    n = len(edges) - 1

    tr = np.interp(edges, d, t_ref)
    tc = np.interp(edges, d, t_cmp)

    rows = []
    for k in range(n):
        ref_s = float(tr[k + 1] - tr[k])
        cmp_s = float(tc[k + 1] - tc[k])
        delta = cmp_s - ref_s
        rows.append({
            "mini": k + 1,
            "dist_start": float(edges[k]),
            "dist_end": float(edges[k + 1]),
            "ref_s": round(ref_s, 3),
            "cmp_s": round(cmp_s, 3),
            "delta_s": round(delta, 4),
            "faster": "CMP" if delta < 0 else "REF",
        })
    return pd.DataFrame(rows)
