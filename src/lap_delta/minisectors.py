"""Adaptive mini-sector boundaries, precomputed per track and cached in SQLite.

Mini-sectors are defined by **equal-time slicing** of a representative lap: cutting the lap
into ~5 s intervals makes each slice *longer on straights* (more distance covered per second)
and *shorter through slow corners*. Boundaries are stored as **fractions of lap distance**
(0–1) so they're robust to small per-lap distance differences, and cached per track layout so
they don't move when the driver or year changes (only when the track layout changes).

The store is a tiny SQLite database (stdlib ``sqlite3``); deleting it forces recomputation.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .schema import Lap

TARGET_SECONDS = 5.0
DEFAULT_DB = Path("data/minisectors.db")


def track_key(lap: Lap) -> str:
    """Stable key for a track *layout* — changes only if the lap length changes materially."""
    d = lap.samples["distance_m"].to_numpy(dtype=float)
    length = float(d[-1] - d[0])
    track = lap.meta.track or lap.meta.event or "unknown"
    return f"{track}|{round(length / 10) * 10}"


def compute_edges_frac(reference_lap: Lap, target_seconds: float = TARGET_SECONDS) -> np.ndarray:
    """Equal-time mini-sector edges as lap-distance fractions (includes 0.0 and 1.0)."""
    df = reference_lap.samples
    d = df["distance_m"].to_numpy(dtype=float)
    t = df["time_s"].to_numpy(dtype=float)
    lap_time = float(t[-1] - t[0])

    n = max(1, round(lap_time / target_seconds))
    target_t = np.linspace(t[0], t[-1], n + 1)
    edges_d = np.interp(target_t, t, d)

    span = d[-1] - d[0]
    frac = (edges_d - d[0]) / span if span > 0 else np.linspace(0.0, 1.0, n + 1)
    frac = np.clip(np.maximum.accumulate(frac), 0.0, 1.0)
    frac[0], frac[-1] = 0.0, 1.0
    return frac


# --- SQLite store ------------------------------------------------------------------

def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS minisector_boundaries (
            track_key TEXT PRIMARY KEY,
            track TEXT,
            length_m REAL,
            n INTEGER,
            target_s REAL,
            boundaries_json TEXT,
            created TEXT
        )
        """
    )
    return conn


def load_edges(key: str, db_path: str | Path = DEFAULT_DB) -> np.ndarray | None:
    """Return cached edge fractions for a track key, or ``None`` if not stored."""
    if not Path(db_path).exists():
        return None
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT boundaries_json FROM minisector_boundaries WHERE track_key = ?", (key,)
        ).fetchone()
    return np.asarray(json.loads(row[0]), dtype=float) if row else None


def save_edges(key: str, edges: np.ndarray, *, track: str, length_m: float,
               target_s: float, db_path: str | Path = DEFAULT_DB) -> None:
    """Persist edge fractions for a track key (upsert)."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO minisector_boundaries
                (track_key, track, length_m, n, target_s, boundaries_json, created)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(track_key) DO UPDATE SET
                track=excluded.track, length_m=excluded.length_m, n=excluded.n,
                target_s=excluded.target_s, boundaries_json=excluded.boundaries_json,
                created=excluded.created
            """,
            (key, track, float(length_m), len(edges) - 1, float(target_s),
             json.dumps([round(float(f), 6) for f in edges]),
             datetime.now(UTC).isoformat(timespec="seconds")),
        )


def get_or_compute(
    reference_lap: Lap,
    db_path: str | Path = DEFAULT_DB,
    target_seconds: float = TARGET_SECONDS,
) -> tuple[np.ndarray, int, bool]:
    """Return ``(edges_frac, n, from_cache)`` for the lap's track.

    Uses the cached boundaries when present; otherwise computes them from ``reference_lap``,
    persists them, and returns them. Cached boundaries are reused for any driver/year.
    """
    key = track_key(reference_lap)
    cached = load_edges(key, db_path)
    if cached is not None:
        return cached, len(cached) - 1, True

    edges = compute_edges_frac(reference_lap, target_seconds)
    d = reference_lap.samples["distance_m"].to_numpy(dtype=float)
    save_edges(key, edges, track=(reference_lap.meta.track or reference_lap.meta.event or ""),
               length_m=float(d[-1] - d[0]), target_s=target_seconds, db_path=db_path)
    return edges, len(edges) - 1, False
