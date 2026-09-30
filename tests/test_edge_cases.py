"""Edge cases across the pipeline: NaNs, distance wraparound, short/incomplete laps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lap_delta.align import compute_delta, make_distance_grid, resample_to_grid
from lap_delta.corners import detect_corners
from lap_delta.schema import CANONICAL_COLUMNS, SchemaError, validate_schema


def _frame(dist, **overrides) -> pd.DataFrame:
    """Minimal canonical-shaped frame from a distance array (channels default to sane values)."""
    n = len(dist)
    data = {
        "distance_m": np.asarray(dist, dtype=float),
        "time_s": np.linspace(0.0, 60.0, n),
        "speed_kph": np.full(n, 150.0),
        "throttle": np.full(n, 1.0),
        "brake": np.zeros(n),
        "gear": np.full(n, 5),
        "rpm": np.full(n, 9000.0),
        "x": np.zeros(n),
        "y": np.zeros(n),
    }
    data.update(overrides)
    return pd.DataFrame(data, columns=list(CANONICAL_COLUMNS))


# --- NaNs --------------------------------------------------------------------------

def test_nan_in_required_column_raises():
    df = _frame(np.linspace(0, 1000, 50))
    df.loc[10, "speed_kph"] = np.nan
    with pytest.raises(SchemaError, match="speed_kph"):
        validate_schema(df)


def test_nan_in_position_only_flagged_when_required():
    df = _frame(np.linspace(0, 1000, 50))
    df.loc[5, "x"] = np.nan
    assert validate_schema(df) is True                       # position optional by default
    with pytest.raises(SchemaError, match="position"):
        validate_schema(df, require_position=True)


# --- Distance wraparound (e.g. iRacing LapDistPct resetting) -----------------------

def test_wraparound_distance_fails_schema():
    dist = np.concatenate([np.linspace(0, 4000, 40), np.linspace(0, 1000, 10)])
    with pytest.raises(SchemaError, match="monotonic"):
        validate_schema(_frame(dist))


def test_resample_clamps_backward_blip_monotonic():
    # A small backward jitter must not break interpolation (align clamps with maximum.accumulate).
    dist = np.linspace(0, 3000, 60)
    dist[30] = dist[29] - 0.5
    grid = np.linspace(0, 3000, 100)
    out = resample_to_grid(_frame(dist), grid)
    assert np.all(np.diff(out["time_s"].to_numpy()) >= -1e-9)


# --- Short / incomplete laps -------------------------------------------------------

def test_single_sample_lap_rejected():
    with pytest.raises(SchemaError, match="at least 2 samples"):
        validate_schema(_frame([0.0]))


def test_two_sample_lap_valid_but_has_no_corners():
    df = _frame([0.0, 100.0])
    assert validate_schema(df) is True
    assert detect_corners(df) == []            # too few samples to detect anything


def test_flat_speed_lap_finds_no_corners():
    df = _frame(np.linspace(0, 3000, 300))     # constant speed -> no minima
    assert detect_corners(df) == []


def test_non_overlapping_laps_raise():
    early = _frame(np.linspace(0, 1000, 30))
    late = _frame(np.linspace(2000, 3000, 30))
    with pytest.raises(ValueError, match="overlap"):
        make_distance_grid(early, late)


def test_delta_on_partial_overlap_uses_reference_range():
    # Comparison covers a shorter distance span; delta still resolves on the reference grid.
    ref = _frame(np.linspace(0, 3000, 120))
    cmp = _frame(np.linspace(0, 2000, 80), time_s=np.linspace(0.0, 65.0, 80))
    delta = compute_delta(cmp, ref)
    assert len(delta) > 0
    assert np.isfinite(delta["delta_s"].to_numpy()).all()
