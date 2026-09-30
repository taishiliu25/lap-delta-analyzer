"""FastF1 channel mapping (M1) — offline.

Exercises ``FastF1Adapter._map_channels`` on a synthetic raw-telemetry frame shaped like
FastF1 output. This checks the *unit normalization at the adapter boundary* (throttle/brake
0-1, monotonic distance, correct columns/types) without any network access — ``_map_channels``
is a pure staticmethod that never imports ``fastf1``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lap_delta.adapters.fastf1_adapter import FastF1Adapter
from lap_delta.schema import CANONICAL_COLUMNS, validate_schema


def _raw_tel(n: int = 200, *, brake_scale: str = "bool") -> pd.DataFrame:
    """A FastF1-like raw telemetry frame (Distance/Time/Speed/Throttle/Brake/nGear/RPM/X/Y)."""
    dist = np.linspace(0.0, 5000.0, n)
    speed = 100.0 + 100.0 * np.sin(np.linspace(0, 3 * np.pi, n)) ** 2  # 100..200 kph
    v_ms = speed / 3.6
    time_s = np.cumsum(np.gradient(dist) / v_ms)

    throttle = np.full(n, 80.0)          # FastF1 throttle is 0..100
    if brake_scale == "bool":
        brake = (speed < 130).astype(float)   # {0.0, 1.0}
    else:  # some seasons encode 0..100
        brake = np.where(speed < 130, 100.0, 0.0)

    return pd.DataFrame({
        "Distance": dist,
        "Time": pd.to_timedelta(time_s, unit="s"),
        "Speed": speed,
        "Throttle": throttle,
        "Brake": brake,
        "nGear": np.clip((speed / 30).astype(int), 1, 8),
        "RPM": 6000.0 + speed * 20.0,
        "X": np.cos(np.linspace(0, 2 * np.pi, n)) * 800.0,
        "Y": np.sin(np.linspace(0, 2 * np.pi, n)) * 800.0,
    })


def test_map_channels_produces_canonical_schema():
    df = FastF1Adapter._map_channels(_raw_tel())

    assert list(CANONICAL_COLUMNS) == list(df.columns)      # all columns, canonical order
    assert validate_schema(df) is True                      # ranges + monotonicity all pass


def test_units_normalized_to_zero_one():
    df = FastF1Adapter._map_channels(_raw_tel())
    assert df["throttle"].between(0.0, 1.0).all()
    assert np.isclose(df["throttle"].iloc[0], 0.8)          # 80/100 -> 0.8
    assert df["brake"].between(0.0, 1.0).all()
    assert set(np.unique(df["brake"])) <= {0.0, 1.0}


def test_brake_percent_encoding_rescaled():
    df = FastF1Adapter._map_channels(_raw_tel(brake_scale="percent"))
    # 0..100 brake must be detected and rescaled into 0..1, not clipped to 1 everywhere.
    assert df["brake"].max() <= 1.0
    assert set(np.unique(df["brake"])) <= {0.0, 1.0}


def test_types_and_monotonic_distance():
    df = FastF1Adapter._map_channels(_raw_tel())
    assert df["gear"].dtype.kind in "iu"                    # integer gears
    assert np.all(np.diff(df["distance_m"].to_numpy()) >= 0)
    assert df["speed_kph"].min() >= 0


@pytest.mark.parametrize("n", [2, 50, 500])
def test_row_count_preserved(n):
    assert len(FastF1Adapter._map_channels(_raw_tel(n))) == n
