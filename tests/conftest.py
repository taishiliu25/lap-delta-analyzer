"""Synthetic canonical-lap fixtures with known properties.

These let every analysis module be tested with zero real telemetry — the whole
pipeline is exercised deterministically, and a real FastF1 lap plugs into the same
schema later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lap_delta.schema import Lap, LapMeta

# Corners as (apex_distance_m, apex_speed_kph).
DEFAULT_CORNERS = ((1000.0, 80.0), (2000.0, 120.0))


def make_synthetic_lap(
    *,
    n: int = 1500,
    length: float = 3000.0,
    base_speed: float = 250.0,
    corners=DEFAULT_CORNERS,
    driver: str = "REF",
    lap_number: int = 5,
    speed_scale: float = 1.0,
    source: str = "synthetic",
) -> Lap:
    """Build a physically-plausible lap: Gaussian speed dips at each corner, brake
    before / throttle after each apex, a circular x/y trace, and time integrated from
    distance and speed. ``speed_scale`` < 1 makes a uniformly slower lap."""
    distance = np.linspace(0.0, length, n)
    speed = np.full(n, base_speed, dtype=float)
    throttle = np.ones(n, dtype=float)
    brake = np.zeros(n, dtype=float)

    for pos, vmin in corners:
        speed -= (base_speed - vmin) * np.exp(-0.5 * ((distance - pos) / 120.0) ** 2)
        braking = (distance > pos - 250.0) & (distance < pos)
        brake[braking] = 1.0
        throttle[braking] = 0.0
        accel = (distance >= pos) & (distance < pos + 250.0)
        throttle[accel] = 1.0

    speed = np.clip(speed * speed_scale, 30.0, None)
    v_ms = speed / 3.6
    dt = np.gradient(distance) / v_ms
    time_s = np.cumsum(dt)
    time_s -= time_s[0]

    theta = 2.0 * np.pi * distance / length
    radius = length / (2.0 * np.pi)

    samples = pd.DataFrame({
        "distance_m": distance,
        "time_s": time_s,
        "speed_kph": speed,
        "throttle": throttle,
        "brake": brake,
        "gear": np.clip((speed / 40.0).astype(int), 1, 8),
        "rpm": 5000.0 + speed * 40.0,
        "x": radius * np.cos(theta),
        "y": radius * np.sin(theta),
    })
    meta = LapMeta(
        source=source, driver=driver, lap_number=lap_number,
        lap_time_s=float(time_s[-1]), is_valid=True, sample_count=n,
    )
    return Lap(samples, meta)


@pytest.fixture
def make_lap():
    """Expose the synthetic-lap factory to tests."""
    return make_synthetic_lap


@pytest.fixture
def ref_lap():
    return make_synthetic_lap(driver="VER", lap_number=10)


@pytest.fixture
def slow_lap():
    # 3% slower everywhere -> a larger total lap time.
    return make_synthetic_lap(driver="PER", lap_number=11, speed_scale=0.97)
