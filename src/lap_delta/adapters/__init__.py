"""Telemetry source adapters.

Importing these classes is cheap: ``fastf1`` / ``pyirsdk`` are imported lazily inside
the adapters, not at module import time.
"""

from __future__ import annotations

from .base import TelemetryAdapter
from .fastf1_adapter import FastF1Adapter
from .iracing_adapter import IracingIbtAdapter

__all__ = ["TelemetryAdapter", "FastF1Adapter", "IracingIbtAdapter"]
