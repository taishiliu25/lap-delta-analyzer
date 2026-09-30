"""Adapter contract: every data source maps to the canonical schema here."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schema import Lap, validate_schema


class TelemetryAdapter(ABC):
    """Base class for telemetry sources. Subclasses map their raw data to :class:`Lap`."""

    #: Short source identifier stored in ``LapMeta.source``.
    source: str = "base"

    @abstractmethod
    def load(self, **kwargs) -> list[Lap]:
        """Load telemetry and return a list of canonical laps."""

    def validate(self, lap: Lap) -> Lap:
        """Assert a produced lap conforms to the canonical schema; return it unchanged."""
        validate_schema(lap)
        return lap
