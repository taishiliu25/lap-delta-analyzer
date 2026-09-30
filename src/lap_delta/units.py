"""Display-time unit conversions.

The canonical schema is always metric/SI. These helpers convert *for display only*
(dashboard labels, coaching text) and never mutate stored values.
"""

from __future__ import annotations

import numpy as np

KPH_TO_MPH = 0.621371
M_TO_FT = 3.280840


def speed(value_kph, imperial: bool):
    """Convert km/h -> mph when ``imperial`` is set (scalar or array)."""
    return np.asarray(value_kph) * KPH_TO_MPH if imperial else value_kph


def distance(value_m, imperial: bool):
    """Convert meters -> feet when ``imperial`` is set (scalar or array)."""
    return np.asarray(value_m) * M_TO_FT if imperial else value_m


def speed_unit(imperial: bool) -> str:
    return "mph" if imperial else "km/h"


def distance_unit(imperial: bool) -> str:
    return "ft" if imperial else "m"
