"""Schema validation (M0)."""

from __future__ import annotations

import pytest

from lap_delta.schema import SchemaError, validate_schema


def test_valid_lap_passes(ref_lap):
    assert validate_schema(ref_lap) is True


def test_missing_column_raises(ref_lap):
    ref_lap.samples = ref_lap.samples.drop(columns=["throttle"])
    with pytest.raises(SchemaError, match="missing canonical columns"):
        validate_schema(ref_lap)


def test_throttle_out_of_range_raises(ref_lap):
    ref_lap.samples.loc[0, "throttle"] = 2.0
    with pytest.raises(SchemaError, match="throttle"):
        validate_schema(ref_lap)


def test_non_monotonic_distance_raises(ref_lap):
    ref_lap.samples.loc[10, "distance_m"] = -500.0
    with pytest.raises(SchemaError, match="monotonic"):
        validate_schema(ref_lap)


def test_bare_dataframe_accepted(ref_lap):
    assert validate_schema(ref_lap.samples) is True


def test_new_meta_fields_default_none_and_dont_affect_validation(ref_lap):
    m = ref_lap.meta
    assert m.sector1_s is None and m.sector2_s is None and m.sector3_s is None
    assert m.team_color is None
    m.sector1_s, m.sector2_s, m.sector3_s, m.team_color = 25.0, 30.0, 20.0, "#3671C6"
    assert validate_schema(ref_lap) is True         # metadata never affects sample validation
