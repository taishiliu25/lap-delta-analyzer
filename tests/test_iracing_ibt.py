"""iRacing .ibt end-to-end (M7) — runs against any real .ibt the user provides.

The adapter's logic is covered offline by ``test_iracing_adapter.py`` (synthetic arrays).
This gated test exercises the *real* binary path: drop any iRacing ``.ibt`` export into
``data/sample/`` and run it. It self-skips when ``pyirsdk`` is absent or no ``.ibt`` is
present, so the default suite never needs the SDK or a binary fixture.

    pytest -m iracing
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lap_delta.align import compute_delta, total_delta
from lap_delta.laps import pick_reference, valid_laps
from lap_delta.schema import validate_schema

pytestmark = pytest.mark.iracing

pytest.importorskip("irsdk", reason="pyirsdk not installed")

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
IBT_FILES = sorted(SAMPLE_DIR.glob("*.ibt"))

if not IBT_FILES:
    pytest.skip("no .ibt in data/sample/ (drop one in to run these)", allow_module_level=True)


@pytest.fixture(params=IBT_FILES, ids=lambda p: p.name)
def laps(request):
    from lap_delta.adapters import IracingIbtAdapter

    return IracingIbtAdapter().load(str(request.param))


def test_parses_and_maps_to_schema(laps):
    """Every split segment is canonical: schema-valid, sane ranges, continuous brake."""
    assert laps, "expected at least one lap segment"
    for lap in laps:
        validate_schema(lap)
        s = lap.samples
        assert lap.meta.source == "iracing"
        assert s["speed_kph"].between(0, 400).all()
        assert s["throttle"].between(0, 1).all()
        assert s["brake"].between(0, 1).all()
        assert float(s["distance_m"].iloc[-1]) > 100


def test_delta_pipeline_runs_on_real_iracing_data(laps):
    """The canonical invariant holds for iRacing too: a lap vs itself is ~zero delta, and
    two real segments produce a finite, monotonic cumulative delta."""
    if len(laps) < 2:
        pytest.skip("need >= 2 segments to compare")

    reference = pick_reference(laps) if valid_laps(laps) else laps[0]
    assert abs(total_delta(compute_delta(reference, reference))) < 1e-6

    other = next(lp for lp in laps if lp is not reference)
    delta_df = compute_delta(other, reference)
    assert delta_df["delta_s"].notna().all()
    assert delta_df["distance_m"].is_monotonic_increasing
