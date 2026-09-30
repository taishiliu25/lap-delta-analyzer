"""Lap selection, filtering and reference picking (M2)."""

from __future__ import annotations

from lap_delta.laps import fastest_per_driver, is_valid_lap, pick_reference, valid_laps


def _lap_set(make_lap):
    ref = make_lap(driver="VER", lap_number=10)                       # fastest, valid
    slow = make_lap(driver="PER", lap_number=11, speed_scale=0.97)    # valid, slower
    pit = make_lap(driver="VER", lap_number=1)
    pit.meta.is_pit = True
    pit.meta.is_out_lap = True
    deleted = make_lap(driver="PER", lap_number=5)
    deleted.meta.is_valid = False
    return ref, slow, pit, deleted


def test_valid_laps_excludes_pit_and_invalid(make_lap):
    ref, slow, pit, deleted = _lap_set(make_lap)
    result = valid_laps([ref, slow, pit, deleted])
    assert set(result) == {ref, slow}
    assert not is_valid_lap(pit)
    assert not is_valid_lap(deleted)


def test_pick_reference_returns_fastest(make_lap):
    ref, slow, pit, deleted = _lap_set(make_lap)
    assert pick_reference([slow, ref, pit, deleted]) is ref


def test_fastest_per_driver(make_lap):
    ref, slow, pit, deleted = _lap_set(make_lap)
    best = fastest_per_driver([ref, slow, pit, deleted])
    assert best["VER"] is ref
    assert best["PER"] is slow
