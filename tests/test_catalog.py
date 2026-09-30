"""Cascading-dropdown parsing helpers (offline, synthetic frames)."""

from __future__ import annotations

import pandas as pd

from lap_delta.catalog import driver_options, event_options, lap_options, session_options


def _schedule():
    return pd.DataFrame([
        {"RoundNumber": 0, "EventName": "Pre-Season Test", "EventFormat": "testing",
         "Session1": "Practice 1", "Session2": None, "Session3": None,
         "Session4": None, "Session5": None},
        {"RoundNumber": 2, "EventName": "Saudi Arabian Grand Prix", "EventFormat": "conventional",
         "Session1": "Practice 1", "Session2": "Practice 2", "Session3": "Practice 3",
         "Session4": "Qualifying", "Session5": "Race"},
        {"RoundNumber": 1, "EventName": "Bahrain Grand Prix", "EventFormat": "conventional",
         "Session1": "Practice 1", "Session2": "Practice 2", "Session3": "Practice 3",
         "Session4": "Qualifying", "Session5": "Race"},
    ])


def test_event_options_excludes_testing_and_sorts_by_round():
    opts = event_options(_schedule())
    rounds = [o["round"] for o in opts]
    assert rounds == [1, 2]                       # testing dropped, sorted ascending
    assert opts[0]["name"] == "Bahrain Grand Prix"
    assert opts[0]["label"] == "R1 — Bahrain Grand Prix"


def test_session_options_conventional():
    row = _schedule().iloc[2]  # Bahrain
    opts = session_options(row)
    assert opts == [
        ("Practice 1", "FP1"), ("Practice 2", "FP2"), ("Practice 3", "FP3"),
        ("Qualifying", "Q"), ("Race", "R"),
    ]


def test_session_options_sprint_format():
    row = pd.Series({
        "RoundNumber": 5, "EventName": "GP", "EventFormat": "sprint_shootout",
        "Session1": "Practice 1", "Session2": "Qualifying", "Session3": "Sprint Shootout",
        "Session4": "Sprint", "Session5": "Race",
    })
    ids = [sid for _, sid in session_options(row)]
    assert ids == ["FP1", "Q", "SS", "S", "R"]


def test_driver_options_parses_codes_names_colors():
    results = pd.DataFrame([
        {"Abbreviation": "VER", "FullName": "Max Verstappen", "TeamName": "Red Bull",
         "TeamColor": "3671C6"},
        {"Abbreviation": "NOR", "FullName": "Lando Norris", "TeamName": "McLaren",
         "TeamColor": "#FF8000"},
    ])
    opts = driver_options(results)
    assert [o["code"] for o in opts] == ["VER", "NOR"]
    assert opts[0]["color"] == "#3671C6"          # '#' prefixed when missing
    assert opts[1]["color"] == "#FF8000"          # already prefixed, unchanged
    assert opts[0]["name"] == "Max Verstappen"


def test_lap_options_filters_and_sorts_fastest_first():
    laps = pd.DataFrame([
        {"Driver": "VER", "LapNumber": 3, "LapTime": pd.Timedelta(seconds=80.5),
         "Deleted": False, "IsAccurate": True},
        {"Driver": "VER", "LapNumber": 5, "LapTime": pd.Timedelta(seconds=79.9),
         "Deleted": False, "IsAccurate": True},
        {"Driver": "VER", "LapNumber": 6, "LapTime": pd.NaT,
         "Deleted": False, "IsAccurate": True},         # no time -> dropped
        {"Driver": "VER", "LapNumber": 7, "LapTime": pd.Timedelta(seconds=78.0),
         "Deleted": True, "IsAccurate": True},           # deleted -> dropped
        {"Driver": "PER", "LapNumber": 4, "LapTime": pd.Timedelta(seconds=81.0),
         "Deleted": False, "IsAccurate": True},           # other driver
    ])
    opts = lap_options(laps, "VER")
    assert [o["lap_number"] for o in opts] == [5, 3]      # fastest first, filtered
    assert opts[0]["lap_time_s"] == 79.9
    assert opts[0]["label"].startswith("L5 — 1:19.900")
