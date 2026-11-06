import math
import time
from dataclasses import dataclass
from typing import Any

import pytest

from backend.engine import RaceEngine, parse_gap, parse_time, is_drs_open, norm_brake
from backend.engine.parsing import as_list, deep_merge

try:
    from backend.ingest.messages import Msg
except Exception:  # ingest not landed yet
    @dataclass
    class Msg:
        topic: str
        data: Any
        utc: float


def m(topic, data, utc=1000.0):
    return Msg(topic, data, utc)


def test_parse_time():
    assert parse_time("1:32.123") == pytest.approx(92.123)
    assert parse_time("28.1") == pytest.approx(28.1)
    assert parse_time("1:02:03.5") == pytest.approx(3723.5)
    assert parse_time("") is None and parse_time(None) is None and parse_time("abc") is None
    assert parse_time("nan") is None


def test_parse_gap():
    assert parse_gap("+1.234") == ("+1.234", 1.234)
    assert parse_gap("1L") == ("1L", None)
    assert parse_gap("LAP 2") == ("LAP 2", None)
    assert parse_gap("") == ("", None)
    assert parse_gap(None) == ("", None)
    assert parse_gap({"Value": "+0.5"}) == ("+0.5", 0.5)


def test_misc_parsers():
    assert [is_drs_open(c) for c in (0, 8, 10, 12, 14, 9, "12", None)] == [False, False, True, True, True, False, True, False]
    assert norm_brake(0) == 0 and norm_brake(1) == 100 and norm_brake(100) == 100 and norm_brake(True) == 100
    assert as_list({"1": "b", "0": "a"}) == ["a", "b"]
    d = {"a": {"x": 1}}
    deep_merge(d, {"a": {"y": 2}, "b": 3})
    assert d == {"a": {"x": 1, "y": 2}, "b": 3}


def drivers_msg():
    return m("DriverList", {
        "1": {"RacingNumber": "1", "Tla": "VER", "FirstName": "Max", "LastName": "Verstappen",
              "TeamName": "Red Bull Racing", "TeamColour": "3671C6", "Line": 1},
        "44": {"RacingNumber": "44", "Tla": "HAM", "FirstName": "Lewis", "LastName": "Hamilton",
               "TeamName": "Mercedes", "TeamColour": "27F4D2", "Line": 2},
    })


def test_never_raises_on_garbage():
    e = RaceEngine()
    for bad in (None, 5, "x", [], {"Lines": 5}, {"Lines": {"1": 5}}, {"Lines": {"1": {"NumberOfLaps": "abc", "Sectors": 7}}}):
        for t in ("TimingData", "TimingAppData", "CarData", "Position", "RaceControlMessages", "TeamRadio", "WeatherData", "Nope"):
            e.apply(m(t, bad))
    e.apply(object())
    s = e.state()
    assert s["type"] == "state"


def test_state_basics_and_patch_merging():
    e = RaceEngine()
    e.apply(drivers_msg())
    e.apply(m("TimingData", {"Lines": {"1": {"Position": "1", "GapToLeader": "", "IntervalToPositionAhead": {"Value": ""},
                                            "LastLapTime": {"Value": "1:32.123", "PersonalFastest": True}, "NumberOfLaps": 5},
                                      "44": {"Position": "2", "GapToLeader": "+1.234", "IntervalToPositionAhead": {"Value": "+1.234"},
                                             "NumberOfLaps": 5}}}))
    e.apply(m("TimingData", {"Lines": {"44": {"GapToLeader": "+2.500"}}}))
    e.apply(m("TimingAppData", {"Lines": {"1": {"GridPos": "3"}, "44": {"GridPos": "1"}}}))
    e.apply(m("WeatherData", {"AirTemp": "28.1", "TrackTemp": "35.0", "Humidity": "40", "Pressure": "1010",
                              "Rainfall": "0", "WindDirection": "120", "WindSpeed": "2.3"}))
    e.apply(m("LapCount", {"CurrentLap": 6, "TotalLaps": 57}))
    e.apply(m("SessionStatus", {"Status": "Started"}, 1000.0))
    e.apply(m("SessionInfo", {"Meeting": {"Name": "Bahrain Grand Prix", "Country": {"Name": "Bahrain"}}, "Name": "Race", "Type": "Race"}))
    e.apply(m("TrackStatus", {"Status": "2", "Message": "Yellow"}))
    e.apply(m("ExtrapolatedClock", {"Utc": "1970-01-01T00:16:40Z", "Remaining": "01:30:00", "Extrapolating": True}, 1010.0))
    s = e.state()
    assert s["order"] == ["1", "44"]
    v, h = s["cars"]["1"], s["cars"]["44"]
    assert v["tla"] == "VER" and v["color"] == "#3671C6" and v["name"] == "Max Verstappen"
    assert v["pos"] == 1 and v["grid"] == 3 and v["posChange"] == 2
    assert h["posChange"] == -1
    assert v["gapLeader"] == "+0.000" and v["gapLeaderSec"] == 0.0 and v["intervalSec"] == 0.0
    assert h["gapLeaderSec"] == 2.5 and h["intervalSec"] == 1.234
    assert v["last"]["sec"] == pytest.approx(92.123) and v["last"]["personal"] is True and v["last"]["overall"] is False
    assert v["lap"] == 5
    assert len(v["sectors"]) == 3 and v["sectors"][0]["sec"] is None
    assert s["lap"] == {"current": 6, "total": 57}
    assert s["weather"]["air"] == 28.1 and s["weather"]["rain"] is False and s["weather"]["windDir"] == 120
    assert s["trackStatus"] == {"code": "2", "label": "Yellow", "flag": "yellow"}
    assert s["session"]["meeting"] == "Bahrain Grand Prix" and s["session"]["status"] == "Started"
    assert s["clock"]["remainingSec"] == pytest.approx(5400 - 10)
    assert s["clock"]["elapsedSec"] == pytest.approx(10.0)
    for key in ("utc", "session", "clock", "lap", "trackStatus", "weather", "order", "cars", "raceControl", "radio", "records", "pitStops"):
        assert key in s
    for key in ("num", "tla", "name", "team", "color", "pos", "grid", "posChange", "gapLeader", "interval", "gapLeaderSec",
                "intervalSec", "lap", "pits", "inPit", "pitOut", "retired", "stopped", "last", "best", "sectors",
                "speedTraps", "tyre", "stints", "tel", "xy", "progress"):
        assert key in v, key


def test_lapped_gap_and_missing_fields():
    e = RaceEngine()
    e.apply(m("TimingData", {"Lines": {"5": {"Position": "9", "GapToLeader": "1L", "IntervalToPositionAhead": {"Value": "LAP 2"}}}}))
    c = e.state()["cars"]["5"]
    assert c["gapLeader"] == "1L" and c["gapLeaderSec"] is None
    assert c["interval"] == "LAP 2" and c["intervalSec"] is None
    assert c["tyre"]["compound"] is None and c["xy"] is None and c["progress"] is None


def test_stints_list_and_dict():
    e = RaceEngine()
    e.apply(drivers_msg())
    e.apply(m("TimingAppData", {"Lines": {"1": {"Stints": [{"Compound": "SOFT", "New": "true", "TotalLaps": 0, "StartLaps": 0}]}}}))
    e.apply(m("TimingAppData", {"Lines": {"44": {"Stints": {"0": {"Compound": "MEDIUM", "New": "false", "StartLaps": 3}}}}}))
    s = e.state()["cars"]
    assert s["1"]["tyre"] == {"compound": "SOFT", "age": 0, "new": True, "stint": 1}
    assert s["44"]["tyre"]["compound"] == "MEDIUM" and s["44"]["tyre"]["new"] is False and s["44"]["tyre"]["age"] == 3


def lap_patch(num, n, sec, pos, s=None, extra=None):
    ln = {"NumberOfLaps": n, "LastLapTime": {"Value": sec}, "Position": str(pos)}
    if s:
        ln["Sectors"] = {str(i): {"Value": v} for i, v in enumerate(s)}
    if extra:
        ln.update(extra)
    return {"Lines": {num: ln}}


def test_lap_completion_history_and_fastest():
    e = RaceEngine()
    e.apply(drivers_msg())
    e.apply(m("TimingAppData", {"Lines": {"1": {"Stints": [{"Compound": "SOFT", "New": "true", "TotalLaps": 0}]}}}))
    e.apply(m("TimingData", {"Lines": {"1": {"Position": "1", "GapToLeader": "", "IntervalToPositionAhead": {"Value": ""}},
                                      "44": {"Position": "2", "GapToLeader": "+1.2", "IntervalToPositionAhead": {"Value": "+1.2"}}}}))
    e.apply(m("TimingData", lap_patch("1", 1, "1:35.000", 1, [28.0, 31.0, 36.0]), 1100))
    e.apply(m("TimingData", lap_patch("1", 2, "1:32.500", 1, [27.5, 30.0]), 1200))
    h = e.history()
    recs = h["laps"]["1"]
    assert [r["lap"] for r in recs] == [1, 2]
    assert recs[0]["sec"] == pytest.approx(95.0) and recs[0]["s"] == [28.0, 31.0, 36.0]
    assert recs[1]["s"][2] == pytest.approx(35.0)  # derived from lap - s1 - s2
    assert recs[1]["compound"] == "SOFT" and recs[1]["tyreAge"] == 2 and recs[1]["pos"] == 1
    assert recs[1]["gapLeader"] == 0.0 and recs[1]["pit"] is False and recs[1]["track"] == "1"
    assert h["maxLap"] == 2
    fl = e.state()["records"]["fastestLap"]
    assert fl["num"] == "1" and fl["sec"] == pytest.approx(92.5) and fl["lap"] == 2
    assert any(ev["kind"] == "fastest" for ev in h["events"])


def test_lap_time_arrives_after_lap_counter():
    e = RaceEngine()
    e.apply(drivers_msg())
    e.apply(m("TimingData", {"Lines": {"1": {"NumberOfLaps": 1}}}))
    assert e.history()["laps"]["1"][0]["sec"] is None
    e.apply(m("TimingData", {"Lines": {"1": {"LastLapTime": {"Value": "1:40.000"}}}}))
    assert e.history()["laps"]["1"][0]["sec"] == pytest.approx(100.0)
    # next lap, time first then counter
    e.apply(m("TimingData", {"Lines": {"1": {"LastLapTime": {"Value": "1:30.000"}}}}))
    e.apply(m("TimingData", {"Lines": {"1": {"NumberOfLaps": 2}}}))
    assert [r["sec"] for r in e.history()["laps"]["1"]] == [pytest.approx(100.0), pytest.approx(90.0)]


def test_pit_stop_and_track_status_in_history():
    e = RaceEngine()
    e.apply(drivers_msg())
    e.apply(m("TimingAppData", {"Lines": {"1": {"Stints": {"0": {"Compound": "SOFT", "New": "true"}}}}}))
    e.apply(m("TimingData", lap_patch("1", 1, "1:35.000", 1), 1100))
    e.apply(m("TrackStatus", {"Status": "4", "Message": "SCDeployed"}, 1110))
    e.apply(m("TimingData", {"Lines": {"1": {"InPit": True}}}, 1120))
    e.apply(m("TimingData", {"Lines": {"1": {"InPit": False, "PitOut": True, "NumberOfPitStops": 1}}}, 1142.4))
    e.apply(m("TimingAppData", {"Lines": {"1": {"Stints": {"1": {"Compound": "HARD", "New": "true", "StartLaps": 0}}}}}, 1143))
    e.apply(m("TrackStatus", {"Status": "1", "Message": "AllClear"}, 1150))
    e.apply(m("TimingData", lap_patch("1", 2, "2:10.000", 1), 1200))
    s = e.state()
    assert len(s["pitStops"]) == 1
    p = s["pitStops"][0]
    assert p["num"] == "1" and p["lap"] == 2 and p["durationSec"] == pytest.approx(22.4)
    assert p["fromCompound"] == "SOFT" and p["toCompound"] == "HARD"
    assert s["cars"]["1"]["pits"] == 1
    assert s["cars"]["1"]["tyre"]["compound"] == "HARD" and s["cars"]["1"]["tyre"]["stint"] == 2
    h = e.history()
    assert h["laps"]["1"][1]["pit"] is True and h["laps"]["1"][1]["track"] == "4"
    kinds = [ev["kind"] for ev in h["events"]]
    assert "pit" in kinds and "sc" in kinds
    assert len(s["pitStops"]) == 1  # NumberOfPitStops did not create a duplicate


def test_race_control_penalty_flags_and_dedupe():
    e = RaceEngine()
    e.apply(drivers_msg())
    msgs = [
        {"Utc": "2024-03-02T15:04:05Z", "Lap": 3, "Category": "Flag", "Flag": "YELLOW", "Scope": "Sector", "Sector": 4, "Message": "YELLOW IN TRACK SECTOR 4"},
        {"Utc": "2024-03-02T15:05:00Z", "Lap": 4, "Category": "Other", "Message": "FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 44 (HAM) - SPEEDING IN PIT LANE"},
        {"Utc": "2024-03-02T15:06:00Z", "Lap": 4, "Category": "Flag", "Flag": "BLUE", "Scope": "Driver", "RacingNumber": "1", "Message": "WAVED BLUE FLAG FOR CAR 1 (VER)"},
        {"Utc": "2024-03-02T15:07:00Z", "Lap": 5, "Category": "SafetyCar", "Message": "SAFETY CAR DEPLOYED"},
    ]
    e.apply(m("RaceControlMessages", {"Messages": msgs[:2]}))
    e.apply(m("RaceControlMessages", {"Messages": {"0": msgs[0], "1": msgs[1], "2": msgs[2], "3": msgs[3]}}))  # dict form + repeats
    s = e.state()
    assert len(s["raceControl"]) == 4
    assert s["raceControl"][0]["message"] == "SAFETY CAR DEPLOYED"  # newest first
    kinds = [(ev["kind"], ev["num"]) for ev in e.history()["events"]]
    assert ("penalty", "44") in kinds and ("sc", None) in kinds
    assert sum(1 for k, _ in kinds if k == "flag") == 1  # yellow only (blue skipped)
    assert any("HAM" in ev["text"] and "5 s" in ev["text"] for ev in e.history()["events"] if ev["kind"] == "penalty")
    from backend.engine import parse_iso
    e.apply(m("TrackStatus", {"Status": "4"}, parse_iso("2024-03-02T15:07:01Z")))
    assert sum(1 for ev in e.history()["events"] if ev["kind"] == "sc") == 1


def test_retire_and_radio():
    e = RaceEngine()
    e.apply(drivers_msg())
    e.apply(m("TimingData", {"Lines": {"44": {"Retired": True}}}))
    e.apply(m("TeamRadio", {"Captures": [{"Utc": "2024-03-02T15:04:05Z", "RacingNumber": "44", "Path": "2024/x/TeamRadio/HAM.mp3"}]}))
    e.apply(m("TeamRadio", {"Captures": {"0": {"Utc": "2024-03-02T15:04:05Z", "RacingNumber": "44", "Path": "2024/x/TeamRadio/HAM.mp3"}}}))
    s = e.state()
    assert s["cars"]["44"]["retired"] is True
    assert len(s["radio"]) == 1 and s["radio"][0]["url"].endswith("2024/x/TeamRadio/HAM.mp3") and s["radio"][0]["tla"] == "HAM"
    kinds = [ev["kind"] for ev in e.history()["events"]]
    assert "retire" in kinds and "radio" in kinds


def test_overtake_detection_not_pit_induced():
    e = RaceEngine()
    e.apply(drivers_msg())
    # lap 1: VER P1, HAM P2
    e.apply(m("TimingData", lap_patch("1", 1, "1:35.0", 1), 1100))
    e.apply(m("TimingData", lap_patch("44", 1, "1:35.5", 2), 1101))
    # lap 2: HAM passes VER on track
    e.apply(m("TimingData", {"Lines": {"1": {"Position": "2"}}}, 1190))
    e.apply(m("TimingData", lap_patch("44", 2, "1:34.0", 1), 1195))
    e.apply(m("TimingData", lap_patch("1", 2, "1:34.5", 2), 1196))
    ev = [x for x in e.history()["events"] if x["kind"] == "overtake"]
    assert len(ev) == 1 and ev[0]["num"] == "44" and "HAM passed VER" in ev[0]["text"]
    # lap 3: VER re-takes via pit (HAM pits)
    e.apply(m("TimingData", {"Lines": {"44": {"InPit": True}}}, 1250))
    e.apply(m("TimingData", {"Lines": {"1": {"Position": "1"}, "44": {"Position": "2"}}}, 1260))
    e.apply(m("TimingData", lap_patch("1", 3, "1:34.0", 1), 1290))
    ev = [x for x in e.history()["events"] if x["kind"] == "overtake"]
    assert len(ev) == 1


def test_cardata_position_and_fast_delta():
    e = RaceEngine()
    e.apply(drivers_msg())
    e.apply(m("CarData.z", {"Entries": [{"Utc": "2024-03-02T15:04:05Z", "Cars": {
        "1": {"Channels": {"0": 11500, "2": 312, "3": 7, "4": 100, "5": 0, "45": 12}},
        "44": {"Channels": {"0": 9000, "2": 120, "3": 3, "4": 0, "5": 1, "45": 0}}}}]}))
    d = e.fast_delta()
    assert d["type"] == "fast" and set(d["cars"]) == {"1", "44"}
    assert d["cars"]["1"]["drs"] is True and d["cars"]["1"]["speed"] == 312 and d["cars"]["1"]["brake"] == 0
    assert d["cars"]["44"]["brake"] == 100 and d["cars"]["44"]["drs"] is False
    assert e.fast_delta() is None
    e.apply(m("Position.z", {"Position": [{"Timestamp": "t", "Entries": {"1": {"Status": "OnTrack", "X": 10, "Y": -20, "Z": 5}}}]}))
    d = e.fast_delta()
    assert set(d["cars"]) == {"1"} and d["cars"]["1"]["x"] == 10 and "progress" not in d["cars"]["1"]
    s = e.state()
    assert s["cars"]["1"]["tel"]["drsRaw"] == 12 and s["cars"]["1"]["xy"] == {"x": 10.0, "y": -20.0, "z": 5.0, "status": "OnTrack"}
    assert s["records"]["topSpeed"] == {"num": "1", "kph": 312}


def test_track_built_from_positions_gives_progress_and_reset_keeps_track():
    e = RaceEngine()
    e.apply(drivers_msg())
    R = 5000.0
    n_per_lap = 200
    t = 1000.0
    for lap in range(3):
        for i in range(n_per_lap):
            a = 2 * math.pi * i / n_per_lap
            x, y = R * math.cos(a) * 1.5, R * math.sin(a)
            e.apply(m("Position", {"Position": [{"Timestamp": "t", "Entries": {"1": {"Status": "OnTrack", "X": x, "Y": y, "Z": 0}}}]}, t))
            t += 0.5
        e.apply(m("TimingData", lap_patch("1", lap + 1, "1:30.0", 1), t))
    assert e.track_dict() is not None
    tr = e.track_dict()
    assert 600 <= len(tr["points"]) <= 1000 and len(tr["bounds"]) == 4 and tr["length"] > 0
    assert tr["startIdx"] == 0 and len(tr["sectors"]) == 2
    a = math.pi / 2
    e.apply(m("Position", {"Position": [{"Timestamp": "t", "Entries": {"1": {"Status": "OnTrack", "X": 0, "Y": R, "Z": 0}}}]}, t))
    p = e.fast_delta()["cars"]["1"]["progress"]
    assert 0 <= p <= 1
    assert abs(((p - 0.25 + 0.5) % 1.0) - 0.5) < 0.03 or abs(((p - 0.75 + 0.5) % 1.0) - 0.5) < 0.03
    v = e.track_version
    e.reset()
    assert e.track_dict() is not None and e.track_version == v
    assert e.state()["order"] == [] and e.history()["maxLap"] == 0


def test_throughput():
    e = RaceEngine()
    e.apply(drivers_msg())
    cars = {str(i): {"Channels": {"0": 10000, "2": 250 + i, "3": 6, "4": 90, "5": 0, "45": 8}} for i in range(1, 21)}
    msgs = []
    for i in range(3000):
        msgs.append(m("CarData.z", {"Entries": [{"Utc": "t", "Cars": cars}]}, 1000 + i * 0.1))
        msgs.append(m("TimingData", {"Lines": {"1": {"GapToLeader": "+%d.0" % (i % 9), "Sectors": {"0": {"Value": "28.%d" % (i % 9)}}}}}, 1000 + i * 0.1))
    t0 = time.perf_counter()
    for x in msgs:
        e.apply(x)
    rate = len(msgs) / (time.perf_counter() - t0)
    assert rate > 5000, rate
