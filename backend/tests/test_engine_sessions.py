"""Engine behaviour for practice / qualifying / sprint sessions + real-feed robustness (CONTRACT §6)."""
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Any

import pytest

from backend.engine import RaceEngine, parse_gap

try:
    from backend.ingest.messages import Msg
except Exception:  # pragma: no cover
    @dataclass
    class Msg:
        topic: str
        data: Any
        utc: float

T0 = 1_700_000_000.0


def m(topic, data, utc=T0):
    return Msg(topic, data, utc)


def drivers():
    return m("DriverList", {
        "1": {"RacingNumber": "1", "Tla": "VER", "FirstName": "Max", "LastName": "Verstappen", "TeamName": "Red Bull Racing", "TeamColour": "3671C6", "Line": 1},
        "4": {"RacingNumber": "4", "Tla": "NOR", "FirstName": "Lando", "LastName": "Norris", "TeamName": "McLaren", "TeamColour": "FF8000", "Line": 2},
        "44": {"RacingNumber": "44", "Tla": "HAM", "FirstName": "Lewis", "LastName": "Hamilton", "TeamName": "Ferrari", "TeamColour": "E80020", "Line": 3},
        "16": {"RacingNumber": "16", "Tla": "LEC", "FirstName": "Charles", "LastName": "Leclerc", "TeamName": "Ferrari", "TeamColour": "E80020", "Line": 4},
    })


def info(typ, name, utc=T0):
    return m("SessionInfo", {"Meeting": {"Name": "Bahrain Grand Prix", "Circuit": {"Key": 63}}, "Type": typ, "Name": name, "Number": 1}, utc)


def mk(typ, name):
    e = RaceEngine()
    e.apply(drivers())
    e.apply(info(typ, name))
    return e


def jsonable(e):
    for obj in (e.state(), e.history(), e.fast_delta() or {}):
        json.dumps(obj, allow_nan=False)


# ---------------------------------------------------------------- kind / part
@pytest.mark.parametrize("typ,name,kind", [
    ("Practice", "Practice 1", "practice"), ("practice", "PRACTICE 3", "practice"),
    ("Qualifying", "Qualifying", "qualifying"), ("Qualifying", "Sprint Qualifying", "sprint_qualifying"),
    ("Qualifying", "Sprint Shootout", "sprint_qualifying"), ("Race", "Sprint", "sprint"),
    ("Race", "Race", "race"), (None, None, "unknown"), ("Weird", "Thing", "unknown"),
])
def test_session_kind(typ, name, kind):
    e = RaceEngine()
    e.apply(info(typ, name))
    s = e.state()["session"]
    assert s["kind"] == kind
    assert s["part"] is None and s["partLabel"] is None


def test_unknown_kind_with_timediff_acts_as_nonrace():
    e = RaceEngine()
    e.apply(drivers())
    e.apply(m("TimingData", {"Lines": {"1": {"Position": "1", "TimeDiffToFastest": "", "BestLapTime": {"Value": "1:31.000", "Lap": 4}}}}))
    st = e.state()
    assert st["lap"]["current"] is None and st["lap"]["total"] is None
    assert st["session"]["kind"] == "practice"


# ---------------------------------------------------------------- practice
def test_practice_timediff_and_gaps_filled():
    e = mk("Practice", "Practice 1")
    e.apply(m("ExtrapolatedClock", {"Utc": "2023-11-14T22:13:20.000Z", "Remaining": "00:59:30", "Extrapolating": True}))
    e.apply(m("SessionStatus", {"Status": "Started"}))
    e.apply(m("TimingData", {"Lines": {
        "1": {"Position": "2", "TimeDiffToFastest": "+0.234", "TimeDiffToPositionAhead": "+0.234", "NumberOfLaps": 5,
              "BestLapTime": {"Value": "1:31.234", "Lap": 4}, "LastLapTime": {"Value": "1:31.9"}},
        "4": {"Position": "1", "TimeDiffToFastest": "", "TimeDiffToPositionAhead": "", "NumberOfLaps": 6,
              "BestLapTime": {"Value": "1:31.000", "Lap": 5}},
        "44": {"Position": "3", "TimeDiffToFastest": "+0.500", "TimeDiffToPositionAhead": "+0.266", "NumberOfLaps": 3,
               "BestLapTime": {"Value": "1:31.500", "Lap": 3}},
        "16": {"Position": "4", "InPit": True, "NumberOfLaps": 0},
    }}, T0 + 10))
    st = e.state()
    assert st["order"] == ["4", "1", "44", "16"]
    c = st["cars"]
    assert c["4"]["pos"] == 1 and c["4"]["gapLeader"] == "+0.000" and c["4"]["gapLeaderSec"] == 0.0
    assert c["1"]["gapLeader"] == "+0.234" and c["1"]["gapLeaderSec"] == pytest.approx(0.234)
    assert c["1"]["gapFastest"] == "+0.234" and c["1"]["gapFastestSec"] == pytest.approx(0.234)
    assert c["44"]["interval"] == "+0.266" and c["44"]["intervalSec"] == pytest.approx(0.266)
    assert c["1"]["posChange"] == 0 and c["1"]["grid"] is None
    assert c["1"]["timedLaps"] == 5
    assert st["lap"]["total"] is None and st["lap"]["current"] is None
    assert st["session"]["kind"] == "practice"
    assert st["records"]["fastestLap"]["num"] == "4" and st["records"]["fastestLap"]["sec"] == pytest.approx(91.0)
    assert st["clock"]["remainingSec"] == pytest.approx(30 - 10, abs=0.01) or st["clock"]["remainingSec"] > 0
    assert c["4"]["best"]["overall"] is True and c["1"]["best"]["overall"] is False
    jsonable(e)


def test_practice_gap_fallback_from_best_laps_when_feed_omits():
    e = mk("Practice", "Practice 2")
    e.apply(m("TimingData", {"Lines": {
        "1": {"Position": "1", "BestLapTime": {"Value": "1:30.000"}},
        "4": {"Position": "2", "BestLapTime": {"Value": "1:30.250"}},
        "44": {"Position": "3", "BestLapTime": {"Value": "1:31.000"}},
        "16": {"Position": "4"},
    }}))
    c = e.state()["cars"]
    assert c["1"]["gapFastest"] == "+0.000"
    assert c["4"]["gapFastest"] == "+0.250" and c["4"]["gapLeaderSec"] == pytest.approx(0.25)
    assert c["44"]["gapFastestSec"] == pytest.approx(1.0)
    assert c["44"]["intervalSec"] == pytest.approx(0.75) and c["44"]["interval"] == "+0.750"
    assert c["16"]["gapFastest"] == "" and c["16"]["gapFastestSec"] is None


def test_practice_no_overtakes_no_pitstops_and_garage_activity():
    e = mk("Practice", "Practice 1")
    e.apply(m("TimingData", {"Lines": {"1": {"Position": "1", "NumberOfLaps": 1, "InPit": False},
                                       "4": {"Position": "2", "NumberOfLaps": 1}}}, T0))
    e.apply(m("TimingData", {"Lines": {"1": {"InPit": True}}}, T0 + 5))
    st = e.state()
    assert st["cars"]["1"]["activity"] == "pit"
    assert st["pitStops"] == []
    e.apply(m("Heartbeat", {"Utc": "x"}, T0 + 120))
    assert e.state()["cars"]["1"]["activity"] == "garage"
    e.apply(m("TimingData", {"Lines": {"1": {"InPit": False, "PitOut": True}}}, T0 + 130))
    assert e.state()["cars"]["1"]["activity"] == "out"
    e.apply(m("TimingData", {"Lines": {"1": {"PitOut": False}}}, T0 + 140))
    assert e.state()["cars"]["1"]["activity"] == "track"
    # position swaps via lap-time improvement => no overtake events and no pit events
    for n in (2, 3, 4):
        e.apply(m("TimingData", {"Lines": {"4": {"Position": "1", "NumberOfLaps": n}, "1": {"Position": "2", "NumberOfLaps": n}}}, T0 + 150 + n))
    kinds = {ev["kind"] for ev in e.history()["events"]}
    assert "overtake" not in kinds and "pit" not in kinds
    jsonable(e)


def test_practice_in_out_laps_in_history_with_null_time():
    e = mk("Practice", "Practice 1")
    e.apply(m("TimingData", {"Lines": {"1": {"PitOut": True, "Position": "1"}}}, T0))
    e.apply(m("TimingData", {"Lines": {"1": {"PitOut": False}}}, T0 + 1))
    e.apply(m("TimingData", {"Lines": {"1": {"NumberOfLaps": 1}}}, T0 + 90))      # out lap, no LastLapTime
    e.apply(m("TimingData", {"Lines": {"1": {"LastLapTime": {"Value": "1:32.5"}, "NumberOfLaps": 2}}}, T0 + 180))
    e.apply(m("TimingData", {"Lines": {"1": {"InPit": True}}}, T0 + 250))
    e.apply(m("TimingData", {"Lines": {"1": {"NumberOfLaps": 3}}}, T0 + 260))      # in lap, no time
    laps = e.history()["laps"]["1"]
    assert [l["lap"] for l in laps] == [1, 2, 3]
    assert laps[0]["sec"] is None and laps[0]["pit"] is True
    assert laps[1]["sec"] == pytest.approx(92.5)
    assert laps[2]["sec"] is None and laps[2]["pit"] is True
    jsonable(e)


def test_fastest_event_when_p1_changes_hands():
    e = mk("Practice", "Practice 3")
    e.apply(m("TimingData", {"Lines": {"1": {"Position": "1", "BestLapTime": {"Value": "1:31.0", "Lap": 2}}}}, T0))
    e.apply(m("TimingData", {"Lines": {"4": {"Position": "1", "BestLapTime": {"Value": "1:30.5", "Lap": 3}},
                                       "1": {"Position": "2"}}}, T0 + 5))
    # a stale line for the previous P1 must not flip the headline back
    e.apply(m("TimingData", {"Lines": {"1": {"Sectors": {"0": {"Value": "28.0"}}}}}, T0 + 6))
    ev = [x for x in e.history()["events"] if x["kind"] == "fastest"]
    assert [x["num"] for x in ev] == ["1", "4"]
    assert "NOR" in ev[1]["text"] and "1:30.5" in ev[1]["text"]


# ---------------------------------------------------------------- qualifying
def quali_lines(part):
    return {"SessionPart": part, "Lines": {
        "1": {"Position": "1", "TimeDiffToFastest": "", "BestLapTime": {"Value": "1:29.000", "Lap": 3},
              "BestLapTimes": [{"Value": "1:29.000", "Lap": 3}, {"Value": "", "Lap": 0}, {"Value": ""}], "NumberOfLaps": 3},
        "4": {"Position": "2", "TimeDiffToFastest": "+0.100", "BestLapTime": {"Value": "1:29.100", "Lap": 2},
              "BestLapTimes": {"0": {"Value": "1:29.100", "Lap": 2}}, "NumberOfLaps": 2},
        "44": {"Position": "3", "TimeDiffToFastest": "+0.900", "BestLapTime": {"Value": "1:29.900", "Lap": 2}, "NumberOfLaps": 2},
        "16": {"Position": "4", "TimeDiffToFastest": "+1.500", "BestLapTime": {"Value": "1:30.500", "Lap": 2}, "NumberOfLaps": 2},
    }}


def test_quali_parts_knockout_cutoff_best_laps_and_red_flag():
    e = mk("Qualifying", "Qualifying")
    e.apply(m("SessionStatus", {"Status": "Started"}, T0))
    e.apply(m("TimingData", quali_lines(1), T0 + 1))
    st = e.state()
    assert st["session"]["kind"] == "qualifying" and st["session"]["part"] == 1 and st["session"]["partLabel"] == "Q1"
    assert st["cars"]["1"]["bestLaps"][0] == {"text": "1:29.000", "sec": pytest.approx(89.0)}
    assert st["cars"]["1"]["bestLaps"][1] == {"text": None, "sec": None}
    assert st["cars"]["4"]["bestLaps"][0]["sec"] == pytest.approx(89.1)
    # cutoff + knocked out
    e.apply(m("TimingData", {"CutOffTime": "00:00:00", "Lines": {"16": {"Cutoff": True}}}, T0 + 2))
    assert e.state()["cars"]["16"]["cutoff"] is True
    # red flag
    e.apply(m("SessionStatus", {"Status": "Aborted"}, T0 + 100))
    st = e.state()
    assert st["session"]["status"] == "Aborted" and st["trackStatus"]["flag"] == "red"
    assert any("RED" in x["text"] for x in e.history()["events"] if x["kind"] == "flag")
    e.apply(m("SessionStatus", {"Status": "Started"}, T0 + 200))
    assert e.state()["trackStatus"]["flag"] == "green"
    assert any("RESUMED" in x["text"] for x in e.history()["events"])
    e.apply(m("SessionStatus", {"Status": "Finished"}, T0 + 300))
    e.apply(m("TimingData", {"Lines": {"16": {"KnockedOut": True}, "44": {"KnockedOut": "false"}}}, T0 + 301))
    c = e.state()["cars"]
    assert c["16"]["knockedOut"] is True and c["44"]["knockedOut"] is False
    # Q2
    e.apply(m("SessionStatus", {"Status": "Started"}, T0 + 400))
    e.apply(m("TimingData", {"SessionPart": 2, "Lines": {"1": {"BestLapTime": {"Value": "1:28.5", "Lap": 2},
                                                         "BestLapTimes": {"1": {"Value": "1:28.500", "Lap": 2}}}}}, T0 + 410))
    st = e.state()
    assert st["session"]["partLabel"] == "Q2"
    assert st["cars"]["1"]["bestLaps"][0]["sec"] == pytest.approx(89.0)
    assert st["cars"]["1"]["bestLaps"][1]["sec"] == pytest.approx(88.5)
    assert st["records"]["fastestLap"]["sec"] == pytest.approx(88.5)
    e.apply(m("TimingData", {"SessionPart": 3}, T0 + 800))
    assert e.state()["session"]["partLabel"] == "Q3"
    assert e.state()["lap"] == {"current": None, "total": None}
    jsonable(e)


def test_quali_order_is_result_and_no_overtakes():
    e = mk("Qualifying", "Qualifying")
    e.apply(m("TimingData", quali_lines(1)))
    e.apply(m("TimingAppData", {"Lines": {"1": {"GridPos": "3"}}}))
    c = e.state()["cars"]
    assert c["1"]["posChange"] == 0 and c["1"]["grid"] is None
    e.apply(m("TimingData", {"Lines": {"4": {"Position": "1", "NumberOfLaps": 3}, "1": {"Position": "2", "NumberOfLaps": 4}}}))
    st = e.state()
    assert st["order"][:2] == ["4", "1"]
    assert [st["cars"][n]["pos"] for n in st["order"]] == [1, 2, 3, 4]
    assert not [x for x in e.history()["events"] if x["kind"] == "overtake"]


def test_cars_without_time_sort_after_and_ranks_unique():
    e = mk("Qualifying", "Qualifying")
    e.apply(m("TimingData", {"Lines": {"1": {"Position": "1", "BestLapTime": {"Value": "1:29.0"}},
                                       "4": {"Position": "1", "BestLapTime": {"Value": "1:28.0"}},
                                       "44": {}, "16": {}}}))
    st = e.state()
    assert st["order"][:2] == ["4", "1"]
    assert sorted(c["pos"] for c in st["cars"].values()) == [1, 2, 3, 4]


def test_sprint_quali_part_label_and_kind():
    e = mk("Qualifying", "Sprint Qualifying")
    e.apply(m("TimingData", quali_lines(2)))
    s = e.state()["session"]
    assert s["kind"] == "sprint_qualifying" and s["partLabel"] == "Q2"
    e2 = mk("Qualifying", "Sprint Shootout")
    assert e2.state()["session"]["kind"] == "sprint_qualifying"


# ---------------------------------------------------------------- sprint / race
def race_timing(lap, order):
    return {"Lines": {n: {"Position": str(i + 1), "NumberOfLaps": lap,
                          "GapToLeader": "" if i == 0 else "+%.3f" % (i * 1.5),
                          "IntervalToPositionAhead": {"Value": "" if i == 0 else "+1.500"},
                          "LastLapTime": {"Value": "1:30.%03d" % (100 + i)}} for i, n in enumerate(order)}}


def test_sprint_behaves_like_race():
    e = mk("Race", "Sprint")
    e.apply(m("LapCount", {"CurrentLap": 1, "TotalLaps": 19}))
    e.apply(m("TimingAppData", {"Lines": {"1": {"GridPos": "2"}, "4": {"GridPos": "1"}, "44": {"GridPos": "3"}, "16": {"GridPos": "4"}}}))
    e.apply(m("SessionStatus", {"Status": "Started"}))
    e.apply(m("TimingData", race_timing(1, ["4", "1", "44", "16"]), T0 + 1))
    e.apply(m("TimingData", race_timing(2, ["1", "4", "44", "16"]), T0 + 100))
    st = e.state()
    assert st["session"]["kind"] == "sprint"
    assert st["lap"]["total"] == 19
    assert st["cars"]["1"]["grid"] == 2 and st["cars"]["1"]["posChange"] == 1
    assert st["cars"]["1"]["gapFastest"] is None or st["cars"]["1"]["gapFastest"] == ""
    assert any(x["kind"] == "overtake" for x in e.history()["events"])


def test_race_pit_stop_not_counted_under_red_flag_but_is_in_green():
    e = mk("Race", "Race")
    e.apply(m("LapCount", {"CurrentLap": 5, "TotalLaps": 57}))
    e.apply(m("TimingData", {"Lines": {"1": {"Position": "1", "NumberOfLaps": 5}}}, T0))
    e.apply(m("TrackStatus", {"Status": "5", "Message": "Red"}, T0 + 1))
    e.apply(m("TimingData", {"Lines": {"1": {"InPit": True}}}, T0 + 2))
    e.apply(m("TimingData", {"Lines": {"1": {"InPit": False, "PitOut": True}}}, T0 + 400))
    assert e.state()["pitStops"] == []
    e.apply(m("TrackStatus", {"Status": "1", "Message": "AllClear"}, T0 + 410))
    e.apply(m("TimingData", {"Lines": {"1": {"InPit": True, "PitOut": False}}}, T0 + 500))
    e.apply(m("TimingData", {"Lines": {"1": {"InPit": False, "PitOut": True, "NumberOfPitStops": 1}}}, T0 + 522))
    ps = e.state()["pitStops"]
    assert len(ps) == 1 and ps[0]["durationSec"] == pytest.approx(22.0)


def test_race_leader_gap_lap_text_normalised():
    e = mk("Race", "Race")
    e.apply(m("TimingData", {"Lines": {"1": {"Position": "1", "GapToLeader": "LAP 7", "NumberOfLaps": 6},
                                       "4": {"Position": "2", "GapToLeader": "+2.2", "IntervalToPositionAhead": {"Value": "+2.2"}},
                                       "44": {"Position": "3", "GapToLeader": "1L", "IntervalToPositionAhead": {"Value": "1 LAP"}},
                                       "16": {"Position": "4", "GapToLeader": "NaN", "IntervalToPositionAhead": "OUT"}}}))
    c = e.state()["cars"]
    assert c["1"]["gapLeader"] == "+0.000"
    assert c["44"]["gapLeader"] == "1L" and c["44"]["gapLeaderSec"] is None and c["44"]["interval"] == "1 LAP"
    assert c["16"]["gapLeader"] == "" and c["16"]["interval"] == "OUT"
    jsonable(e)


# ---------------------------------------------------------------- robustness
def test_parse_gap_formats():
    assert parse_gap("+1.5") == ("+1.5", 1.5)
    assert parse_gap("NaN") == ("", None)
    assert parse_gap(float("nan")) == ("", None)
    assert parse_gap("OUT") == ("OUT", None)
    assert parse_gap("LAP 3") == ("LAP 3", None)
    assert parse_gap("1 LAP") == ("1 LAP", None)
    assert parse_gap("+1:02.500")[1] == pytest.approx(62.5)


def test_snapshot_json_string_and_empty_timestamp():
    e = RaceEngine()
    snap = json.dumps({"Lines": {"1": {"Position": "1", "GapToLeader": "LAP 3", "NumberOfLaps": 3,
                                       "Sectors": [{"Value": "28.1", "Segments": [{"Status": 2048}]}, {"Value": ""}, {"Value": ""}],
                                       "BestLapTime": {"Value": "1:30.1", "Lap": 2}, "Speeds": {"I1": {"Value": "310", "Status": 0}}}},
                       "Withheld": False})
    e.apply(Msg("DriverList", json.dumps({"1": {"RacingNumber": "1", "Tla": "VER"}, "_kf": True}), 0))
    e.apply(Msg("TimingData", snap, 0))
    st = e.state()
    assert list(st["cars"]) == ["1"] and st["cars"]["1"]["tla"] == "VER"
    assert st["cars"]["1"]["lap"] == 3 and st["cars"]["1"]["sectors"][0]["sec"] == pytest.approx(28.1)
    assert e.counts["errors"] == 0


def test_value_types_strings_bools_nulls():
    e = mk("Race", "Race")
    e.apply(m("TimingData", {"Lines": {"1": {"Position": 1, "NumberOfLaps": "7", "InPit": "false", "PitOut": "true",
                                             "Retired": "False", "Stopped": None, "NumberOfPitStops": "1",
                                             "LastLapTime": {"Value": "NaN", "OverallFastest": "true"},
                                             "BestLapTime": {"Value": None},
                                             "Sectors": {"0": {"Value": ""}, "1": {"Value": "abc"}, "2": None},
                                             "Speeds": {"I1": {"Value": ""}, "ST": {"Value": "320"}}, "_kf": True}}}))
    c = e.state()["cars"]["1"]
    assert c["lap"] == 7 and c["inPit"] is False and c["pitOut"] is True and c["retired"] is False
    assert c["last"]["sec"] is None and c["last"]["text"] == ""
    assert c["speedTraps"]["st"] == 320
    jsonable(e)


def test_position_and_cardata_duplicates_zeros_missing_cars():
    e = mk("Practice", "Practice 1")
    e.apply(m("Position", {"Position": [
        {"Timestamp": "2023-11-14T22:13:21.000Z", "Entries": {"1": {"Status": "OnTrack", "X": 100, "Y": 200, "Z": 5}, "4": {"X": 0, "Y": 0, "Z": 0}}},
        {"Timestamp": "2023-11-14T22:13:20.500Z", "Entries": {"1": {"Status": "OnTrack", "X": 999, "Y": 999, "Z": 5}}},   # older
        {"Timestamp": "2023-11-14T22:13:21.000Z", "Entries": {"99": {"X": 1, "Y": 1}, "abc": {"X": 1, "Y": 1}, "1": {"X": "nan", "Y": 3}}},
    ]}))
    c = e.state()["cars"]
    assert c["1"]["xy"]["x"] == 100 and c["4"]["xy"] is None
    assert "abc" not in c
    e.apply(m("CarData", {"Entries": [
        {"Utc": "2023-11-14T22:13:21.000Z", "Cars": {"1": {"Channels": {"0": 11000, "2": 300, "3": 7, "4": 100, "5": 0, "45": 12}}}},
        {"Utc": "2023-11-14T22:13:20.000Z", "Cars": {"1": {"Channels": {"2": 50}}, "2": {"Channels": {"2": 1}}, "x": {"Channels": {}}}},
    ]}))
    t = e.state()["cars"]["1"]["tel"]
    assert t["speed"] == 300 and t["drs"] is True and t["gear"] == 7
    assert "x" not in e.state()["cars"]
    fd = e.fast_delta()
    assert fd and fd["cars"]["1"]["speed"] == 300


def test_race_control_variants_and_num_extraction():
    e = mk("Race", "Race")
    msgs = [
        {"Utc": "2023-11-14T22:13:21", "Lap": 3, "Category": "Flag", "Flag": "YELLOW", "Scope": "Sector", "Sector": 4, "Message": "YELLOW IN TRACK SECTOR 4"},
        {"Utc": "2023-11-14T22:13:22", "Lap": 3, "Category": "Flag", "Flag": "BLUE", "Scope": "Driver", "RacingNumber": "4", "Message": "WAVED BLUE FLAG FOR CAR 4 (NOR)"},
        {"Utc": "2023-11-14T22:13:23", "Lap": 3, "Category": "Flag", "Flag": "BLACK AND WHITE", "Scope": "Driver", "RacingNumber": "44", "Message": "BLACK AND WHITE FLAG FOR CAR 44 (HAM) - TRACK LIMITS"},
        {"Utc": "2023-11-14T22:13:24", "Lap": 3, "Category": "Other", "Message": "FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 4 (NOR) FOR CAUSING A COLLISION"},
        {"Utc": "2023-11-14T22:13:25", "Lap": 3, "Category": "Other", "Message": "CAR 4 (NOR) NO FURTHER INVESTIGATION"},
        {"Utc": "2023-11-14T22:13:26", "Lap": 3, "Category": "Other", "Message": "FIA STEWARDS: NO PENALTY FOR CAR 1 (VER)"},
        {"Utc": "2023-11-14T22:13:27", "Lap": 3, "Category": "Other", "Message": "CAR 16 (LEC) TIME 1:23.456 DELETED - TRACK LIMITS AT TURN 4"},
        {"Utc": "2023-11-14T22:13:28", "Lap": 3, "Category": "Other", "Message": "FASTEST LAP 1:23.456 (VER)"},
        {"Utc": "2023-11-14T22:13:29", "Lap": 3, "Category": "Drs", "Flag": None, "Message": "DRS ENABLED"},
        {"Utc": "2023-11-14T22:13:30", "Lap": 3, "Category": "CarEvent", "Message": "CAR 12 STOPPED ON TRACK"},
        {"Utc": "2023-11-14T22:13:31", "Lap": 3, "Category": "SafetyCar", "Mode": "SAFETY CAR", "Status": "DEPLOYED", "Message": "SAFETY CAR DEPLOYED"},
        {"Utc": "2023-11-14T22:13:32", "Lap": 3, "Category": "Flag", "Flag": "DOUBLE YELLOW", "Scope": "Sector", "Sector": 7, "Message": "DOUBLE YELLOW IN TRACK SECTOR 7"},
        {"Utc": "2023-11-14T22:13:33", "Lap": 3, "Category": "Flag", "Flag": "CHEQUERED", "Scope": "Track", "Message": "CHEQUERED FLAG"},
        {"Utc": "2023-11-14T22:13:34", "Lap": 3, "Category": "Flag", "Flag": "RED", "Scope": "Track", "Message": "RED FLAG"},
        {"Utc": "2023-11-14T22:13:35", "Category": None, "Flag": None, "Message": None},
        "garbage", 5, None,
    ]
    e.apply(m("RaceControlMessages", {"Messages": msgs}))
    e.apply(m("RaceControlMessages", {"Messages": {"17": {"Utc": "2023-11-14T22:13:40", "Category": "Flag", "Flag": "GREEN", "Scope": "Track", "Message": "GREEN LIGHT - PIT EXIT OPEN"}}}))
    st = e.state()
    rc = st["raceControl"]
    assert rc[0]["message"].startswith("GREEN LIGHT")
    byflag = {x["message"]: x for x in rc if x["message"]}
    assert byflag["WAVED BLUE FLAG FOR CAR 4 (NOR)"]["num"] == "4"
    assert byflag["CAR 16 (LEC) TIME 1:23.456 DELETED - TRACK LIMITS AT TURN 4"]["num"] == "16"
    ev = e.history()["events"]
    pen = [x for x in ev if x["kind"] == "penalty"]
    assert len(pen) == 2 and pen[0]["num"] == "4" and "5 s penalty" in pen[0]["text"]
    assert "deleted" in pen[1]["text"] and pen[1]["num"] == "16"
    kinds = [x["kind"] for x in ev]
    assert kinds.count("sc") == 1 and "flag" in kinds
    assert not any("BLUE" in x["text"] for x in ev)
    jsonable(e)


def test_bounded_growth_long_session():
    e = mk("Practice", "Practice 1")
    for i in range(3000):
        e.apply(m("RaceControlMessages", {"Messages": [{"Utc": "2023-11-14T22:%02d:%02d" % (i // 60 % 60, i % 60), "Category": "Flag", "Flag": "YELLOW", "Message": "YELLOW %d" % i}]}, T0 + i))
        e.apply(m("TeamRadio", {"Captures": [{"Utc": "2023-11-14T22:00:00", "RacingNumber": "1", "Path": "x/%d.mp3" % i}]}, T0 + i))
        e.apply(m("CarData", {"Entries": [{"Utc": "2023-11-14T22:13:%02d.%03dZ" % (i % 60, i % 1000), "Cars": {"1": {"Channels": {"2": 200 + i % 100}}}}]}, T0 + i))
    for i in range(1000):
        e.apply(m("TimingData", {"Lines": {"1": {"NumberOfLaps": i, "InPit": i % 2 == 0, "PitOut": i % 3 == 0}}}, T0 + i))
    s = e.state()
    assert len(s["raceControl"]) == 100 and len(s["radio"]) == 30
    assert len(e.history()["events"]) <= 500
    assert len(e.rc) <= 400 and len(e.radio) <= 200
    assert len(e.drivers["1"].recs) <= 400
    assert e.pitstops == []
    jsonable(e)


def test_throughput_apply_fast():
    e = mk("Race", "Race")
    entry = {"Utc": "2023-11-14T22:13:21.000Z", "Cars": {n: {"Channels": {"0": 11000, "2": 300, "3": 7, "4": 100, "5": 0, "45": 12}} for n in ("1", "4", "44", "16")}}
    msg = m("CarData", {"Entries": [entry, entry]})
    t = time.perf_counter()
    for _ in range(5000):
        e.apply(msg)
    assert (time.perf_counter() - t) / 5000 < 0.002


# ---------------------------------------------------------------- fuzz
GARBAGE = [None, "", " ", "NaN", "nan", "inf", float("nan"), float("inf"), -float("inf"), True, False, 0, -1, 10**30, 1.5,
           "abc", "+1.5", "1L", "LAP 3", "OUT", "1:02.345", "true", "false", [], {}, [1, 2], {"Value": None}, {"Value": "NaN"},
           {"a": {"b": [None]}}, "2023-11-14T22:13:21Z", "\u0000", "💥"]
TOPICS = ["DriverList", "TimingData", "TimingAppData", "CarData", "CarData.z", "Position", "Position.z", "RaceControlMessages",
          "TrackStatus", "WeatherData", "LapCount", "SessionStatus", "SessionInfo", "ExtrapolatedClock", "TeamRadio",
          "Heartbeat", "SessionData", "TimingStats", "Nope"]
KEYS = ["Lines", "Position", "GapToLeader", "IntervalToPositionAhead", "TimeDiffToFastest", "TimeDiffToPositionAhead", "LastLapTime",
        "BestLapTime", "BestLapTimes", "Sectors", "Speeds", "InPit", "PitOut", "Retired", "Stopped", "NumberOfLaps", "NumberOfPitStops",
        "KnockedOut", "Cutoff", "Value", "Lap", "OverallFastest", "PersonalFastest", "SessionPart", "CutOffTime", "Stints", "Compound",
        "TotalLaps", "StartLaps", "New", "GridPos", "Entries", "Cars", "Channels", "0", "1", "2", "3", "4", "5", "45", "X", "Y", "Z",
        "Status", "Messages", "Category", "Flag", "Scope", "Sector", "Message", "Utc", "Timestamp", "Lap", "RacingNumber", "Captures",
        "Path", "CurrentLap", "Remaining", "Extrapolating", "Name", "Type", "Meeting", "Rainfall", "AirTemp", "Tla", "TeamColour",
        "Line", "FirstName", "_kf", "4", "44", "abc"]


def rnd_value(r, depth=0):
    c = r.random()
    if c < 0.45 or depth > 3:
        return r.choice(GARBAGE)
    if c < 0.8:
        return {r.choice(KEYS): rnd_value(r, depth + 1) for _ in range(r.randint(0, 4))}
    return [rnd_value(r, depth + 1) for _ in range(r.randint(0, 3))]


@pytest.mark.parametrize("seed", range(6))
def test_fuzz_never_raises_and_state_json_safe(seed):
    r = random.Random(seed)
    e = RaceEngine()
    e.apply(drivers())
    for i in range(1500):
        topic = r.choice(TOPICS)
        data = rnd_value(r)
        if r.random() < 0.5 and topic in ("TimingData", "TimingAppData"):
            data = {"Lines": {r.choice(["1", "4", "44", "x", 7]): {r.choice(KEYS): rnd_value(r) for _ in range(r.randint(0, 6))}}}
            if r.random() < 0.3:
                data[r.choice(KEYS)] = rnd_value(r)
        if r.random() < 0.1 and topic == "SessionInfo":
            data = {"Type": r.choice(["Practice", "Qualifying", "Race", None, 5]), "Name": r.choice(["Sprint", "Practice 1", "Sprint Qualifying", 7, None])}
        e.apply(Msg(topic, data, r.choice([T0 + i, 0, None, float("nan"), "x", T0 - 5])))
        if i % 100 == 0:
            jsonable(e)
    e.apply(object())
    e.apply(None)
    jsonable(e)
    json.dumps(e.stats(), allow_nan=False)
    e.reset()
    jsonable(e)


def test_state_valid_json_with_nan_injected_directly():
    e = mk("Practice", "Practice 1")
    e.apply(m("TimingData", {"Lines": {"1": {"Position": "1", "BestLapTime": {"Value": "1:30.0"}}}}))
    e.drivers["1"].best_sec = float("nan")
    e.drivers["4"].tdf_text, e.drivers["4"].tdf_sec = "+1", float("inf")
    e.weather["AirTemp"] = "nan"
    json.dumps(e.state(), allow_nan=False)


def test_timediff_alt_spelling_accepted():
    from backend.engine import RaceEngine
    from backend.ingest.messages import Msg
    e = RaceEngine()
    e.apply(Msg("DriverList", {"1": {"RacingNumber": "1", "Tla": "VER"}, "4": {"RacingNumber": "4", "Tla": "NOR"}}, 1.0))
    e.apply(Msg("TimingData", {"Lines": {
        "1": {"Position": "1", "BestLapTime": {"Value": "1:30.000"}, "TimeDiffToFastest": ""},
        "4": {"Position": "2", "BestLapTime": {"Value": "1:30.250"}, "TimeDiffToFastest": "+0.250",
              "TimeDifftoPositionAhead": "+0.250"}}}, 2.0))
    assert e.state()["cars"]["4"]["intervalSec"] == 0.25
