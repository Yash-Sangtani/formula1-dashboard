"""Tests for backend.sim: determinism, raw topic shapes, internal consistency of a simulated race."""
from __future__ import annotations

import bisect
import json
import math
from collections import Counter, defaultdict
from typing import Dict, List

import pytest

from backend.ingest.messages import Msg, as_list, decode_z, deep_merge, parse_line, to_raw_line
from backend.sim import SimConfig, SimRace, generate, generate_messages, get_track, list_tracks, record_to_file
from backend.sim.tracks import Track

LAPS = 15


@pytest.fixture(scope="module")
def race() -> SimRace:
    return generate(SimConfig(seed=1, laps=LAPS))


@pytest.fixture(scope="module")
def msgs(race) -> List[Msg]:
    return race.messages()


@pytest.fixture(scope="module")
def track(race) -> Track:
    return race.track


@pytest.fixture(scope="module")
def long_race() -> SimRace:
    return generate(SimConfig(seed=3, laps=57))


def by_topic(msgs, topic):
    return [m for m in msgs if m.topic == topic]


# ----------------------------------------------------------------------------- determinism
def test_deterministic_same_seed():
    a = generate(SimConfig(seed=5, laps=4))
    b = generate(SimConfig(seed=5, laps=4))
    assert len(a) == len(b)
    assert [to_raw_line(m) for m in a] == [to_raw_line(m) for m in b]
    assert a.final_order == b.final_order


def test_different_seed_differs():
    a = generate(SimConfig(seed=5, laps=4))
    b = generate(SimConfig(seed=6, laps=4))
    assert [to_raw_line(m) for m in a][:400] != [to_raw_line(m) for m in b][:400] or a.final_order != b.final_order


def test_messages_are_time_ordered_and_lazy(race, msgs):
    t = [m.utc for m in msgs]
    assert t == sorted(t)
    assert t[0] == race.cfg.base_utc
    assert len(race) == len(msgs) == len(race.times) == len(race.topics)
    assert race.get(10).data == msgs[10].data
    # telemetry is stored compactly: one array row per tick, not dict objects
    assert len(race.cd_rows) == len(race.pos_rows) == Counter(race.topics)["CarData"]


# ----------------------------------------------------------------------------- shapes
def test_all_contract_topics_present(msgs):
    topics = {m.topic for m in msgs}
    need = {"DriverList", "SessionInfo", "SessionStatus", "LapCount", "TimingData", "TimingAppData", "CarData",
            "Position", "RaceControlMessages", "TrackStatus", "WeatherData", "ExtrapolatedClock", "TeamRadio"}
    assert need <= topics
    assert not any(m.topic.endswith(".z") for m in msgs)


def test_driver_list_shape(msgs):
    dl = by_topic(msgs, "DriverList")
    assert len(dl) == 1
    d = dl[0].data
    assert len(d) == 20
    assert len({v["Tla"] for v in d.values()}) == 20
    for num, v in d.items():
        assert v["RacingNumber"] == num
        assert len(v["TeamColour"]) == 6 and int(v["TeamColour"], 16) >= 0
        assert {"Tla", "FirstName", "LastName", "TeamName", "Line"} <= v.keys()
    assert len({v["TeamName"] for v in d.values()}) == 10
    assert d["1"]["TeamColour"] == "3671C6"


def test_session_info_and_status_flow(msgs):
    si = by_topic(msgs, "SessionInfo")[0].data
    assert si["Type"] == "Race" and si["Name"] == "Race" and si["Meeting"]["Name"] and si["Meeting"]["Country"]["Name"]
    st = [m.data["Status"] for m in by_topic(msgs, "SessionStatus")]
    assert st == ["Inactive", "Started", "Finished", "Finalised", "Ends"]
    assert by_topic(msgs, "LapCount")[0].data == {"CurrentLap": 1, "TotalLaps": LAPS}
    laps = [m.data["CurrentLap"] for m in by_topic(msgs, "LapCount")]
    assert laps == sorted(laps) and laps[-1] == LAPS


def test_timing_data_shape(msgs):
    seen_keys = set()
    for m in by_topic(msgs, "TimingData"):
        assert set(m.data) == {"Lines"}
        for num, ln in m.data["Lines"].items():
            assert num.isdigit()
            seen_keys |= set(ln)
            if "Position" in ln:
                assert isinstance(ln["Position"], str) and ln["Position"].isdigit()
            if "LastLapTime" in ln:
                v = ln["LastLapTime"]
                assert isinstance(v["Value"], str) and ":" in v["Value"]
                assert isinstance(v["OverallFastest"], bool) and isinstance(v["PersonalFastest"], bool)
            if "Sectors" in ln:
                for k, s in ln["Sectors"].items():
                    assert k in ("0", "1", "2") and "." in s["Value"]
            if "IntervalToPositionAhead" in ln:
                assert set(ln["IntervalToPositionAhead"]) >= {"Value"}
            if "Speeds" in ln:
                assert set(ln["Speeds"]) <= {"I1", "I2", "FL", "ST"}
    assert {"Position", "GapToLeader", "IntervalToPositionAhead", "LastLapTime", "BestLapTime", "Sectors", "NumberOfLaps",
            "NumberOfPitStops", "InPit", "PitOut", "Retired", "Stopped", "Speeds"} <= seen_keys


def test_timing_app_data_shape(msgs):
    first = by_topic(msgs, "TimingAppData")[0].data["Lines"]
    assert len(first) == 20
    for num, ln in first.items():
        assert ln["GridPos"].isdigit()
        st = ln["Stints"]["0"]
        assert st["Compound"] in ("SOFT", "MEDIUM", "HARD") and st["New"] in ("true", "false")
        assert st["TotalLaps"] == 0 and st["StartLaps"] == 0
    assert sorted(int(l["GridPos"]) for l in first.values()) == list(range(1, 21))


def test_race_control_messages_shapes_and_variety(msgs):
    entries = []
    n_list = n_dict = 0
    for m in by_topic(msgs, "RaceControlMessages"):
        v = m.data["Messages"]
        n_list += isinstance(v, list)
        n_dict += isinstance(v, dict)
        entries += as_list(v)
    assert n_list == 1 and n_dict >= 5  # snapshot list first, then index-keyed patches (both forms the engine must accept)
    cats = Counter(e["Category"] for e in entries)
    assert {"Flag", "Other", "Drs"} <= set(cats)
    flags = {e.get("Flag") for e in entries if e["Category"] == "Flag"}
    assert {"YELLOW", "GREEN", "CHEQUERED"} <= flags
    for e in entries:
        assert {"Utc", "Lap", "Category", "Message"} <= e.keys() and 1 <= e["Lap"] <= LAPS
    text = " ".join(e["Message"] for e in entries)
    assert "DRS ENABLED" in text and "FASTEST LAP" in text and "PENALTY" in text and "CHEQUERED FLAG" in text
    assert "SAFETY CAR" in text.upper()


def test_track_status_sequence(msgs):
    codes = [m.data["Status"] for m in by_topic(msgs, "TrackStatus")]
    assert codes[0] == "1" and codes[-1] == "1"
    assert set(codes) <= set("1245667")
    assert "2" in codes and ("4" in codes or "6" in codes)
    for m in by_topic(msgs, "TrackStatus"):
        assert m.data["Message"]
    # a deployed SC/VSC period always ends back in AllClear
    i = max(codes.index(c) for c in ("4", "6") if c in codes)
    assert "1" in codes[i:]


def test_weather_drifts(msgs):
    w = by_topic(msgs, "WeatherData")
    assert len(w) >= 5
    assert all(isinstance(v, str) for v in w[0].data.values())
    assert {"AirTemp", "Humidity", "Pressure", "Rainfall", "TrackTemp", "WindDirection", "WindSpeed"} == set(w[0].data)
    assert len({m.data["TrackTemp"] for m in w}) > 3
    assert all(10 < float(m.data["AirTemp"]) < 50 for m in w)


def test_extrapolated_clock_and_radio(msgs):
    clocks = by_topic(msgs, "ExtrapolatedClock")
    assert clocks[0].data["Extrapolating"] is False and clocks[1].data["Extrapolating"] is True
    assert clocks[-1].data["Extrapolating"] is False
    rem = [m.data["Remaining"] for m in clocks]
    assert all(len(r) == 8 and r[2] == ":" for r in rem)
    caps = []
    for m in by_topic(msgs, "TeamRadio"):
        caps += as_list(m.data["Captures"])
    assert len(caps) >= 3
    for c in caps:
        assert c["RacingNumber"].isdigit() and c["Path"].startswith("TeamRadio/") and c["Path"].endswith(".mp3")


# ----------------------------------------------------------------------------- .z payloads
def test_z_topics_encode_and_decode(msgs):
    for m in by_topic(msgs, "CarData")[:50] + by_topic(msgs, "Position")[:50]:
        line = to_raw_line(m, encode=True)
        assert line.startswith("['%s.z', '" % m.topic)
        back = parse_line(line)
        assert back.topic == m.topic and back.data == m.data
        raw = to_raw_line(m, encode=False)
        assert raw.startswith("['%s', {" % m.topic)
    cd = by_topic(msgs, "CarData")[100].data
    assert list(cd) == ["Entries"] and len(cd["Entries"]) == 1
    e = cd["Entries"][0]
    assert e["Utc"].endswith("Z") and len(e["Cars"]) == 20
    assert set(next(iter(e["Cars"].values()))["Channels"]) == {"0", "2", "3", "4", "5", "45"}
    pe = by_topic(msgs, "Position")[100].data["Position"][0]
    assert pe["Timestamp"].endswith("Z") and set(next(iter(pe["Entries"].values()))) == {"Status", "X", "Y", "Z"}


def test_telemetry_rate_is_4hz(msgs):
    t = [m.utc for m in by_topic(msgs, "CarData")]
    d = [b - a for a, b in zip(t, t[1:])]
    assert max(d) == pytest.approx(0.25, abs=1e-6) and min(d) == pytest.approx(0.25, abs=1e-6)
    assert [m.utc for m in by_topic(msgs, "Position")] == t


# ----------------------------------------------------------------------------- replay: state consistency
def replay_timing(msgs):
    """Replay TimingData like the engine would; yields (msg, state) after each patch."""
    state: Dict = {}
    for m in msgs:
        if m.topic == "TimingData":
            deep_merge(state, m.data)
            yield m, state


def test_positions_are_always_a_permutation(msgs):
    n = 0
    for m, st in replay_timing(msgs):
        if any("Position" in l for l in m.data["Lines"].values()):
            pos = sorted(int(l["Position"]) for l in st["Lines"].values())
            assert pos == list(range(1, 21)), (m.utc, pos)
            n += 1
    assert n > 50


def test_final_classification_and_lap_counts(race, msgs):
    final = None
    for _, final in replay_timing(msgs):
        pass
    lines = final["Lines"]
    order = [n for n, _ in sorted(lines.items(), key=lambda kv: int(kv[1]["Position"]))]
    assert order == race.final_order
    winner = lines[order[0]]
    assert winner["NumberOfLaps"] == LAPS and winner["GapToLeader"] == ""
    classified = [n for n in order if not lines[n].get("Retired")]
    laps = [lines[n]["NumberOfLaps"] for n in classified]
    assert laps == sorted(laps, reverse=True) and laps[0] == LAPS
    retired = [n for n in order if lines[n].get("Retired")]
    assert retired and order[-len(retired):] == [n for n in order if n in retired]  # retirees classified last
    assert all(lines[n]["NumberOfLaps"] < LAPS for n in retired)
    assert all(lines[n].get("Stopped") for n in retired)


def test_final_gaps_monotonic(msgs):
    final = None
    for _, final in replay_timing(msgs):
        pass
    lines = final["Lines"]
    order = sorted((l for l in lines.values() if not l.get("Retired")), key=lambda l: int(l["Position"]))
    gaps = []
    for l in order[1:]:
        g = l["GapToLeader"]
        assert g.startswith("+"), g
        gaps.append(float(g[1:]))
        iv = float(l["IntervalToPositionAhead"]["Value"][1:])
        assert 0 <= iv <= gaps[-1] + 1e-6
    assert gaps == sorted(gaps)
    assert order[0]["GapToLeader"] == "" and order[0]["IntervalToPositionAhead"]["Value"] == ""
    # for the cars on the lead lap the intervals telescope to the gap to the leader (same loop crossing)
    lead_lap = [l for l in order if l["NumberOfLaps"] == order[0]["NumberOfLaps"]]
    ivs = [float(l["IntervalToPositionAhead"]["Value"][1:]) for l in lead_lap[1:]]
    assert sum(ivs) == pytest.approx(float(lead_lap[-1]["GapToLeader"][1:]), abs=0.025)


def test_intervals_and_gaps_consistent_on_every_update(msgs):
    """Whenever a car reports gaps they are sane: 0 <= interval <= gap-to-leader (or lapped markers)."""
    checked = 0
    for m in by_topic(msgs, "TimingData"):
        for ln in m.data["Lines"].values():
            g, iv = ln.get("GapToLeader"), ln.get("IntervalToPositionAhead", {}).get("Value")
            if g and iv and g[0] == "+" and iv[0] == "+":
                assert 0 <= float(iv[1:]) <= float(g[1:]) + 1e-6
                checked += 1
    assert checked > 500


def test_fastest_flags_consistent(msgs):
    best = 1e9
    personal: Dict[str, float] = defaultdict(lambda: 1e9)

    def secs(v):
        mm, ss = v.split(":")
        return int(mm) * 60 + float(ss)

    n = 0
    for m in by_topic(msgs, "TimingData"):
        for num, ln in m.data["Lines"].items():
            if "LastLapTime" in ln:
                t = secs(ln["LastLapTime"]["Value"])
                ov, pe = ln["LastLapTime"]["OverallFastest"], ln["LastLapTime"]["PersonalFastest"]
                assert ov == (t < best - 1e-9) and pe == (t < personal[num] - 1e-9)
                assert not ov or pe
                best = min(best, t)
                personal[num] = min(personal[num], t)
                n += 1
                if pe:
                    assert secs(ln["BestLapTime"]["Value"]) == pytest.approx(t, abs=1e-6)
    assert n >= 20 * (LAPS - 3)
    assert 80 < best < 100  # plausible F1 lap time


def test_lap_and_sector_times_add_up(msgs):
    sec: Dict[str, Dict[str, float]] = defaultdict(dict)
    checked = 0
    for m in by_topic(msgs, "TimingData"):
        for num, ln in m.data["Lines"].items():
            for k, s in ln.get("Sectors", {}).items():
                sec[num][k] = float(s["Value"])
            if "LastLapTime" in ln and len(sec[num]) == 3:
                v = ln["LastLapTime"]["Value"]
                t = int(v.split(":")[0]) * 60 + float(v.split(":")[1])
                assert sum(sec[num].values()) == pytest.approx(t, abs=0.0025)
                sec[num] = {}
                checked += 1
            elif "LastLapTime" in ln:
                sec[num] = {}
    assert checked > 100


def test_pit_stops_and_tyre_stints(msgs):
    state: Dict = {}
    pit_events = Counter()
    inpit: Dict[str, bool] = {}
    for m in msgs:
        if m.topic == "TimingData":
            for num, ln in m.data["Lines"].items():
                if "InPit" in ln:
                    if ln["InPit"] and not inpit.get(num):
                        pit_events[num] += 1
                    inpit[num] = ln["InPit"]
                if ln.get("PitOut"):
                    assert inpit.get(num) is False  # PitOut only after leaving the pit lane
        elif m.topic == "TimingAppData":
            deep_merge(state, m.data)
    assert len(pit_events) >= 12  # most cars stopped (SC/VSC period triggers many stops)
    final_stops = {}
    for m in by_topic(msgs, "TimingData"):
        for num, ln in m.data["Lines"].items():
            if "NumberOfPitStops" in ln:
                final_stops[num] = ln["NumberOfPitStops"]
    for num, n in pit_events.items():
        assert final_stops[num] in (n, n - 1)  # the counter ticks on pit exit
    for num, ln in state["Lines"].items():
        stints = as_list(ln["Stints"])
        assert len(stints) == final_stops[num] + 1
        if len(stints) > 1:  # the two-compound rule
            assert len({s["Compound"] for s in stints}) >= 2
        assert all(s["Compound"] in ("SOFT", "MEDIUM", "HARD") for s in stints)
        assert all("TotalLaps" in s and "StartLaps" in s for s in stints)
        assert sum(s["TotalLaps"] - s["StartLaps"] for s in stints) <= LAPS


def test_two_compounds_rule_for_finishers(long_race):
    state: Dict = {}
    for m in long_race:
        if m.topic == "TimingAppData":
            deep_merge(state, m.data)
    many = 0
    for ln in state["Lines"].values():
        stints = as_list(ln["Stints"])
        if len(stints) > 1:
            many += 1
            assert len({s["Compound"] for s in stints}) >= 2
    assert many >= 12


def test_retirement_flow(msgs):
    retired = [(m.utc, n) for m in by_topic(msgs, "TimingData") for n, l in m.data["Lines"].items() if l.get("Retired")]
    assert 1 <= len(retired) <= 3
    texts = [e["Message"] for m in by_topic(msgs, "RaceControlMessages") for e in as_list(m.data["Messages"])]
    assert any("STOPPED ON TRACK" in t for t in texts)


# ----------------------------------------------------------------------------- telemetry vs geometry
def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def nearest_idx(track: Track, pts, x, y):
    best, bi = 1e18, 0
    for i, (ax, ay, _) in enumerate(pts):
        d = (ax - x) ** 2 + (ay - y) ** 2
        if d < best:
            best, bi = d, i
    return bi


def test_positions_lie_on_the_track_polyline(msgs, track):
    pts = track.units()
    n = len(pts)
    checked = 0
    for m in by_topic(msgs, "Position")[::37]:
        for num, e in m.data["Position"][0]["Entries"].items():
            assert e["Status"] in ("OnTrack", "OffTrack")
            x, y = e["X"], e["Y"]
            i = nearest_idx(track, pts, x, y)
            d = min(
                seg_dist(x, y, pts[j][0], pts[j][1], pts[(j + 1) % n][0], pts[(j + 1) % n][1])
                for j in ((i - 1) % n, i)
            )
            assert d <= 1.5, (m.utc, num, d)  # F1 units are 1/10 m; both X/Y and the vertices are rounded ints
            checked += 1
    assert checked > 1000


def test_track_geometry(track):
    pts = track.units()
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # F1 units: 1/10 m. A ~5 km circuit spans a couple of km = tens of thousands of units
    assert 15000 < max(xs) - min(xs) < 60000
    assert 4500 < track.length < 7000
    assert 12 <= len(track.corners) <= 22
    zs = [p[2] for p in pts]
    assert max(zs) - min(zs) > 50  # elevation present
    assert track.drs_zones and all(0 <= a < 1 and 0 <= b < 1 for a, b in track.drs_zones)


def test_speed_channels_in_range(race, msgs):
    """60-350 km/h for every car that is racing (not in the pit lane, retired/retiring, or on the standing start)."""
    start = race.race_start_utc
    inpit: Dict[str, bool] = {}
    retired = set()
    lo, hi, n = 1e9, 0, 0
    for m in msgs:
        if m.topic == "TimingData":
            for num, ln in m.data["Lines"].items():
                if "InPit" in ln:
                    inpit[num] = ln["InPit"]
                if ln.get("Retired"):
                    retired.add(num)
        elif m.topic == "CarData" and m.utc > start + 12:
            for num, c in m.data["Entries"][0]["Cars"].items():
                if inpit.get(num) or num in retired:
                    continue
                ch = c["Channels"]
                sp = ch["2"]
                lo, hi, n = min(lo, sp), max(hi, sp), n + 1
                assert 60 <= sp <= 350, (m.utc - start, num, sp)
                assert 1 <= ch["3"] <= 8
                assert 0 <= ch["4"] <= 100 and ch["5"] in (0, 100)
                assert ch["45"] in (0, 8, 12)
                assert 8000 <= ch["0"] <= 13000
    assert n > 50000 and lo < 120 and hi > 290  # real spread, slow corners and fast straights


def test_telemetry_channels_internally_consistent(race, msgs):
    gear_speed = defaultdict(list)
    brake_acc, thr_acc = [], []
    prev: Dict[str, int] = {}
    start = race.race_start_utc
    for m in by_topic(msgs, "CarData"):
        if m.utc < start + 15:
            continue
        for num, c in m.data["Entries"][0]["Cars"].items():
            ch = c["Channels"]
            sp = ch["2"]
            gear_speed[ch["3"]].append(sp)
            if num in prev:
                acc = (sp - prev[num]) / 0.25 / 3.6
                if ch["5"] == 100:
                    brake_acc.append(acc)
                if ch["4"] == 100:
                    thr_acc.append(acc)
            prev[num] = sp
    means = [sum(gear_speed[g]) / len(gear_speed[g]) for g in sorted(gear_speed) if len(gear_speed[g]) > 100]
    assert means == sorted(means)  # higher gear <=> higher speed
    assert sum(brake_acc) / len(brake_acc) < -3  # braking decelerates
    assert sum(thr_acc) / len(thr_acc) > -0.5  # full throttle does not slow the car on average


def test_speed_follows_track_curvature(race, msgs, track):
    """Cars are slower in corners and fast on straights: speed correlates with the circuit's curvature."""
    pts = track.units()
    start = race.race_start_utc
    corner, straight = [], []
    inpit = set()
    pos_by_t = {m.utc: m for m in by_topic(msgs, "Position")}
    for m in msgs:
        if m.topic == "TimingData":
            for num, ln in m.data["Lines"].items():
                if "InPit" in ln:
                    (inpit.add if ln["InPit"] else inpit.discard)(num)
        elif m.topic == "CarData" and m.utc > start + 120 and int(m.utc * 4) % 12 == 0:
            e = pos_by_t[m.utc].data["Position"][0]["Entries"]["4"]
            if "4" in inpit:
                continue
            i = nearest_idx(track, pts, e["X"], e["Y"])
            k = track.curv[i]
            sp = m.data["Entries"][0]["Cars"]["4"]["Channels"]["2"]
            (corner if k > 1 / 120 else straight if k < 1 / 1500 else []).append(sp)
    assert len(corner) > 20 and len(straight) > 20
    assert sum(straight) / len(straight) > sum(corner) / len(corner) + 60


def test_speed_matches_position_derivative(race, msgs):
    """Reported speed ~ distance travelled between 4 Hz Position samples (kinematic consistency)."""
    pos = by_topic(msgs, "Position")
    cd = by_topic(msgs, "CarData")
    start = race.race_start_utc
    errs = []
    for k in range(int((start - race.times[0] + 60) * 4), len(pos) - 1, 7):
        a = pos[k].data["Position"][0]["Entries"]["81"]
        b = pos[k + 1].data["Position"][0]["Entries"]["81"]
        if a["Status"] != "OnTrack" or b["Status"] != "OnTrack":
            continue
        d = math.hypot(b["X"] - a["X"], b["Y"] - a["Y"]) / 10.0 / 0.25  # m/s
        v = cd[k + 1].data["Entries"][0]["Cars"]["81"]["Channels"]["2"] / 3.6
        if v > 30:
            errs.append(abs(d - v) / v)
    assert len(errs) > 100
    errs.sort()
    assert errs[len(errs) // 2] < 0.04  # median error < 4%


# ----------------------------------------------------------------------------- race dynamics
def test_overtakes_happen(race, msgs):
    changes = 0
    last: Dict[str, int] = {}
    start = race.race_start_utc
    inpit = set()
    for m in by_topic(msgs, "TimingData"):
        for num, ln in m.data["Lines"].items():
            if "InPit" in ln:
                (inpit.add if ln["InPit"] else inpit.discard)(num)
            if "Position" in ln:
                p = int(ln["Position"])
                if m.utc > start + 60 and num in last and p < last[num] and num not in inpit:
                    changes += 1
                last[num] = p
    assert changes >= 15


def test_cars_get_lapped_in_a_long_race(long_race):
    lapped = Counter()
    for m in long_race:
        if m.topic == "TimingData":
            for num, ln in m.data["Lines"].items():
                g = ln.get("GapToLeader", "")
                if g.endswith("L"):
                    lapped[num] += 1
    assert len(lapped) >= 2
    final: Dict = {}
    for m in long_race:
        if m.topic == "TimingData":
            deep_merge(final, m.data)
    laps = sorted(l["NumberOfLaps"] for l in final["Lines"].values() if not l.get("Retired"))
    assert laps[-1] == 57 and laps[0] == 56  # lapped cars finished a lap down
    assert [m.data["CurrentLap"] for m in by_topic(list(long_race), "LapCount")][-1] == 57


def test_drs_only_when_enabled_and_green(race, msgs):
    on = [m.utc for m in by_topic(msgs, "CarData") if any(c["Channels"]["45"] == 12 for c in m.data["Entries"][0]["Cars"].values())]
    assert on, "DRS never opened"
    texts = [(m.utc, e["Message"]) for m in by_topic(msgs, "RaceControlMessages") for e in as_list(m.data["Messages"])]
    enabled = min(t for t, s in texts if s == "DRS ENABLED")
    assert min(on) >= enabled


def test_sc_period_bunches_the_field(msgs):
    codes = [(m.utc, m.data["Status"]) for m in by_topic(msgs, "TrackStatus")]
    sc = [t for t, c in codes if c == "4"]
    vsc = [t for t, c in codes if c == "6"]
    assert sc or vsc
    if sc:
        t0 = sc[0]
        end = next(t for t, c in codes if t > t0 and c == "1")
        # speeds under the safety car are well below racing speeds
        speeds = [c["Channels"]["2"] for m in by_topic(msgs, "CarData") if t0 + 30 < m.utc < end - 20
                  for n, c in m.data["Entries"][0]["Cars"].items() if n == "81"]
        assert speeds and max(speeds) < 300 and sum(speeds) / len(speeds) < 190


# ----------------------------------------------------------------------------- recording files
def test_record_to_file_roundtrip(tmp_path):
    r = generate(SimConfig(seed=2, laps=3))
    p = tmp_path / "sub" / "rec.txt"
    n = record_to_file(r, p)
    lines = p.read_text().splitlines()
    assert n == len(lines) == len(r)
    assert sum(l.startswith("['CarData.z', '") for l in lines) == Counter(r.topics)["CarData"]
    assert sum(l.startswith("['Position.z', '") for l in lines) == Counter(r.topics)["Position"]
    assert all(l.startswith("['") and l.endswith("Z']") for l in lines)
    back = [parse_line(l) for l in lines]
    assert all(b is not None for b in back)
    for a, b in zip(r, back):
        assert a.topic == b.topic and a.data == b.data and a.utc == pytest.approx(b.utc, abs=5e-4)
    # decoded variant
    p2 = tmp_path / "dec.txt"
    record_to_file(r, p2, encode=False)
    assert "CarData.z" not in p2.read_text()
    back2 = [parse_line(l) for l in p2.read_text().splitlines()]
    assert [b.data for b in back2] == [b.data for b in back]
    # accepts plain lists of Msg, with a limit
    assert record_to_file(list(r)[:10], tmp_path / "ten.txt", limit=7) == 7


def test_record_to_file_accepts_sim_source(tmp_path):
    from backend.ingest.sources import SimSource

    n = record_to_file(SimSource(seed=2, laps=2), tmp_path / "s.txt")
    assert n > 1000


def test_cli_writes_recording(tmp_path, monkeypatch, capsys):
    from backend.sim.__main__ import main

    out = tmp_path / "recordings" / "sim.txt"
    assert main(["--laps", "2", "--seed", "4", "--out", str(out)]) == 0
    assert out.exists() and out.stat().st_size > 10000
    assert "wrote" in capsys.readouterr().out
    assert main(["--laps", "2", "--seed", "4", "--out", str(tmp_path / "b.txt"), "--decoded", "--no-incidents"]) == 0
    assert "TrackStatus', {'Status': '4'" not in (tmp_path / "b.txt").read_text()


# ----------------------------------------------------------------------------- misc configs / tracks
@pytest.mark.parametrize("name", list_tracks())
def test_track_presets(name):
    t = get_track(name)
    assert t.n > 300 and t.name == name
    # closed, smooth: no jump between consecutive vertices
    pts = t.units()
    steps = [math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) for i in range(len(pts))]
    assert max(steps) < 1.5 * min(steps) + 10
    assert t.lap_time_ref > 50
    assert get_track(name) is t  # cached


def test_presets_are_distinct():
    a, b = get_track("bahrain-like"), get_track("monza-like")
    assert abs(a.length - b.length) > 100
    ar = lambda t: (max(t.xs) - min(t.xs)) / (max(t.ys) - min(t.ys))
    assert abs(ar(a) - ar(b)) > 1.0
    with pytest.raises(KeyError):
        get_track("nowhere")


@pytest.mark.parametrize("name", ["monza-like", "tight-like"])
def test_other_tracks_generate(name):
    r = generate(SimConfig(seed=2, laps=4, track=name))
    assert r.track.name == name and len(r.final_order) == 20
    assert {m.topic for m in r} >= {"CarData", "Position", "TimingData"}


def test_small_and_edge_configs():
    r = generate(SimConfig(seed=1, laps=1))
    st = [m.data["Status"] for m in r if m.topic == "SessionStatus"]
    assert st[-1] == "Ends" and "Finished" in st
    r = generate(SimConfig(seed=1, laps=3, cars=6, incidents=False))
    assert len(r.nums) == 6
    assert not [m for m in r if m.topic == "TrackStatus" and m.data["Status"] in ("2", "4", "6")]
    assert generate_messages(seed=1, laps=2)[0].topic == "SessionInfo"


def test_many_seeds_complete_without_error():
    for seed in range(10, 16):
        r = generate(SimConfig(seed=seed, laps=6 + seed % 5))
        assert len(r.final_order) == 20 and r.end_utc > r.race_start_utc


def test_time_index_helpers(race):
    i = bisect.bisect_right(race.times, race.race_start_utc)
    assert race.topics[i - 1] in ("SessionStatus", "ExtrapolatedClock", "CarData", "Position", "Heartbeat", "TimingData")
    assert json.dumps(race.get(5).data)  # JSON-serialisable
