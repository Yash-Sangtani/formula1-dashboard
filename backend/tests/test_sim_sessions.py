"""Tests for the non-race simulator sessions: sprint, practice, qualifying, sprint qualifying (CONTRACT §6)."""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Dict, List

import pytest

from backend.ingest.messages import Msg, deep_merge, parse_line, to_raw_line
from backend.sim import SimConfig, SimRace, generate, record_to_file

SCALE = 0.35  # shortens fp / quali parts so the tests stay fast


@pytest.fixture(scope="module")
def fp() -> SimRace:
    return generate(SimConfig(seed=1, session="fp", time_scale=SCALE))


@pytest.fixture(scope="module")
def quali() -> SimRace:
    return generate(SimConfig(seed=1, session="quali", time_scale=SCALE))


@pytest.fixture(scope="module")
def squali() -> SimRace:
    return generate(SimConfig(seed=2, session="sprint_quali", time_scale=SCALE))


@pytest.fixture(scope="module")
def sprint() -> SimRace:
    return generate(SimConfig(seed=1, laps=30, session="sprint", incidents=False))


def topic(msgs, name) -> List[Msg]:
    return [m for m in msgs if m.topic == name]


def lap_sec(text: str) -> float:
    if not text:
        return math.inf
    m, _, s = text.rpartition(":")
    return (int(m) if m else 0) * 60 + float(s)


def replay(msgs, upto: float = math.inf, topics=("TimingData",)) -> Dict[str, dict]:
    st: dict = {}
    for m in msgs:
        if m.utc > upto:
            break
        if m.topic in topics:
            deep_merge(st, m.data)
    return st


# ----------------------------------------------------------------------------- race is untouched
def test_race_output_unchanged_golden():
    h = hashlib.sha256()
    for seed, laps in ((1, 6), (4, 9)):
        for m in generate(SimConfig(seed=seed, laps=laps)):
            h.update(to_raw_line(m).encode())
    # default session stays "race": regenerate with the explicit kwarg and compare
    h2 = hashlib.sha256()
    for seed, laps in ((1, 6), (4, 9)):
        for m in generate(SimConfig(seed=seed, laps=laps, session="race")):
            h2.update(to_raw_line(m).encode())
    assert h.hexdigest() == h2.hexdigest()
    info = [m for m in generate(SimConfig(seed=1, laps=2)) if m.topic == "SessionInfo"][0].data
    assert (info["Type"], info["Name"]) == ("Race", "Race")


def test_unknown_session_rejected():
    with pytest.raises(ValueError):
        generate(SimConfig(session="warmup"))


# ----------------------------------------------------------------------------- sprint
def test_sprint(sprint):
    ms = sprint.messages()
    info = topic(ms, "SessionInfo")[0].data
    assert (info["Type"], info["Name"]) == ("Race", "Sprint")
    lc = topic(ms, "LapCount")
    assert lc[0].data["TotalLaps"] <= 15 < 30  # capped, sprint is short
    # no mandatory stop: nobody pits without a safety car
    last = replay(ms)["Lines"]
    assert all(ln.get("NumberOfPitStops", 0) == 0 for ln in last.values())
    assert {"GapToLeader"} <= set(next(iter(last.values())))
    assert topic(ms, "SessionStatus")[-1].data["Status"] == "Ends"


# ----------------------------------------------------------------------------- practice
def test_fp_topics_and_shapes(fp):
    ms = fp.messages()
    topics = {m.topic for m in ms}
    assert {"SessionInfo", "DriverList", "TimingData", "TimingAppData", "CarData", "Position", "ExtrapolatedClock",
            "SessionStatus", "TrackStatus", "RaceControlMessages", "WeatherData", "TeamRadio"} <= topics
    assert "LapCount" not in topics
    info = topic(ms, "SessionInfo")[0].data
    assert (info["Type"], info["Name"]) == ("Practice", "Practice 1")
    keys = set()
    for m in topic(ms, "TimingData"):
        for ln in m.data.get("Lines", {}).values():
            keys |= set(ln)
    assert "GapToLeader" not in keys and "IntervalToPositionAhead" not in keys
    assert {"TimeDiffToFastest", "TimeDiffToPositionAhead", "BestLapTime", "LastLapTime", "Sectors", "InPit",
            "PitOut", "NumberOfLaps", "Position"} <= keys
    assert "BestLapTimes" not in keys and "KnockedOut" in keys  # KnockedOut only in the prelude (always False)
    assert not any(ln.get("KnockedOut") for m in topic(ms, "TimingData") for ln in m.data.get("Lines", {}).values())


def test_fp_order_gaps_and_laps(fp):
    ms = fp.messages()
    st = replay(ms)["Lines"]
    timed = [(lap_sec(ln["BestLapTime"]["Value"]), n) for n, ln in st.items() if ln["BestLapTime"].get("Value")]
    assert len(timed) >= 18
    timed.sort()
    by_pos = sorted(st, key=lambda n: int(st[n]["Position"]))
    assert [n for _, n in timed] == by_pos[: len(timed)]
    p1 = st[by_pos[0]]
    assert p1["TimeDiffToFastest"] == "" and p1["TimeDiffToPositionAhead"] == ""
    for i, n in enumerate(by_pos[1 : len(timed)], 1):
        d = lap_sec(st[n]["BestLapTime"]["Value"]) - timed[0][0]
        assert float(st[n]["TimeDiffToFastest"]) == pytest.approx(d, abs=2e-3)
    # NumberOfLaps counts timed laps only: some long-run car has many, nobody has an absurd number
    laps = [ln["NumberOfLaps"] for ln in st.values()]
    assert max(laps) >= 6 and max(laps) < 40


def test_fp_clock_counts_down_with_red_flag(fp):
    ms = fp.messages()
    clk = topic(ms, "ExtrapolatedClock")
    assert clk[0].data["Remaining"] == "00:21:00" and clk[0].data["Extrapolating"] is False  # 60 min * 0.35
    run = [c for c in clk if c.data["Extrapolating"]]
    rem = [sum(int(x) * k for x, k in zip(c.data["Remaining"].split(":"), (3600, 60, 1))) for c in clk]
    assert rem[0] == 1260 and rem[-1] == 0 and clk[-1].data["Extrapolating"] is False
    assert run
    sst = [m.data["Status"] for m in topic(ms, "SessionStatus")]
    assert sst[0] == "Inactive" and "Aborted" in sst and sst.count("Started") == 2
    assert sst[-3:] == ["Finished", "Finalised", "Ends"]
    ts = [m.data["Status"] for m in topic(ms, "TrackStatus")]
    assert "5" in ts and ts[-1] == "1"
    # clock stopped during the red flag
    red0 = [m.utc for m in topic(ms, "TrackStatus") if m.data["Status"] == "5"][0]
    stopped = [c for c in clk if c.utc == red0]
    assert stopped and stopped[0].data["Extrapolating"] is False


def test_fp_garage_and_run_plans(fp):
    ms = fp.messages()
    pos = topic(ms, "Position")
    first = pos[0].data["Position"][0]["Entries"]
    assert all(e["Status"] == "OffTrack" for e in first.values())
    xy0 = {n: (e["X"], e["Y"]) for n, e in first.items()}
    assert len(set(xy0.values())) == len(xy0)  # distinct garage slots
    # a car idling in the garage keeps its XY, cars keep sending while some are on track
    on_track = set()
    off = fixed = 0
    for m in pos[::40]:
        for n, e in m.data["Position"][0]["Entries"].items():
            if e["Status"] == "OnTrack":
                on_track.add(n)
            else:
                off += 1
                fixed += (e["X"], e["Y"]) == xy0[n]  # the rest are brief garage <-> pit-box rolls
    assert len(on_track) >= 18 and off > 100 and fixed >= 0.8 * off
    # at any time not all cars are out: garage time exists
    occupied = [sum(e["Status"] == "OnTrack" for e in m.data["Position"][0]["Entries"].values()) for m in pos[::20]]
    assert min(occupied) <= 2 and max(occupied) >= 5
    assert all(len(m.data["Position"][0]["Entries"]) == 20 for m in pos[::50])
    # InPit True at the start; every car leaves and returns (InPit toggles)
    ins = Counter()
    for m in topic(ms, "TimingData"):
        for n, ln in m.data.get("Lines", {}).items():
            if ln.get("InPit") is True:
                ins[n] += 1
    assert all(v >= 3 for v in ins.values())
    # tyre programme: several compounds, runs per car
    comps = set()
    for m in topic(ms, "TimingAppData"):
        for ln in m.data["Lines"].values():
            for s in (ln.get("Stints") or {}).values():
                if "Compound" in s:
                    comps.add(s["Compound"])
    assert {"SOFT", "MEDIUM", "HARD"} <= comps


# ----------------------------------------------------------------------------- qualifying
@pytest.mark.parametrize("name", ["quali", "squali"])
def test_quali_parts_consistent(name, quali, squali):
    r = quali if name == "quali" else squali
    ms = r.messages()
    parts = r.info["parts"]
    assert [p["part"] for p in parts] == [1, 2, 3]
    assert [len(p["participants"]) for p in parts] == [20, 15, 10]
    assert [len(p["knocked_out"]) for p in parts] == [5, 5, 0]
    assert sum(1 for p in parts if p["red"]) == 1
    for a, b in zip(parts, parts[1:]):
        assert a["end_utc"] < b["start_utc"]
        assert a["laps_end_utc"] < b["start_utc"]  # track empty between the parts
        assert set(b["participants"]) == set(a["participants"]) - set(a["knocked_out"])
    # SessionPart in TimingData
    sp = [(m.utc, m.data["SessionPart"]) for m in topic(ms, "TimingData") if "SessionPart" in m.data]
    assert [p for _, p in sp] == [1, 1, 2, 3] or [p for _, p in sp][-3:] == [1, 2, 3]
    starts = {p: u for u, p in sp}
    for p in parts:
        assert starts[p["part"]] == pytest.approx(p["start_utc"])
    # session status cycle
    sst = [m.data["Status"] for m in topic(ms, "SessionStatus")]
    assert sst.count("Finished") == 3 and sst.count("Aborted") == 1 and sst[-2:] == ["Finalised", "Ends"]
    assert "LapCount" not in {m.topic for m in ms}
    # knockouts: KnockedOut patches match, and happen after the part ended but before the next one starts
    ko = [(m.utc, n) for m in topic(ms, "TimingData") for n, ln in m.data.get("Lines", {}).items() if ln.get("KnockedOut")]
    assert len(ko) == 10
    for i in (0, 1):
        got = {n for u, n in ko if parts[i]["end_utc"] <= u < parts[i + 1]["start_utc"]}
        assert got == set(parts[i]["knocked_out"])
    # 15 cars after Q1 / 10 after Q2 in the timing state
    after_q1 = replay(ms, parts[0]["final_utc"] + 1)["Lines"]
    assert sum(1 for ln in after_q1.values() if ln["KnockedOut"]) == 5
    assert sorted(int(ln["Position"]) for n, ln in after_q1.items() if ln["KnockedOut"]) == [16, 17, 18, 19, 20]
    after_q2 = replay(ms, parts[1]["final_utc"] + 1)["Lines"]
    assert sum(1 for ln in after_q2.values() if ln["KnockedOut"]) == 10
    assert sorted(int(ln["Position"]) for n, ln in after_q2.items() if ln["KnockedOut"]) == list(range(11, 21))


def test_quali_final_order_and_shapes(quali):
    ms = quali.messages()
    st = replay(ms)["Lines"]
    order = quali.final_order
    assert sorted(order) == sorted(st) and len(order) == 20
    assert [int(st[n]["Position"]) for n in order] == list(range(1, 21))
    bl = lambda n, i: lap_sec(st[n]["BestLapTimes"][i].get("Value", ""))
    for lo, hi, idx in ((0, 10, 2), (10, 15, 1), (15, 20, 0)):
        t = [bl(n, idx) for n in order[lo:hi]]
        assert t == sorted(t)
    # whoever is knocked out in Q1 did not run in Q2/Q3 (no best time there)
    for n in order[15:]:
        assert not st[n]["BestLapTimes"][1].get("Value") and not st[n]["BestLapTimes"][2].get("Value")
    for n in order[10:15]:
        assert not st[n]["BestLapTimes"][2].get("Value")
    assert st[order[0]]["TimeDiffToFastest"] == "" and st[order[0]]["Cutoff"] is False
    keys = {k for m in topic(ms, "TimingData") for ln in m.data.get("Lines", {}).values() for k in ln}
    assert {"BestLapTimes", "KnockedOut", "Cutoff", "TimeDiffToFastest", "TimeDiffToPositionAhead"} <= keys
    assert "GapToLeader" not in keys
    # cutoff flags were raised during Q1 for cars in the drop zone
    assert any(ln.get("Cutoff") is True for m in topic(ms, "TimingData") for ln in m.data.get("Lines", {}).values())


def test_quali_red_flag_and_cars_on_track(quali):
    ms = quali.messages()
    parts = quali.info["parts"]
    red = [p for p in parts if p["red"]][0]
    t = [(m.utc, m.data["Status"]) for m in topic(ms, "TrackStatus")]
    assert (red["red"][0], "5") in [(u, s) for u, s in t] and (red["red"][1], "1") in t
    rc = [m.data["Messages"] for m in topic(ms, "RaceControlMessages")]
    flat = [e for x in rc for e in (x if isinstance(x, list) else x.values())]
    assert any(e.get("Flag") == "RED" for e in flat) and any(e.get("Flag") == "CHEQUERED" for e in flat)
    # empty track between the parts: every Position entry OffTrack and speed 0 in the middle of the break
    mid = (parts[0]["laps_end_utc"] + parts[1]["start_utc"]) / 2
    pos = min(topic(ms, "Position"), key=lambda m: abs(m.utc - mid))
    assert all(e["Status"] == "OffTrack" for e in pos.data["Position"][0]["Entries"].values())
    cd = min(topic(ms, "CarData"), key=lambda m: abs(m.utc - mid))
    assert all(c["Channels"]["2"] == 0 for c in cd.data["Entries"][0]["Cars"].values())
    # during a part many cars are on track at once (traffic)
    p1 = parts[0]
    busy = max(sum(e["Status"] == "OnTrack" for e in m.data["Position"][0]["Entries"].values())
               for m in topic(ms, "Position") if p1["start_utc"] < m.utc < p1["end_utc"])
    assert busy >= 12


def test_quali_info_types(quali, squali):
    i = topic(quali.messages(), "SessionInfo")[0].data
    assert (i["Type"], i["Name"]) == ("Qualifying", "Qualifying")
    j = topic(squali.messages(), "SessionInfo")[0].data
    assert (j["Type"], j["Name"]) == ("Qualifying", "Sprint Qualifying")
    # sprint quali parts are shorter than quali parts
    dq = [p["end_utc"] - p["start_utc"] for p in quali.info["parts"]]
    ds = [p["end_utc"] - p["start_utc"] for p in squali.info["parts"]]
    assert dq[0] > ds[0]


# ----------------------------------------------------------------------------- determinism
@pytest.mark.parametrize("session", ["fp", "quali", "sprint_quali", "sprint"])
def test_deterministic_by_seed(session):
    kw = dict(session=session, time_scale=0.2, laps=3)
    a = generate(SimConfig(seed=7, **kw))
    b = generate(SimConfig(seed=7, **kw))
    c = generate(SimConfig(seed=8, **kw))
    la = [to_raw_line(m) for m in a]
    assert la == [to_raw_line(m) for m in b]
    assert la != [to_raw_line(m) for m in c]
    assert a.final_order == b.final_order
    t = [m.utc for m in a]
    assert t == sorted(t)


@pytest.mark.parametrize("session", ["race", "sprint", "fp", "quali", "sprint_quali"])
def test_snapshot_first_round_trip(session, tmp_path):
    r = generate(SimConfig(seed=3, laps=3, session=session, time_scale=0.2, snapshot_first=True))
    msgs = r.messages()
    assert 6 <= r.snapshot_count < 20
    path = tmp_path / "rec.txt"
    n = record_to_file(r, path)
    lines = path.read_text().splitlines()
    snap = r.snapshot_lines()
    assert lines[: len(snap)] == snap
    assert all(ln.endswith(", '']") for ln in snap)  # empty timestamp, JSON string payload
    assert all(isinstance(eval(ln)[1], str) for ln in snap)
    parsed = [parse_line(ln, default_utc=r.cfg.base_utc) for ln in lines]
    assert all(p is not None for p in parsed) and len(parsed) == n
    # snapshot part == merged leading messages; the rest are the original messages one to one
    want = replay(msgs[: r.snapshot_count], topics={m.topic for m in msgs})
    got: dict = {}
    for p in parsed[: len(snap)]:
        assert p.utc == r.cfg.base_utc
        deep_merge(got.setdefault(p.topic, {}), p.data)
    ref: dict = {}
    for m in msgs[: r.snapshot_count]:
        deep_merge(ref.setdefault(m.topic, {}), m.data)
    assert got == ref and want is not None
    rest = parsed[len(snap):]
    assert len(rest) == len(msgs) - r.snapshot_count
    for a, b in list(zip(rest, msgs[r.snapshot_count:]))[:400]:
        assert a.topic == b.topic and a.data == b.data and a.utc == pytest.approx(b.utc, abs=5e-4)
    # without the flag nothing changes: plain timestamped lines only
    plain = generate(SimConfig(seed=3, laps=3, session=session, time_scale=0.2))
    assert [to_raw_line(m) for m in plain][:5] == [to_raw_line(m) for m in msgs][:5]
    p2 = tmp_path / "plain.txt"
    assert record_to_file(plain, p2) == len(plain) and ", '']" not in p2.read_text()[:2000]


def test_record_to_file_sim_source_session(tmp_path):
    from backend.ingest.sources import SimSource

    src = SimSource(seed=2, laps=2, session="fp", time_scale=0.1, snapshot_first=True)
    n = record_to_file(src, tmp_path / "s.txt")
    first = (tmp_path / "s.txt").read_text().splitlines()[0]
    assert n > 500 and first.startswith("['SessionInfo', '{") and first.endswith(", '']")


def test_cli_sessions(tmp_path, capsys):
    from backend.sim.__main__ import main

    out = tmp_path / "q.txt"
    assert main(["--session", "quali", "--seed", "3", "--time-scale", "0.2", "--snapshot", "--out", str(out)]) == 0
    assert "quali time" in capsys.readouterr().out
    lines = out.read_text().splitlines()
    assert lines[0].startswith("['SessionInfo', '{") and lines[0].endswith(", '']")
    msgs = [parse_line(ln) for ln in lines]
    assert all(msgs)
    assert main(["--session", "sprint", "--seed", "3", "--out", str(tmp_path / "s.txt"), "--no-incidents"]) == 0
    assert "'Name': 'Sprint'" in (tmp_path / "s.txt").read_text()
