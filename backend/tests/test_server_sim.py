"""End-to-end: SimSource -> engine -> WebSocket, asserting the contract's message types and fields."""
import json
import time

import pytest
from starlette.testclient import TestClient

from backend.ingest.sources import SimSource
from backend.server import create_app

CAR_KEYS = {"num", "tla", "name", "team", "color", "pos", "grid", "posChange", "gapLeader", "interval", "gapLeaderSec",
            "intervalSec", "lap", "pits", "inPit", "pitOut", "retired", "stopped", "last", "best", "sectors", "speedTraps",
            "tyre", "stints", "tel", "xy", "progress"}
STATE_KEYS = {"type", "utc", "session", "clock", "lap", "trackStatus", "weather", "order", "cars", "raceControl", "radio",
              "records", "pitStops"}


def drain(ws, seconds):
    end = time.time() + seconds
    out = []
    while time.time() < end:
        out.append(json.loads(ws.receive_text()))
    return out


@pytest.fixture(scope="module")
def run():
    app = create_app(SimSource(seed=3, speed=60, laps=6), mode="sim", frontend_dir=None)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            msgs = [json.loads(ws.receive_text()) for _ in range(2)]
            t_end = time.time() + 14
            while time.time() < t_end:
                msgs.append(json.loads(ws.receive_text()))
                st = client.get("/api/status").json()
                if st["ended"] and any(m["type"] == "history" for m in msgs[-50:]) and time.time() > t_end - 10:
                    break
        yield client, msgs


def test_message_types_and_hello(run):
    client, msgs = run
    assert msgs[0]["type"] == "hello" and msgs[0]["mode"] == "sim" and msgs[0]["replay"] is True
    assert msgs[1]["type"] == "state"
    types = {m["type"] for m in msgs}
    assert {"hello", "state", "fast", "track", "history"} <= types


def test_final_state_fields(run):
    client, _ = run
    s = client.get("/api/state").json()
    assert STATE_KEYS <= set(s)
    assert len(s["order"]) >= 10 and set(s["order"]) == set(s["cars"])
    for c in s["cars"].values():
        assert CAR_KEYS <= set(c)
        assert len(c["sectors"]) == 3
    leader = s["cars"][s["order"][0]]
    assert leader["pos"] == 1 and leader["lap"] >= 1
    assert leader["tyre"]["compound"] in ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")
    assert s["lap"]["current"] >= 1
    assert s["session"]["status"]
    assert any(c["xy"] is not None for c in s["cars"].values())
    assert s["records"]["fastestLap"] is not None


def test_track_and_history(run):
    client, msgs = run
    tr = client.get("/api/track").json()["track"]
    assert tr is not None
    assert 600 <= len(tr["points"]) <= 1000
    assert {"points", "length", "bounds", "startIdx", "drsZones", "sectors", "rotation"} <= set(tr)
    assert 3000 < tr["length"] < 9000, tr["length"]
    s = client.get("/api/state").json()
    progs = [c["progress"] for c in s["cars"].values() if c["progress"] is not None]
    assert progs and all(0 <= p <= 1 for p in progs)
    h = client.get("/api/history").json()
    assert h["maxLap"] >= 3 and h["laps"] and h["events"]
    rec = next(iter(h["laps"].values()))[0]
    assert {"lap", "sec", "s", "compound", "tyreAge", "pos", "gapLeader", "interval", "pit", "track"} <= set(rec)
    assert {"utc", "lap", "kind", "num", "text"} <= set(h["events"][0])


def test_status_and_seek(run):
    client, _ = run
    st = client.get("/api/status").json()
    assert st["mode"] == "sim" and st["totalMsgs"] > 1000 and st["engine"]["drivers"] >= 10
    info = st["control"]
    start, end = info["start"], info["end"]
    r = client.post("/api/control", json={"seek": start + (end - start) * 0.4})
    assert r.status_code == 200
    time.sleep(1.5)
    s = client.get("/api/state").json()
    assert s["order"] and s["lap"]["current"] >= 1
    assert s["utc"] < end
