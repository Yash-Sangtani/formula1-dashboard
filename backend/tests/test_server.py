import asyncio
import json
import math
import time
from dataclasses import dataclass
from typing import Any

import pytest
from starlette.testclient import TestClient

from backend.server import create_app, Client

try:
    from backend.ingest.messages import Msg
except Exception:
    @dataclass
    class Msg:
        topic: str
        data: Any
        utc: float


class FakeControl:
    def __init__(self):
        self.speed = 1.0
        self.paused = False
        self.seeks = []

    async def seek(self, utc):
        self.seeks.append(utc)

    def info(self):
        return {"start": 0, "end": 100}


class FakeSource:
    mode = "file"

    def __init__(self):
        self.control = FakeControl()

    async def messages(self):
        yield Msg("DriverList", {"1": {"RacingNumber": "1", "Tla": "VER", "FirstName": "Max", "LastName": "Verstappen",
                                       "TeamName": "RB", "TeamColour": "3671C6", "Line": 1}}, 1000.0)
        yield Msg("TimingData", {"Lines": {"1": {"Position": "1", "NumberOfLaps": 1, "LastLapTime": {"Value": "1:30.000"}}}}, 1001.0)
        t = 1002.0
        for lap in range(3):
            for i in range(120):
                a = 2 * math.pi * i / 120
                yield Msg("Position", {"Position": [{"Timestamp": "t", "Entries": {"1": {"Status": "OnTrack", "X": 3000 * math.cos(a), "Y": 2000 * math.sin(a), "Z": 0}}}]}, t)
                yield Msg("CarData", {"Entries": [{"Utc": "t", "Cars": {"1": {"Channels": {"2": 300, "3": 7, "4": 100, "5": 0, "45": 12}}}}]}, t)
                t += 0.5
                if i % 40 == 0:
                    await asyncio.sleep(0.02)
            yield Msg("TimingData", {"Lines": {"1": {"NumberOfLaps": lap + 2, "LastLapTime": {"Value": "1:29.%d" % lap}}}}, t)
        await asyncio.sleep(0.3)


def recv_until(ws, types, limit=200):
    seen = {}
    for _ in range(limit):
        d = json.loads(ws.receive_text())
        seen.setdefault(d["type"], d)
        if all(t in seen for t in types):
            break
    return seen


def test_ws_contract_and_rest():
    src = FakeSource()
    app = create_app(src, mode="file", frontend_dir=None)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            first = json.loads(ws.receive_text())
            assert first["type"] == "hello" and first["mode"] == "file" and first["replay"] is True and "info" in first
            assert json.loads(ws.receive_text())["type"] == "state"
            seen = recv_until(ws, ["fast", "track", "history"])
            assert {"fast", "track", "history"} <= set(seen)
            f = seen["fast"]
            assert f["cars"]["1"].keys() >= {"x", "y"}
            tr = seen["track"]["track"]
            assert tr["points"] and len(tr["bounds"]) == 4 and "drsZones" in tr and len(tr["sectors"]) == 2
            h = seen["history"]
            assert "laps" in h and "events" in h and "maxLap" in h
        time.sleep(0.8)
        st = client.get("/api/state").json()
        assert st["type"] == "state" and "1" in st["cars"]
        assert client.get("/api/history").json()["type"] == "history"
        assert client.get("/api/track").json()["track"] is not None
        s = client.get("/api/status").json()
        assert s["mode"] == "file" and s["totalMsgs"] > 100 and "msgsPerSec" in s and s["engine"]["drivers"] == 1
        assert s["lastMessageAgeSec"] is not None and s["error"] is None
        with client.websocket_connect("/ws") as ws:
            for _ in range(3):
                ws.receive_text()
            ws.send_text(json.dumps({"type": "control", "speed": 5}))
            ws.send_text(json.dumps({"type": "control", "paused": True}))
            ws.send_text(json.dumps({"type": "control", "seek": 1500}))
            time.sleep(0.3)
        assert src.control.speed == 5 and src.control.paused is True and src.control.seeks == [1500.0]
        r = client.post("/api/control", json={"speed": 2})
        assert r.status_code == 200 and src.control.speed == 2


def test_source_error_reported_and_state_still_served():
    class Bad:
        mode = "file"
        control = FakeControl()

        async def messages(self):
            yield Msg("DriverList", {"7": {"Tla": "BOT"}}, 1.0)
            raise RuntimeError("boom")

    with TestClient(create_app(Bad(), mode="file")) as c:
        time.sleep(0.3)
        s = c.get("/api/status").json()
        assert "boom" in (s["error"] or "")
        assert "7" in c.get("/api/state").json()["cars"]


def test_client_buffer_merges_fast_frames():
    class W: pass
    async def go():
        c = Client(W())
        c.put_fast({"utc": 1, "cars": {"1": {"x": 1, "speed": 10}}}, "a")
        c.put_fast({"utc": 2, "cars": {"1": {"x": 2}, "2": {"x": 5}}}, "b")
        c.put("state", "S1")
        c.put("state", "S2")
        out = c.take()
        assert out[0] == "S2"
        f = json.loads(out[1])
        assert f["utc"] == 2 and f["cars"]["1"] == {"x": 2, "speed": 10} and f["cars"]["2"] == {"x": 5}
        assert c.take() == []
    asyncio.run(go())


def test_static_frontend_no_cache(tmp_path):
    (tmp_path / "index.html").write_text("<html>hi</html>")
    with TestClient(create_app(FakeSource(), mode="file", frontend_dir=tmp_path, dev=True)) as c:
        r = c.get("/")
        assert r.status_code == 200 and "hi" in r.text and "no-cache" in r.headers["cache-control"]
        assert c.get("/api/status").status_code == 200
