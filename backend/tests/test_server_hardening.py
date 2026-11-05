"""Server hardening: phase/waiting state, client limits, resilience, NaN safety, track reset / circuit change / fallback."""
from __future__ import annotations

import asyncio
import json
import math
import time

import numpy as np
import pytest
from starlette.testclient import TestClient

from backend.engine import RaceEngine
from backend.ingest.messages import Msg, format_ts
from backend.ingest.sources import LiveSource
from backend.server import Client, _safe_dumps, create_app, session_context
from backend.track import OfficialLayoutFallback

BASE = 1_700_000_000.0


def flines(topic, data, off=None):
    return str([topic, data, "" if off is None else format_ts(BASE + off)])


class Script:
    """Fake fastf1 client: writes ``steps`` (list of (sleep_before, line)) then idles until kicked."""

    def __init__(self, steps):
        self.steps = steps

    def __call__(self, filename, filemode, timeout, no_auth):
        steps = self.steps

        class C:
            def __init__(s):
                s.timeout = timeout
                s._t_last_message = time.time()

            def start(s):
                with open(filename, filemode) as f:
                    for dt, ln in steps:
                        time.sleep(dt)
                        f.write(ln + "\n")
                        f.flush()
                while s.timeout != 1:
                    time.sleep(0.02)

        return C()


def live(tmp_path, steps, **kw):
    return LiveSource(recordings_dir=str(tmp_path / "rec"), client_factory=Script(steps), poll=0.01, watch_interval=0.05,
                      retry_delay=0.05, **kw)


def wait_for(fn, limit=10.0, step=0.05):
    end = time.time() + limit
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(step)
    raise AssertionError("timeout waiting for condition")


def recv_types(ws, n=200, stop=None):
    out = []
    for _ in range(n):
        d = json.loads(ws.receive_text())
        out.append(d)
        if stop and stop(d):
            break
    return out


# ----------------------------------------------------------------------------- waiting / live / error phases
def test_live_waiting_state_in_status_hello_and_status_frame(tmp_path):
    steps = [(0.0, flines("SessionInfo", '{"Meeting": {"Name": "Test GP"}, "Name": "Practice 1", "Type": "Practice"}')),
             (0.0, flines("SessionStatus", '{"Status": "Inactive"}')),
             (0.0, flines("DriverList", '{"1": {"RacingNumber": "1", "Tla": "VER"}}')),
             (1.5, flines("SessionStatus", {"Status": "Started"}, 5.0))]
    src = live(tmp_path, steps)
    with TestClient(create_app(src, mode="live", frontend_dir=None, track_fallback=False)) as c:
        st = wait_for(lambda: (lambda s: s if s["totalMsgs"] >= 3 else None)(c.get("/api/status").json()))
        assert st["phase"] == "waiting" and st["waiting"] is True and st["error"] is None
        assert "waiting for the session" in st["message"] and "Practice 1" in st["message"]
        assert st["recording"]["path"].endswith(".txt") and st["recording"]["bytes"] > 0
        assert st["uptimeSec"] >= 0 and st["lastMessageAgeSec"] is not None and st["mode"] == "live"
        with c.websocket_connect("/ws") as ws:
            hello = json.loads(ws.receive_text())
            assert hello["type"] == "hello" and hello["phase"] == "waiting" and hello["waiting"] is True
            assert hello["info"]["recording"] == st["recording"]["path"] and hello["error"] is None
            frames = recv_types(ws, 300, stop=lambda d: d["type"] == "status" and d["phase"] == "live")
            sf = [d for d in frames if d["type"] == "status"]
            assert sf and sf[-1]["phase"] == "live" and sf[-1]["waiting"] is False
        assert c.get("/api/status").json()["phase"] == "live"


def test_live_error_phase_shows_reason_and_recovers(tmp_path):
    n = []

    def factory(filename, filemode, timeout, no_auth):
        n.append(1)
        if len(n) < 3:
            raise ConnectionError("Max retries exceeded with url: /signalrcore/negotiate")
        return Script([(0.0, flines("SessionStatus", {"Status": "Started"}, 1.0))])(filename, filemode, timeout, no_auth)

    src = LiveSource(recordings_dir=str(tmp_path / "rec"), client_factory=factory, poll=0.01, watch_interval=0.05,
                     retry_delay=0.3, max_delay=0.4)
    with TestClient(create_app(src, mode="live", frontend_dir=None, track_fallback=False)) as c:
        st = wait_for(lambda: (lambda s: s if s["phase"] == "error" else None)(c.get("/api/status").json()), 5)
        assert "Cannot reach livetiming.formula1.com" in st["error"]
        wait_for(lambda: c.get("/api/status").json()["phase"] == "live", 10)
        assert c.get("/api/status").json()["error"] is None


def test_replay_phase_ended_and_error():
    class Src:
        mode = "file"
        control = type("C", (), {"speed": 1.0, "paused": False, "info": lambda s: {}})()

        async def messages(self):
            yield Msg("DriverList", {"1": {"Tla": "VER"}}, 1.0)

    with TestClient(create_app(Src(), mode="file", frontend_dir=None)) as c:
        st = wait_for(lambda: (lambda s: s if s["ended"] else None)(c.get("/api/status").json()))
        assert st["phase"] == "ended" and st["recording"] is None


# ----------------------------------------------------------------------------- resilience
def test_engine_exception_does_not_stop_the_stream():
    class Boom(RaceEngine):
        def apply(self, msg):
            if msg.topic == "LapCount":
                raise RuntimeError("engine bug")
            return super().apply(msg)

    class Src:
        mode = "file"
        control = type("C", (), {"speed": 1.0, "paused": False, "info": lambda s: {}})()

        async def messages(self):
            for i in range(5):
                yield Msg("LapCount", {"CurrentLap": i}, float(i))
                yield Msg("DriverList", {str(i + 1): {"Tla": "D%d" % i}}, float(i))

    with TestClient(create_app(Src(), mode="file", engine=Boom(), frontend_dir=None)) as c:
        st = wait_for(lambda: (lambda s: s if s["ended"] else None)(c.get("/api/status").json()))
        assert st["totalMsgs"] == 10 and st["applyErrors"] == 5 and st["error"] is None
        assert len(c.get("/api/state").json()["cars"]) == 5


def test_live_source_that_raises_is_reopened_forever():
    opened = []

    class Flaky:
        mode = "live"
        control = type("C", (), {"speed": 1.0, "paused": False, "info": lambda s: {}})()

        async def messages(self):
            opened.append(1)
            if len(opened) < 3:
                raise RuntimeError("transient %d" % len(opened))
            yield Msg("DriverList", {"1": {"Tla": "VER"}}, 1.0)
            await asyncio.sleep(30)

    import backend.server as S

    with TestClient(create_app(Flaky(), mode="live", frontend_dir=None)) as c:
        wait_for(lambda: c.get("/api/status").json()["totalMsgs"] >= 1, 20)
        s = c.get("/api/status").json()
        assert len(opened) == 3 and s["error"] is None and s["phase"] in ("live", "waiting")


def test_broadcaster_survives_state_exceptions():
    class Bad(RaceEngine):
        calls = 0

        def state(self):
            Bad.calls += 1
            if Bad.calls in (2, 3):
                raise ValueError("boom")
            return super().state()

    class Src:
        mode = "file"
        control = type("C", (), {"speed": 1.0, "paused": False, "info": lambda s: {}})()

        async def messages(self):
            i = 0
            while True:
                i += 1
                yield Msg("LapCount", {"CurrentLap": i}, float(i))
                await asyncio.sleep(0.2)

    with TestClient(create_app(Src(), mode="file", engine=Bad(), frontend_dir=None)) as c:
        with c.websocket_connect("/ws") as ws:
            got = recv_types(ws, 400, stop=lambda d: d["type"] == "state" and d.get("lap", {}).get("current", 0) >= 5)
            assert got[-1]["type"] == "state"


def test_zero_clients_is_fine_and_late_joiner_gets_full_snapshot():
    class Src:
        mode = "file"
        control = type("C", (), {"speed": 1.0, "paused": False, "info": lambda s: {}})()

        async def messages(self):
            yield Msg("DriverList", {"1": {"RacingNumber": "1", "Tla": "VER"}}, 1.0)
            for i in range(50):
                yield Msg("LapCount", {"CurrentLap": i + 1, "TotalLaps": 50}, 2.0 + i)
                await asyncio.sleep(0.02)

    with TestClient(create_app(Src(), mode="file", frontend_dir=None)) as c:
        time.sleep(1.5)
        assert c.get("/api/status").json()["clients"] == 0
        with c.websocket_connect("/ws") as ws:
            first = [json.loads(ws.receive_text()) for _ in range(3)]
        assert [d["type"] for d in first] == ["hello", "state", "history"]
        assert first[1]["lap"]["current"] >= 40 and "1" in first[1]["cars"]


def test_client_limit_rejects_extra_viewers_politely():
    class Src:
        mode = "file"
        control = type("C", (), {"speed": 1.0, "paused": False, "info": lambda s: {}})()

        async def messages(self):
            yield Msg("DriverList", {"1": {"Tla": "VER"}}, 1.0)
            await asyncio.sleep(20)

    with TestClient(create_app(Src(), mode="file", frontend_dir=None, max_clients=1)) as c:
        with c.websocket_connect("/ws") as ws1:
            assert json.loads(ws1.receive_text())["type"] == "hello"
            wait_for(lambda: c.get("/api/status").json()["clients"] == 1)
            with c.websocket_connect("/ws") as ws2:
                d = json.loads(ws2.receive_text())
                assert d["type"] == "error" and "too many" in d["message"]
            s = c.get("/api/status").json()
            assert s["rejectedClients"] == 1 and s["clients"] == 1 and s["maxClients"] == 1


def test_slow_client_is_dropped_not_waited_for(monkeypatch):
    import backend.server as S

    monkeypatch.setattr(S, "SEND_TIMEOUT", 0.2)

    class Stuck:
        closed_with = None

        async def send_text(self, t):
            await asyncio.sleep(60)

        async def close(self, code=1000):
            Stuck.closed_with = code

    async def go():
        c = Client(Stuck())
        c.put("state", "x")
        t = asyncio.create_task(c.writer())
        await asyncio.wait_for(t, 3)
        return c

    c = asyncio.run(go())
    assert c.closed and Stuck.closed_with is not None


def test_oversized_and_garbage_client_messages_are_ignored():
    class Src:
        mode = "file"
        control = type("C", (), {"speed": 1.0, "paused": False, "info": lambda s: {}})()

        async def messages(self):
            yield Msg("DriverList", {"1": {"Tla": "VER"}}, 1.0)
            await asyncio.sleep(20)

    with TestClient(create_app(Src(), mode="file", frontend_dir=None)) as c:
        with c.websocket_connect("/ws") as ws:
            ws.send_text("x" * 100000)
            ws.send_text("not json")
            ws.send_text(json.dumps([1, 2]))
            ws.send_text(json.dumps({"type": "control", "speed": "fast"}))
            recv_types(ws, 6)
        assert c.get("/healthz").json() == {"ok": True}


# ----------------------------------------------------------------------------- JSON safety
def test_safe_dumps_sanitises_nan_inf_numpy_and_keys():
    obj = {"a": float("nan"), "b": [1.0, float("inf"), -float("inf")], 3: (1, 2), "n": np.int64(5), "f": np.float64("nan"),
           "arr": np.array([1.0, np.nan]), "s": {1, 2} and {"x"}}
    out = json.loads(_safe_dumps(obj))
    assert out["a"] is None and out["b"] == [1.0, None, None] and out["3"] == [1, 2] and out["n"] == 5
    assert out["f"] is None and out["arr"] == [1.0, None] and out["s"] == ["x"]
    assert "NaN" not in _safe_dumps(obj) and "Infinity" not in _safe_dumps(obj)
    assert json.loads(_safe_dumps({"ok": 1.5})) == {"ok": 1.5}
    circ = {}
    circ["self"] = circ
    assert json.loads(_safe_dumps(circ))["type"] == "error"  # never raises


def test_nan_in_engine_state_never_breaks_rest_or_ws():
    class Nan(RaceEngine):
        def state(self):
            s = super().state()
            s["weather"] = {"air": float("nan"), "track": float("inf")}
            return s

    class Src:
        mode = "file"
        control = type("C", (), {"speed": 1.0, "paused": False, "info": lambda s: {}})()

        async def messages(self):
            yield Msg("DriverList", {"1": {"Tla": "VER"}}, 1.0)
            await asyncio.sleep(20)

    with TestClient(create_app(Src(), mode="file", engine=Nan(), frontend_dir=None)) as c:
        r = c.get("/api/state")
        assert r.status_code == 200 and r.json()["weather"] == {"air": None, "track": None}
        with c.websocket_connect("/ws") as ws:
            recv_types(ws, 3)


# ----------------------------------------------------------------------------- track reset / circuit change / fallback
def loop_msgs(n, laps, key=None, r=4000.0, t0=10.0, dt=0.25, cx=0.0):
    out = []
    t = t0
    for lap in range(laps):
        for i in range(n):
            a = 2 * math.pi * i / n
            rr = r + 0.2 * r * math.sin(3 * a)
            out.append(Msg("Position", {"Position": [{"Timestamp": "t", "Entries": {"1": {"Status": "OnTrack", "X": rr * math.cos(a) + cx, "Y": 0.7 * rr * math.sin(a), "Z": 0}}}]}, t))
            t += dt
        out.append(Msg("TimingData", {"Lines": {"1": {"NumberOfLaps": lap + 1}}}, t))
    return out


class Scripted:
    mode = "file"
    control = type("C", (), {"speed": 1.0, "paused": False, "info": lambda s: {}})()

    def __init__(self, parts):
        self.parts = parts  # list of (delay, [msgs])

    async def messages(self):
        yield Msg("DriverList", {"1": {"RacingNumber": "1", "Tla": "VER"}}, 1.0)
        for delay, msgs in self.parts:
            await asyncio.sleep(delay)
            for m in msgs:
                yield m
        await asyncio.sleep(30)


def sinfo(key, path="2025/2025-03-16_X_Grand_Prix/2025-03-16_Practice_1/", mk=1):
    return Msg("SessionInfo", {"Meeting": {"Key": mk, "Name": "X GP", "Circuit": {"Key": key}}, "Name": "Practice 1", "Type": "Practice", "Path": path}, 2.0)


def test_session_context_parsing():
    eng = RaceEngine()
    assert session_context(eng) == (None, None, None)
    eng.apply(sinfo(63))
    assert session_context(eng) == (63, 2025, 1)
    eng.apply(Msg("SessionInfo", {"Meeting": {"Circuit": {"Key": 0}}}, 3.0))
    assert session_context(eng)[0] is None  # key 0 = unknown (sim)


def test_post_track_reset_clears_and_relearns_and_viewers_get_null_track():
    src = Scripted([(0.0, loop_msgs(120, 3)), (6.0, loop_msgs(120, 3))])
    with TestClient(create_app(src, mode="file", frontend_dir=None, track_fallback=False)) as c:
        with c.websocket_connect("/ws") as ws:
            wait_for(lambda: c.get("/api/track").json()["track"] is not None)
            recv_types(ws, 400, stop=lambda d: d["type"] == "track" and d["track"] is not None)
            v = c.get("/api/status").json()["trackVersion"]
            r = c.post("/api/track/reset")
            assert r.status_code == 200 and r.json()["ok"] is True
            frames = recv_types(ws, 400, stop=lambda d: d["type"] == "track" and d["track"] is None)
            assert frames[-1]["type"] == "track" and frames[-1]["track"] is None
            assert c.get("/api/status").json()["track"]["resets"] == 1
            assert c.get("/api/status").json()["trackVersion"] > v
            # learns again from the second batch of laps
            frames = recv_types(ws, 800, stop=lambda d: d["type"] == "track" and d["track"] is not None)
            assert frames[-1]["track"]["source"] == "learned"


def test_circuit_change_resets_track_automatically():
    src = Scripted([(0.0, [sinfo(10)] + loop_msgs(120, 3)), (2.5, [sinfo(11, mk=2)])])
    with TestClient(create_app(src, mode="file", frontend_dir=None, track_fallback=False)) as c:
        wait_for(lambda: c.get("/api/track").json()["track"] is not None)
        wait_for(lambda: c.get("/api/status").json()["track"]["circuitKey"] == 10)
        s = c.get("/api/status").json()
        assert s["track"]["circuitKey"] == 10 and s["track"]["resets"] == 0
        wait_for(lambda: c.get("/api/status").json()["track"]["resets"] == 1, 10)
        s = c.get("/api/status").json()
        assert s["track"]["built"] is False and s["track"]["circuitKey"] == 11
        assert c.get("/api/track").json()["track"] is None


def test_official_layout_fallback_wired_into_server():
    pts = [(4000 * math.cos(2 * math.pi * i / 400) * (1 + 0.2 * math.sin(3 * 2 * math.pi * i / 400)),
            0.7 * 4000 * math.sin(2 * math.pi * i / 400) * (1 + 0.2 * math.sin(3 * 2 * math.pi * i / 400))) for i in range(400)]
    layout = {"x": [p[0] for p in pts], "y": [p[1] for p in pts], "rotation": 12,
              "corners": [{"number": 1, "trackPosition": {"x": pts[10][0], "y": pts[10][1]}}]}
    asked = []
    fb = OfficialLayoutFallback(lambda url, t: asked.append(url) or layout)
    # a car drives only a partial lap -> nothing to learn from, but enough samples to validate the official outline
    src = Scripted([(0.0, [sinfo(63)] + loop_msgs(400, 1)[:300])])
    with TestClient(create_app(src, mode="file", frontend_dir=None, track_fallback=fb)) as c:
        t = wait_for(lambda: c.get("/api/track").json()["track"], 10)
        assert t["source"] == "official" and t["corners"][0]["n"] == 1
        assert asked == ["https://api.multiviewer.app/api/v1/circuits/63/2025"]
        s = c.get("/api/status").json()
        assert s["track"]["source"] == "official" and s["track"]["built"] is True
        cars = c.get("/api/state").json()["cars"]
        assert cars["1"]["progress"] is not None


def test_fallback_failure_is_invisible_to_clients():
    def boom(url, timeout):
        raise OSError("network is blocked")

    src = Scripted([(0.0, [sinfo(63)] + loop_msgs(120, 3))])
    with TestClient(create_app(src, mode="file", frontend_dir=None, track_fallback=OfficialLayoutFallback(boom))) as c:
        t = wait_for(lambda: c.get("/api/track").json()["track"], 10)
        assert t["source"] == "learned"
        assert c.get("/api/status").json()["error"] is None
