"""Tests for backend.ingest.sources: FileSource timing/speed/pause/seek/loop, SimSource, LiveSource (fakes only)."""
from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from backend.ingest.messages import Msg, encode_z, format_ts, parse_line
from backend.ingest.sources import Control, FileSource, LiveSource, LiveSourceError, SimSource

BASE = 1_700_000_000.0


def line(topic: str, data, off: float | None) -> str:
    ts = "" if off is None else format_ts(BASE + off)
    return str([topic, data, ts])


def write(path: Path, lines: list[str]) -> str:
    path.write_text("\n".join(lines) + "\n")
    return str(path)


async def collect(src, limit: float = 10.0):
    """Drain a source; returns [(wall_seconds_since_start, Msg)]."""
    out = []
    t0 = time.monotonic()

    async def run():
        async for m in src.messages():
            out.append((time.monotonic() - t0, m))

    await asyncio.wait_for(run(), limit)
    return out


def simple_file(tmp_path: Path, n: int = 6, dt: float = 0.2) -> str:
    lines = [line("DriverList", {"1": {"Tla": "VER"}}, 0.0)]
    lines += [line("LapCount", {"CurrentLap": i}, i * dt) for i in range(1, n)]
    return write(tmp_path / "rec.txt", lines)


# ----------------------------------------------------------------------------- FileSource timing
def test_filesource_honours_spacing(tmp_path):
    path = simple_file(tmp_path, n=6, dt=0.2)  # 1.0 s of recording
    out = asyncio.run(collect(FileSource(path, speed=1.0)))
    assert [m.topic for _, m in out] == ["DriverList"] + ["LapCount"] * 5
    assert [m.data["CurrentLap"] for _, m in out[1:]] == [1, 2, 3, 4, 5]
    ts = [t for t, _ in out]
    assert ts[0] < 0.1
    for i in range(1, 6):  # each message is released at (not before) its scheduled time
        assert ts[i] >= i * 0.2 - 0.03
        assert ts[i] <= i * 0.2 + 0.25
    assert [m.utc for _, m in out] == pytest.approx([BASE + i * 0.2 for i in range(6)], abs=1e-3)


def test_filesource_speed_multiplier(tmp_path):
    path = simple_file(tmp_path, n=6, dt=0.4)  # 2.0 s of recording
    t0 = time.monotonic()
    asyncio.run(collect(FileSource(path, speed=10.0)))
    fast = time.monotonic() - t0
    assert 0.15 <= fast <= 0.7  # ~0.2 s
    t0 = time.monotonic()
    asyncio.run(collect(FileSource(path, speed=4.0)))
    mid = time.monotonic() - t0
    assert 0.4 <= mid <= 1.2  # ~0.5 s


def test_speed_change_mid_playback(tmp_path):
    path = simple_file(tmp_path, n=11, dt=0.5)  # 5 s of recording
    src = FileSource(path, speed=1.0)

    async def go():
        t0 = time.monotonic()
        n = 0
        async for _ in src.messages():
            n += 1
            if n == 3:  # ~1.0 s in
                src.control.speed = 50.0
        return time.monotonic() - t0

    elapsed = asyncio.run(go())
    assert elapsed < 2.0  # would be 5 s at speed 1


def test_control_validation_and_info(tmp_path):
    c = Control(2.0)
    assert c.speed == 2.0
    with pytest.raises(ValueError):
        c.speed = 0
    with pytest.raises(ValueError):
        c.speed = -1
    c.paused = True
    info = c.info()
    assert info["speed"] == 2.0 and info["paused"] is True and info["replay"] is True
    path = simple_file(tmp_path)
    src = FileSource(path, speed=100.0)
    asyncio.run(collect(src))
    i = src.control.info()
    assert i["start"] == pytest.approx(BASE) and i["end"] == pytest.approx(BASE + 1.0)
    assert i["duration"] == pytest.approx(1.0) and i["messages"] == 6


def test_live_control_seek_is_noop():
    async def go():
        c = Control(replay=False)
        await c.seek(5.0)
        return c.seek_pending

    assert asyncio.run(go()) is False


# ----------------------------------------------------------------------------- pause
def test_pause_and_resume(tmp_path):
    path = simple_file(tmp_path, n=8, dt=0.1)
    src = FileSource(path, speed=1.0)
    got: list[float] = []

    async def go():
        async def run():
            async for m in src.messages():
                got.append(time.monotonic())

        task = asyncio.create_task(run())
        await asyncio.sleep(0.25)
        src.control.paused = True
        await asyncio.sleep(0.05)
        n_paused = len(got)
        await asyncio.sleep(0.6)
        assert len(got) == n_paused, "messages were released while paused"
        assert n_paused < 8
        src.control.paused = False
        await asyncio.wait_for(task, 5)
        return n_paused

    n_paused = asyncio.run(go())
    assert len(got) == 8
    # the time spent paused must not be "caught up" in one burst: media clock stood still
    assert got[-1] - got[n_paused - 1] >= (8 - n_paused) * 0.1 * 0.6


def test_start_paused_then_play(tmp_path):
    path = simple_file(tmp_path, n=4, dt=0.05)
    src = FileSource(path, speed=1.0, paused=True)
    got = []

    async def go():
        async def run():
            async for m in src.messages():
                got.append(m)

        task = asyncio.create_task(run())
        await asyncio.sleep(0.3)
        first = len(got)
        src.control.paused = False
        await asyncio.wait_for(task, 3)
        return first

    first = asyncio.run(go())
    assert first <= 1  # only the t=0 message(s)
    assert len(got) == 4


# ----------------------------------------------------------------------------- initial messages
def test_snapshot_lines_and_first_driverlist_are_released_first(tmp_path):
    lines = [
        line("SessionInfo", {"Name": "Race"}, None),  # untimed CompletionMessage-style snapshot
        line("TimingData", {"Lines": {"1": {"Position": "1"}}}, 100.0),
        line("TimingData", {"Lines": {"1": {"Position": "2"}}}, 100.5),
        line("DriverList", {"1": {"Tla": "VER"}}, 105.0),  # shows up late in the recording
        line("DriverList", {"1": {"Tla": "XXX"}}, 106.0),
        line("TimingData", {"Lines": {"1": {"Position": "3"}}}, 106.5),
    ]
    path = write(tmp_path / "late.txt", lines)
    out = asyncio.run(collect(FileSource(path, speed=100.0)))
    topics = [m.topic for _, m in out]
    assert topics[:3] == ["SessionInfo", "DriverList", "TimingData"]
    assert out[1][1].data["1"]["Tla"] == "VER"
    assert [m.data["1"]["Tla"] for _, m in out if m.topic == "DriverList"] == ["VER", "XXX"]  # later one still plays
    assert len(out) == 6
    # the early copies don't wait for 105 s
    assert out[1][0] < 0.2
    assert out[0][1].utc == pytest.approx(BASE + 100.0)  # untimed -> inherits first timestamp


def test_completion_message_dump_header(tmp_path):
    lines = [
        str(["DriverList", '{"1": {"Tla": "VER"}}', ""]),
        str(["CarData.z", '"%s"' % encode_z({"Entries": []}), ""]),
        line("TimingData", {"Lines": {}}, 1.0),
        line("TimingData", {"Lines": {}}, 1.1),
    ]
    out = asyncio.run(collect(FileSource(write(tmp_path / "c.txt", lines), speed=100.0)))
    assert [m.topic for _, m in out] == ["DriverList", "CarData", "TimingData", "TimingData"]
    assert out[0][1].data == {"1": {"Tla": "VER"}}


def test_garbage_lines_skipped_and_unsorted_sorted(tmp_path):
    lines = [
        line("LapCount", {"CurrentLap": 2}, 0.2),
        "total garbage",
        "",
        line("LapCount", {"CurrentLap": 1}, 0.1),
        "['LapCount', {'CurrentLap': 99}, '2024-",  # truncated final write
        line("LapCount", {"CurrentLap": 3}, 0.3),
    ]
    out = asyncio.run(collect(FileSource(write(tmp_path / "g.txt", lines), speed=100.0)))
    assert [m.data["CurrentLap"] for _, m in out] == [1, 2, 3]


def test_json_lines_and_z_in_file(tmp_path):
    p = tmp_path / "j.txt"
    p.write_text(
        '{"topic": "LapCount", "data": {"CurrentLap": 1}, "utc": %r}\n' % BASE
        + str(["Position.z", encode_z({"Position": []}), format_ts(BASE + 0.1)]) + "\n"
    )
    out = asyncio.run(collect(FileSource(str(p), speed=100.0)))
    assert [(m.topic, m.data) for _, m in out] == [("LapCount", {"CurrentLap": 1}), ("Position", {"Position": []})]


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        asyncio.run(collect(FileSource(str(tmp_path / "nope.txt"))))


# ----------------------------------------------------------------------------- seek
def seek_file(tmp_path: Path) -> str:
    lines = [line("DriverList", {"1": {"Tla": "VER"}}, 0.0)]
    for i in range(1, 21):  # 20 s: CarData every second, LapCount every 5 s
        lines.append(line("CarData", {"Entries": [{"i": i}]}, float(i)))
        if i % 5 == 0:
            lines.append(line("LapCount", {"CurrentLap": i // 5}, float(i) + 0.01))
    return write(tmp_path / "seek.txt", lines)


def test_seek_forward_and_backward_replays_state(tmp_path):
    src = FileSource(seek_file(tmp_path), speed=1.0, paused=True)
    got: list[Msg] = []

    async def go():
        async def run():
            async for m in src.messages():
                got.append(m)

        task = asyncio.create_task(run())
        await asyncio.sleep(0.15)
        assert [m.topic for m in got] == ["DriverList"]
        await src.control.seek(BASE + 10.5)
        await asyncio.sleep(0.3)
        after_fwd = list(got)
        got.clear()
        await src.control.seek(BASE + 5.5)
        await asyncio.sleep(0.3)
        after_back = list(got)
        got.clear()
        src.control.paused = False
        src.control.speed = 200.0
        await asyncio.wait_for(task, 5)
        return after_fwd, after_back, list(got)

    fwd, back, rest = asyncio.run(go())
    # forward: everything from the start up to the target, nothing beyond; stale CarData dropped
    assert fwd[0].topic == "DriverList"
    assert [m.data["CurrentLap"] for m in fwd if m.topic == "LapCount"] == [1, 2]
    assert all(m.utc <= BASE + 10.5 + 1e-6 for m in fwd)
    cars = [m.data["Entries"][0]["i"] for m in fwd if m.topic == "CarData"]
    assert cars == [8, 9, 10]  # only the last 3 s before the target
    # backward: state is re-sent from the beginning (the engine reset before seeking)
    assert back[0].topic == "DriverList"
    assert [m.data["CurrentLap"] for m in back if m.topic == "LapCount"] == [1]
    assert all(m.utc <= BASE + 5.5 + 1e-6 for m in back)
    # then it plays on from the seek point
    laps = [m.data["CurrentLap"] for m in rest if m.topic == "LapCount"]
    assert laps == [2, 3, 4]
    assert info_utc_ok(src)


def info_utc_ok(src) -> bool:
    return src.control.info()["utc"] == pytest.approx(BASE + 20.01, abs=0.5)


def test_seek_clamps_to_recording(tmp_path):
    src = FileSource(seek_file(tmp_path), speed=1.0, paused=True)
    got = []

    async def go():
        async def run():
            async for m in src.messages():
                got.append(m)

        task = asyncio.create_task(run())
        await asyncio.sleep(0.1)
        await src.control.seek(BASE + 9999)
        await asyncio.sleep(0.3)
        n_all = len(got)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return n_all

    n_all = asyncio.run(go())
    assert [m.data["CurrentLap"] for m in got if m.topic == "LapCount"] == [1, 2, 3, 4]
    assert n_all >= 5


def test_restart_after_finish_honours_pending_seek(tmp_path):
    """The server calls messages() again after a replay ended and a seek arrived."""
    src = FileSource(seek_file(tmp_path), speed=1000.0)
    asyncio.run(collect(src))  # played to the end
    assert src.control.info()["utc"] == pytest.approx(BASE + 20.01, abs=0.1)

    async def go():
        await src.control.seek(BASE + 5.5)
        return await collect(src)

    out = asyncio.run(go())
    laps = [m.data["CurrentLap"] for _, m in out if m.topic == "LapCount"]
    assert laps == [1, 2, 3, 4]  # replayed from 0 up to the seek point, then on to the end
    assert out[0][1].topic == "DriverList"


def test_loop_restarts(tmp_path):
    path = simple_file(tmp_path, n=3, dt=0.05)
    src = FileSource(path, speed=1.0, loop=True)

    async def go():
        n = 0
        async for _ in src.messages():
            n += 1
            if n >= 8:
                return n

    assert asyncio.run(asyncio.wait_for(go(), 5)) == 8


# ----------------------------------------------------------------------------- SimSource
def test_simsource_streams_full_race_fast():
    src = SimSource(seed=2, speed=500.0, laps=3)
    out = asyncio.run(collect(src, limit=60))
    topics = {m.topic for _, m in out}
    assert {"DriverList", "TimingData", "CarData", "Position", "SessionStatus"} <= topics
    assert src.control.info()["laps"] == 3 and src.mode == "sim"
    utcs = [m.utc for _, m in out]
    assert utcs == sorted(utcs)


def test_simsource_roundtrip_matches_direct():
    direct = SimSource(seed=3, speed=1.0, laps=2).all_messages()[:600]
    s = SimSource(seed=3, speed=1000.0, laps=2, roundtrip=True)

    async def take():
        res = []
        async for m in s.messages():
            res.append(m)
            if len(res) == 600:
                break
        return res

    rt = asyncio.run(take())
    assert len(rt) == 600
    for a, b in zip(direct, rt):
        assert a.topic == b.topic and a.data == b.data and a.utc == pytest.approx(b.utc, abs=1e-3)


def test_simsource_pacing():
    src = SimSource(seed=1, speed=1.0, laps=2, pre_race_s=2.0)

    async def go():
        t0 = time.monotonic()
        stamp = {}
        async for m in src.messages():
            stamp.setdefault(m.topic, (time.monotonic() - t0, m.utc))
            if m.topic == "SessionStatus" and m.data["Status"] == "Started":
                return time.monotonic() - t0, m.utc - src.race.times[0]

    wall, media = asyncio.run(asyncio.wait_for(go(), 10))
    assert media == pytest.approx(2.0, abs=0.05)
    assert 1.9 <= wall <= 3.0


# ----------------------------------------------------------------------------- LiveSource (fakes)
import backend.ingest.sources as S  # noqa: E402


class FakeClient:
    """Stands in for fastf1's SignalRClient: start() appends lines, then blocks until the runner "kicks" it the
    same way it kicks the real client (``timeout`` lowered to 1)."""

    def __init__(self, filename, filemode="a", timeout=60, no_auth=False, lines=(), delay=0.0, hold=True):
        self.filename, self.filemode, self.timeout, self.no_auth = filename, filemode, timeout, no_auth
        self.lines, self.delay, self.hold = list(lines), delay, hold
        self._t_last_message = time.time()

    def write_lines(self):
        with open(self.filename, self.filemode) as f:
            for ln in self.lines:
                f.write(ln + "\n")
                f.flush()
                if self.delay:
                    time.sleep(self.delay)

    def wait_kick(self, limit=20.0):
        t0 = time.time()
        while self.hold and self.timeout != 1 and time.time() - t0 < limit:
            time.sleep(0.01)

    def start(self):
        self.write_lines()
        self.wait_kick()


def mk(tmp_path, factory, **kw):
    kw.setdefault("poll", 0.01)
    kw.setdefault("watch_interval", 0.02)
    kw.setdefault("retry_delay", 0.05)
    kw.setdefault("max_delay", 0.2)
    return LiveSource(recordings_dir=str(tmp_path / "rec"), client_factory=factory, **kw)


async def first(src, n, limit=15.0, pred=None):
    got = []

    async def go():
        async for m in src.messages():
            got.append(m)
            if pred(m, got) if pred else len(got) >= n:
                return

    await asyncio.wait_for(go(), limit)
    return got


def test_livesource_tails_recording_and_stops(tmp_path):
    lines = [
        str(["DriverList", '{"1": {"Tla": "VER"}}', ""]),  # CompletionMessage-style dump first
        line("TimingData", {"Lines": {"1": {"Position": "1"}}}, 1.0),
        str(["CarData.z", encode_z({"Entries": [{"Utc": format_ts(BASE), "Cars": {}}]}), format_ts(BASE + 1.1)]),
    ]
    made = []

    def factory(filename, filemode, timeout, no_auth):
        c = FakeClient(filename, filemode, timeout, no_auth, lines=lines)
        made.append(c)
        return c

    src = mk(tmp_path, factory, no_auth=True)
    got = asyncio.run(first(src, 3))
    assert [m.topic for m in got] == ["DriverList", "TimingData", "CarData"]
    assert got[0].data == {"1": {"Tla": "VER"}}
    assert made and made[0].no_auth is True and made[0].filemode == "a"
    assert made[0].timeout in (0, 1)  # the client's own idle exit is disabled; the runner owns the policy
    files = list((tmp_path / "rec").glob("*.txt"))
    assert len(files) == 1
    replay = [parse_line(l) for l in files[0].read_text().splitlines()]
    assert [m.topic for m in replay] == ["DriverList", "TimingData", "CarData"]
    info = src.control.info()
    assert info["recording"] == str(files[0]) and src.control.replay is False
    assert info["recordingBytes"] == files[0].stat().st_size
    src.close()
    assert not src.runner.is_alive()


def test_livesource_partial_lines_are_not_emitted(tmp_path):
    path = tmp_path / "live.txt"
    first_l = line("LapCount", {"CurrentLap": 1}, 1.0)
    second = line("LapCount", {"CurrentLap": 2}, 2.0)

    class Slow(FakeClient):
        def start(self):
            with open(self.filename, "a") as f:
                f.write(first_l + "\n" + second[:20])
                f.flush()
                time.sleep(0.3)
                f.write(second[20:] + "\n")
                f.flush()
            self.wait_kick()

    src = LiveSource(filename=str(path), client_factory=lambda *a: Slow(*a), poll=0.01, watch_interval=0.02)
    got = asyncio.run(first(src, 2))
    src.close()
    assert [m.data["CurrentLap"] for m in got] == [1, 2]


def test_tail_handles_flush_boundaries_crlf_and_truncation(tmp_path):
    p = tmp_path / "t.txt"
    p.write_bytes(b"")
    t = S._Tail(str(p))
    assert t.read() == []
    with open(p, "ab") as f:
        f.write(b"abc\r\nde")
        f.flush()
        assert t.read() == [b"abc\r"]
        f.write(b"f\n\ngh")
        f.flush()
        assert t.read() == [b"def", b""]
        f.write(b"i\n")
        f.flush()
        assert t.read() == [b"ghi"]
    p.write_bytes(b"x\n")  # truncated + rewritten shorter than our position
    got = t.read() + t.read()
    assert got in ([], [b"x"])
    t.close()
    # multibyte char split across two reads is only decoded once the line is complete
    p.write_bytes(b"")
    t = S._Tail(str(p))
    snow = "Hülkenberg".encode()
    with open(p, "ab") as f:
        f.write(snow[:2])
        f.flush()
        assert t.read() == []
        f.write(snow[2:] + b"\n")
        f.flush()
        assert t.read() == [snow]
    t.close()


def test_livesource_network_error_never_gives_up_and_is_explained(tmp_path):
    calls = []

    def factory(*a):
        calls.append(a)
        if len(calls) < 4:
            raise ConnectionError("HTTPSConnectionPool(host='livetiming.formula1.com'): Max retries exceeded")
        return FakeClient(*a, lines=[line("LapCount", {"CurrentLap": 9}, 1.0)])

    src = mk(tmp_path, factory)
    seen = {}

    def pred(m, got):
        return True

    async def go():
        async def run():
            async for m in src.messages():
                return m

        task = asyncio.create_task(run())
        for _ in range(200):
            await asyncio.sleep(0.01)
            i = src.control.info()
            if i["error"]:
                seen.update(i)
                break
        m = await asyncio.wait_for(task, 10)
        return m

    m = asyncio.run(go())
    assert m.data == {"CurrentLap": 9}
    assert len(calls) == 4  # failed 3 times (more than the old max_failures) and kept going
    assert "Cannot reach livetiming.formula1.com" in seen["error"] and "firewall" in seen["error"]
    assert seen["phase"] == "error" and seen["message"] == seen["error"]
    src.close()


def test_livesource_auth_check_failure_then_success(tmp_path):
    results = ["F1 sign-in failed (F1AuthError: bad password)", "No valid F1 TV token", None]
    made = []

    def auth():
        return results.pop(0) if results else None

    def factory(*a):
        c = FakeClient(*a, lines=[line("LapCount", {"CurrentLap": 1}, 1.0)])
        made.append(c)
        return c

    src = mk(tmp_path, factory, auth_check=auth)
    errs = []

    async def go():
        async def run():
            async for m in src.messages():
                return m

        task = asyncio.create_task(run())
        while not task.done():
            e = src.control.info()["error"]
            if e and e not in errs:
                errs.append(e)
            await asyncio.sleep(0.005)
        return task.result()

    m = asyncio.run(asyncio.wait_for(go(), 15))
    assert m.topic == "LapCount" and len(made) == 1  # no client is built until auth passes
    assert any("sign-in failed" in e for e in errs)
    assert src.control.info()["authState"] == "ok"
    src.close()


def test_livesource_no_auth_skips_auth_check(tmp_path):
    called = []
    src = mk(tmp_path, lambda *a: FakeClient(*a, lines=[line("LapCount", {"CurrentLap": 1}, 1.0)]), no_auth=True,
             auth_check=lambda: called.append(1) or "nope")
    asyncio.run(first(src, 1))
    src.close()
    assert called == [] and src.control.info()["authState"] == "skipped"


def test_livesource_waits_for_late_session_without_error_or_giving_up(tmp_path):
    """Feed connects (snapshot + heartbeats) 'before the session': stay connected, phase waiting, no error."""
    snap = [str(["SessionInfo", '{"Meeting": {"Name": "Test GP"}, "Name": "Practice 1", "Type": "Practice"}', ""]),
            str(["SessionStatus", '{"Status": "Inactive"}', ""])]
    runs = []

    class Late(FakeClient):
        def start(self):
            runs.append(time.time())
            with open(self.filename, "a") as f:
                for ln in snap:
                    f.write(ln + "\n")
                f.flush()
                for i in range(6):  # a few heartbeats, then nothing for a while
                    time.sleep(0.05)
                    f.write(line("Heartbeat", {"Utc": format_ts(BASE + i)}, i) + "\n")
                    f.flush()
                time.sleep(1.0)  # "session not started": silence
                f.write(line("SessionStatus", {"Status": "Started"}, 20.0) + "\n")
                f.write(line("TimingData", {"Lines": {}}, 21.0) + "\n")
                f.flush()
            self.wait_kick()

    src = mk(tmp_path, lambda *a: Late(*a), timeout=30, waiting_timeout=30)
    phases = []

    async def go():
        async def run():
            async for m in src.messages():
                if m.topic == "TimingData":
                    return

        task = asyncio.create_task(run())
        while not task.done():
            i = src.control.info()
            phases.append((i["phase"], i["error"], i["message"]))
            await asyncio.sleep(0.05)

    asyncio.run(asyncio.wait_for(go(), 15))
    waiting = [p for p in phases if p[0] == "waiting"]
    assert waiting and all(p[1] is None for p in phases)
    assert any("waiting for the session" in p[2] and "Practice 1" in p[2] for p in waiting)
    assert phases[-1][0] == "live" or src.control.info()["phase"] == "live"
    assert len(runs) == 1  # never reconnected while waiting
    src.close()


def test_livesource_mid_stream_drop_reconnects_and_keeps_recording(tmp_path):
    n = {"runs": 0}

    class Flaky(FakeClient):
        def start(self):
            n["runs"] += 1
            with open(self.filename, "a") as f:
                f.write(line("LapCount", {"CurrentLap": n["runs"]}, float(n["runs"])) + "\n")
            # returns immediately -> a dropped connection; LiveSource must start a new client in the same file

    src = mk(tmp_path, lambda *a: Flaky(*a))
    got = asyncio.run(first(src, 4))
    src.close()
    assert [m.data["CurrentLap"] for m in got] == [1, 2, 3, 4]
    assert len(list((tmp_path / "rec").glob("*.txt"))) == 1


def test_livesource_detects_closed_websocket_while_start_keeps_blocking(tmp_path):
    runs = []

    class Dropper(FakeClient):
        _is_connected = False

        def start(self):
            runs.append(self)
            self._is_connected = True
            with open(self.filename, "a") as f:
                f.write(line("LapCount", {"CurrentLap": len(runs)}, float(len(runs))) + "\n")
            time.sleep(0.15)
            self._is_connected = False  # websocket closed; the real supervise loop would idle until its timeout
            self.wait_kick()

    src = mk(tmp_path, lambda *a: Dropper(*a), drop_grace=0.1)
    got = asyncio.run(first(src, 3))
    src.close()
    assert [m.data["CurrentLap"] for m in got] == [1, 2, 3]


def test_livesource_connect_hang_is_recycled(tmp_path):
    """fastf1's _run() spins `while not self._is_connected` forever when the websocket never opens."""
    runs = []

    class Hang(FakeClient):
        _is_connected = False

        def start(self):
            runs.append(self)
            if len(runs) >= 3:
                self._is_connected = True
                self.write_lines()
                self.wait_kick()
                return
            while not self._is_connected:
                time.sleep(0.01)
            # released by the runner's kick -> behaves like a failed run (nothing written)

    src = mk(tmp_path, lambda *a: Hang(*a, lines=[line("LapCount", {"CurrentLap": 5}, 1.0)]), connect_timeout=0.2)
    got = asyncio.run(first(src, 1))
    assert got[0].data == {"CurrentLap": 5} and len(runs) == 3
    src.close()


def test_livesource_abandons_client_thread_that_ignores_kick(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "KICK_GRACE", 0.3)
    runs = []

    class Deaf(FakeClient):
        _is_connected = True

        def start(self):
            runs.append(self)
            if len(runs) == 1:
                time.sleep(3)  # ignores timeout/stop entirely
                return
            self.write_lines()
            self.wait_kick()

    src = mk(tmp_path, lambda *a: Deaf(*a, lines=[line("LapCount", {"CurrentLap": 1}, 1.0)]), timeout=0.2, waiting_timeout=0.2)
    got = asyncio.run(first(src, 1))
    src.close()
    assert got and len(runs) >= 2


def test_livesource_idle_feed_that_was_live_is_recycled(tmp_path):
    runs = []

    class Dies(FakeClient):
        def start(self):
            runs.append(1)
            self.write_lines()  # one live line, then total silence (half-open socket)
            self.wait_kick()

    src = mk(tmp_path, lambda *a: Dies(*a, lines=[line("LapCount", {"CurrentLap": len(runs) + 1}, 1.0)]),
             timeout=0.3, waiting_timeout=60)
    got = asyncio.run(first(src, 2))
    src.close()
    assert len(got) == 2 and len(runs) >= 2


def test_livesource_survives_systemexit_and_garbage(tmp_path):
    runs = []

    class Exits(FakeClient):
        def start(self):
            runs.append(1)
            if len(runs) == 1:
                raise SystemExit(1)
            with open(self.filename, "a") as f:
                f.write("this is not a feed line\n")
                f.write(line("LapCount", {"CurrentLap": 7}, 1.0) + "\n")
            self.wait_kick()

    src = mk(tmp_path, lambda *a: Exits(*a))
    got = asyncio.run(first(src, 1))
    assert got[0].data == {"CurrentLap": 7} and src.bad_lines == 1
    src.close()


def test_livesource_terminates_partial_line_left_by_previous_client(tmp_path):
    runs = []

    class Killed(FakeClient):
        def start(self):
            runs.append(1)
            with open(self.filename, "a") as f:
                if len(runs) == 1:
                    f.write(line("LapCount", {"CurrentLap": 1}, 1.0)[:25])  # process died mid-write
                    return
                f.write(line("LapCount", {"CurrentLap": 2}, 2.0) + "\n")
            self.wait_kick()

    src = mk(tmp_path, lambda *a: Killed(*a))
    got = asyncio.run(first(src, 1))
    src.close()
    assert got[0].data == {"CurrentLap": 2}  # not glued onto the partial line


def test_livesource_missing_fastf1_is_fatal_but_does_not_raise_or_spin(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "fastf1", None)
    monkeypatch.setitem(sys.modules, "fastf1.livetiming", None)
    monkeypatch.setitem(sys.modules, "fastf1.livetiming.client", None)
    src = LiveSource(recordings_dir=str(tmp_path), retry_delay=0.01, poll=0.01, watch_interval=0.02)

    async def go():
        t = asyncio.create_task(first(src, 1, limit=30))
        await asyncio.sleep(0.5)
        i = src.control.info()
        t.cancel()
        return i

    i = asyncio.run(go())
    assert "fastf1" in i["error"] and i["phase"] == "error"
    assert len(list(tmp_path.glob("*.txt"))) == 1  # no recording-file litter per retry


def test_import_does_not_require_fastf1():
    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; sys.modules['fastf1'] = None; sys.path.insert(0, %r);"
        "import backend.ingest.sources as s, backend.ingest.messages, backend.sim;"
        "assert s.LiveSource(no_auth=True).mode == 'live'; print('ok')" % str(root)
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


# ----------------------------------------------------------------------------- the real fastf1 client (offline)
def test_default_factory_builds_real_client_without_network(tmp_path):
    pytest.importorskip("fastf1")
    from fastf1.livetiming.client import SignalRClient

    c = S._default_client_factory(str(tmp_path / "x.txt"), "a", 5, True)
    assert isinstance(c, SignalRClient) and c.timeout == 5 and c._no_auth is True


def _real_clients(tmp_path, encoding=None):
    pytest.importorskip("fastf1")
    from fastf1.livetiming.client import SignalRClient

    a = S._default_client_factory(str(tmp_path / "ours.txt"), "a", 0, True)
    b = SignalRClient(str(tmp_path / "stock.txt"), filemode="a", no_auth=True, logger=logging.getLogger("stock"))
    a._output_file = open(tmp_path / "ours.txt", "a", encoding=encoding)
    b._output_file = open(tmp_path / "stock.txt", "a")
    return a, b


import logging  # noqa: E402


def test_recording_client_formats_exactly_like_stock_fastf1(tmp_path):
    from signalrcore.messages.completion_message import CompletionMessage

    a, b = _real_clients(tmp_path)
    msgs = [
        ["Heartbeat", {"Utc": "2025-03-16T04:00:00.123Z", "_kf": True}, "2025-03-16T04:00:00.123Z"],
        ["TimingData", {"Lines": {"1": {"InPit": False, "Sectors": {"0": {"Value": "28.1", "Stopped": True}}}}}, "2025-03-16T04:00:01.1Z"],
        ["CarData.z", encode_z({"Entries": []}), "2025-03-16T04:00:01.1Z"],
        CompletionMessage("1", {"Heartbeat": {"Utc": "x"}, "CarData.z": encode_z({"Entries": []}), "DriverList": {"1": {"Tla": "VER"}}}, None),
    ]
    for m in msgs:
        a._on_message(m)
        b._on_message(m)
    a._output_file.close()
    b._output_file.close()
    assert (tmp_path / "ours.txt").read_bytes() == (tmp_path / "stock.txt").read_bytes()
    parsed = [parse_line(l) for l in (tmp_path / "ours.txt").read_text().splitlines()]
    assert parsed[0].topic == "Heartbeat" and parsed[1].data["Lines"]["1"]["InPit"] is False
    snap = parsed[3:]
    assert [p.topic for p in snap][-1] == "DriverList" and snap[-1].data == {"1": {"Tla": "VER"}}


def test_recording_client_writes_utf8_even_with_ascii_locale_file(tmp_path):
    a, _ = _real_clients(tmp_path, encoding="ascii")  # stock client would raise UnicodeEncodeError and drop the line
    a._on_message(["RaceControlMessages", {"Messages": [{"Message": "CAR 27 (HÜLKENBERG) TIME 1:29.1 DELETED"}]}, "2025-03-16T04:00:01.1Z"])
    a._output_file.close()
    m = parse_line((tmp_path / "ours.txt").read_bytes().splitlines()[0])
    assert "HÜLKENBERG" in m.data["Messages"][0]["Message"]


def test_recording_client_reports_rejected_subscription(tmp_path):
    from signalrcore.messages.completion_message import CompletionMessage

    a, _ = _real_clients(tmp_path)
    a._on_message(CompletionMessage("1", None, "Unauthorized"))
    assert "rejected the subscription" in a.feed_error and "Unauthorized" in a.feed_error
    # a malformed completion (no result, no error) must not raise either
    a._on_message(CompletionMessage("1", None, None))
    a._output_file.close()
    assert (tmp_path / "ours.txt").read_bytes() == b""


def test_default_auth_check_never_prompts_and_explains(tmp_path, monkeypatch):
    pytest.importorskip("fastf1")
    acct = S._f1_account()
    monkeypatch.delenv("F1_USERNAME", raising=False)
    monkeypatch.delenv("F1_PASSWORD", raising=False)
    monkeypatch.setattr(acct.token_manager, "get_stored", lambda: None)
    called = []
    monkeypatch.setattr(acct, "get_token", lambda *a, **k: called.append(1))
    monkeypatch.setattr("builtins.input", lambda *a: (_ for _ in ()).throw(AssertionError("prompted")))
    msg = S._default_auth_check()
    assert "F1_USERNAME" in msg and "--authenticate" in msg and called == []
    # credentials present: signs in through fastf1 (faked), None = ok
    monkeypatch.setenv("F1_USERNAME", "u")
    monkeypatch.setenv("F1_PASSWORD", "p")
    assert S._default_auth_check() is None and called == [1]
    # sign-in failure is reported, with no secret in the text
    from fastf1.exceptions import F1AuthError

    def boom(*a, **k):
        raise F1AuthError("Sign-in failed: bad credentials")

    monkeypatch.setattr(acct, "get_token", boom)
    msg = S._default_auth_check()
    assert "sign-in failed" in msg.lower() and "p" != msg


# ----------------------------------------------------------------------------- long run / throughput
def test_tail_throughput_200k_lines_two_hour_session(tmp_path):
    """A 2 h session is ~200k lines.  The tailer must keep up with a fast writer and stay linear."""
    t = encode_z({"Entries": [{"Utc": format_ts(BASE), "Cars": {str(i): {"Channels": {"0": 11000, "2": 300 + i}} for i in range(1, 21)}}]})
    templ = [
        str(["CarData.z", t, format_ts(BASE + 5)]),
        str(["Position.z", t, format_ts(BASE + 5)]),
        line("TimingData", {"Lines": {"1": {"Position": "1", "GapToLeader": "+1.2"}}}, 5.0),
        line("Heartbeat", {"Utc": format_ts(BASE + 5)}, 5.0),
    ]
    N = 200_000

    class Writer(FakeClient):
        def start(self):
            with open(self.filename, "a") as f:
                for i in range(0, N, 5000):
                    f.write("\n".join(templ[(j % 4)] for j in range(i, i + 5000)) + "\n")
                    f.flush()
                    time.sleep(0.001)
            self.wait_kick()

    src = mk(tmp_path, lambda *a: Writer(*a))
    t0 = time.monotonic()

    async def go():
        n = 0
        lags = []
        async for m in src.messages():
            n += 1
            if n % 20000 == 0:
                lags.append(time.monotonic())
            if n >= N:
                return n

    n = asyncio.run(asyncio.wait_for(go(), 240))
    dt = time.monotonic() - t0
    src.close()
    assert n == N
    assert dt < 120, "tailing 200k lines took %.1fs" % dt


def test_event_loop_stays_responsive_while_tailing_a_backlog(tmp_path):
    t = encode_z({"Entries": []})
    lines = [str(["CarData.z", t, format_ts(BASE + i * 0.01)]) for i in range(30000)]
    p = tmp_path / "rec" / "x.txt"
    p.parent.mkdir()
    p.write_text("\n".join(lines) + "\n")
    src = LiveSource(filename=str(p), client_factory=lambda *a: FakeClient(*a), poll=0.01, watch_interval=0.02)
    gaps = []

    async def go():
        async def ticker():
            last = time.monotonic()
            while True:
                await asyncio.sleep(0.01)
                now = time.monotonic()
                gaps.append(now - last)
                last = now

        tk = asyncio.create_task(ticker())
        n = 0
        async for m in src.messages():
            n += 1
            if n >= 30000:
                break
        tk.cancel()
        return n

    assert asyncio.run(asyncio.wait_for(go(), 60)) == 30000
    src.close()
    assert max(gaps) < 1.0


# ----------------------------------------------------------------------------- real fastf1 client driven through a fake hub
class _FakeConn:
    """Replaces signalrcore's HubConnection inside fastf1's client: opens (or never opens), answers Subscribe with a
    snapshot CompletionMessage and then streams ``feed`` invocations - the shapes the real hub produces."""

    instances: list = []

    def __init__(self, open_ok=True, feed=()):
        self.open_ok, self.feed_msgs = open_ok, list(feed)
        self.cb = {}
        self.stopped = False
        _FakeConn.instances.append(self)

    def on_open(self, cb):
        self.cb["open"] = cb

    def on_close(self, cb):
        self.cb["close"] = cb

    def on(self, ev, cb):
        self.cb[ev] = cb

    def start(self):
        if self.open_ok:
            threading.Timer(0.05, self.cb["open"]).start()

    def send(self, method, args, on_invocation=None):
        if method != "Subscribe":
            raise AssertionError(method)
        from signalrcore.messages.completion_message import CompletionMessage

        def pump():
            on_invocation(CompletionMessage("1", {"SessionStatus": {"Status": "Inactive"}, "Heartbeat": {"Utc": "x"}}, None))
            for m in self.feed_msgs:
                time.sleep(0.02)
                self.cb["feed"](m)

        threading.Thread(target=pump, daemon=True).start()

    def stop(self):
        self.stopped = True
        if "close" in self.cb:
            self.cb["close"]()


def _patch_hub(monkeypatch, conns):
    pytest.importorskip("fastf1")
    import fastf1.livetiming.client as fc

    class Builder:
        def with_url(self, *a, **k):
            return self

        def configure_logging(self, *a, **k):
            return self

        def build(self):
            return conns.pop(0)

    class Resp:
        cookies = {"AWSALBCORS": "abc"}

    monkeypatch.setattr(fc, "HubConnectionBuilder", Builder)
    monkeypatch.setattr(fc.requests, "options", lambda *a, **k: Resp())


def test_real_fastf1_client_through_fake_hub_end_to_end(tmp_path, monkeypatch):
    feed = [["Heartbeat", {"Utc": "2025-03-16T04:00:00.123Z"}, "2025-03-16T04:00:00.123Z"],
            ["SessionStatus", {"Status": "Started"}, "2025-03-16T04:00:01.000Z"],
            ["TimingData", {"Lines": {"1": {"Position": "1"}}}, "2025-03-16T04:00:02.000Z"]]
    conn = _FakeConn(feed=feed)
    _patch_hub(monkeypatch, [conn])
    src = LiveSource(no_auth=True, recordings_dir=str(tmp_path), poll=0.01, watch_interval=0.05)
    got = asyncio.run(first(src, 5))
    assert [m.topic for m in got] == ["SessionStatus", "Heartbeat", "Heartbeat", "SessionStatus", "TimingData"]
    assert src.control.info()["phase"] == "live" and src.control.info()["connected"] is True
    t0 = time.time()
    src.close()  # real SignalRClient._supervise exits on its next 1 s tick after our kick
    assert time.time() - t0 < 5 and not src.runner.is_alive()
    assert conn.stopped
    rec = list(tmp_path.glob("*.txt"))[0]
    assert [parse_line(l).topic for l in rec.read_text().splitlines()][-1] == "TimingData"


def test_real_fastf1_client_connect_hang_is_broken_and_retried(tmp_path, monkeypatch):
    bad = _FakeConn(open_ok=False)  # websocket never opens -> stock _run() would spin forever
    good = _FakeConn(feed=[["Heartbeat", {"Utc": "x"}, "2025-03-16T04:00:00.123Z"]])
    _patch_hub(monkeypatch, [bad, good])
    src = LiveSource(no_auth=True, recordings_dir=str(tmp_path), poll=0.01, watch_interval=0.05, connect_timeout=0.4,
                     retry_delay=0.05)
    got = asyncio.run(first(src, 1, pred=lambda m, g: m.topic == "Heartbeat"))
    assert got and bad.stopped
    src.close()
