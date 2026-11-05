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
class FakeClient:
    """Stands in for fastf1's SignalRClient: start() blocks, appending lines, until timeout is lowered."""

    def __init__(self, filename, filemode="a", timeout=60, no_auth=False, lines=(), delay=0.02, hold=True):
        self.filename, self.filemode, self.timeout, self.no_auth = filename, filemode, timeout, no_auth
        self.lines, self.delay, self.hold = list(lines), delay, hold
        self._t_last_message = time.time()

    def start(self):
        with open(self.filename, self.filemode) as f:
            for ln in self.lines:
                f.write(ln + "\n")
                f.flush()
                time.sleep(self.delay)
        t0 = time.time()
        while self.hold and self.timeout > 1 and time.time() - t0 < 10:
            time.sleep(0.02)


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

    src = LiveSource(no_auth=True, recordings_dir=str(tmp_path / "rec"), client_factory=factory, poll=0.01)

    async def go():
        got = []
        async for m in src.messages():
            got.append(m)
            if len(got) == 3:
                break
        return got

    got = asyncio.run(asyncio.wait_for(go(), 10))
    assert [m.topic for m in got] == ["DriverList", "TimingData", "CarData"]
    assert got[0].data == {"1": {"Tla": "VER"}}
    assert made and made[0].no_auth is True and made[0].filemode == "a"
    # the raw feed is teed to recordings/<ts>.txt and is itself replayable
    files = list((tmp_path / "rec").glob("*.txt"))
    assert len(files) == 1
    replay = [parse_line(l) for l in files[0].read_text().splitlines()]
    assert [m.topic for m in replay] == ["DriverList", "TimingData", "CarData"]
    assert src.control.info()["recording"] == str(files[0]) and src.control.replay is False
    time.sleep(1.3)  # runner thread honours stop() (client timeout lowered) and exits
    assert not src.runner.is_alive()


def test_livesource_partial_lines_are_not_emitted(tmp_path):
    path = tmp_path / "live.txt"
    first = line("LapCount", {"CurrentLap": 1}, 1.0)
    second = line("LapCount", {"CurrentLap": 2}, 2.0)

    class Slow(FakeClient):
        def start(self):
            with open(self.filename, "a") as f:
                f.write(first + "\n" + second[:20])
                f.flush()
                time.sleep(0.3)
                f.write(second[20:] + "\n")
                f.flush()
            while self.timeout > 1:
                time.sleep(0.02)

    src = LiveSource(filename=str(path), client_factory=lambda *a: Slow(*a), poll=0.01)

    async def go():
        got = []
        async for m in src.messages():
            got.append(m)
            if len(got) == 2:
                return got

    got = asyncio.run(asyncio.wait_for(go(), 10))
    assert [m.data["CurrentLap"] for m in got] == [1, 2]


def test_livesource_network_error_is_explained(tmp_path):
    calls = []

    def factory(*a):
        calls.append(a)
        raise ConnectionError("HTTPSConnectionPool(host='livetiming.formula1.com'): Max retries exceeded")

    src = LiveSource(recordings_dir=str(tmp_path), client_factory=factory, retry_delay=0.01, max_failures=2, poll=0.01)
    with pytest.raises(LiveSourceError) as ei:
        asyncio.run(asyncio.wait_for(collect(src), 10))
    msg = str(ei.value)
    assert "Cannot reach livetiming.formula1.com" in msg and "firewall" in msg
    assert len(calls) >= 2  # it retried before giving up
    assert src.control.info()["error"] == msg


def test_livesource_auth_error_is_explained(tmp_path):
    class F1AuthError(Exception):
        pass

    def factory(*a):
        raise F1AuthError("invalid credentials")

    src = LiveSource(recordings_dir=str(tmp_path), client_factory=factory, retry_delay=0.01, max_failures=1, poll=0.01)
    with pytest.raises(LiveSourceError) as ei:
        asyncio.run(asyncio.wait_for(collect(src), 10))
    assert "authentication failed" in str(ei.value) and "no_auth" in str(ei.value)
    # with no_auth the hint about credentials is not repeated
    src2 = LiveSource(recordings_dir=str(tmp_path), no_auth=True, client_factory=factory, retry_delay=0.01, max_failures=1)
    with pytest.raises(LiveSourceError) as ei2:
        asyncio.run(asyncio.wait_for(collect(src2), 10))
    assert "authentication failed" in str(ei2.value) and "F1_USERNAME" not in str(ei2.value)


def test_livesource_http_401_negotiate_error(tmp_path):
    def factory(*a):
        raise RuntimeError("401 Client Error: Unauthorized for url: https://livetiming.formula1.com/signalrcore/negotiate")

    src = LiveSource(recordings_dir=str(tmp_path), client_factory=factory, retry_delay=0.01, max_failures=1)
    with pytest.raises(LiveSourceError, match="authentication failed"):
        asyncio.run(asyncio.wait_for(collect(src), 10))


def test_livesource_no_data_restarts_then_errors(tmp_path):
    """Client returns (its own no-data timeout) without writing anything: restart, then a clear error."""
    runs = []

    class Silent(FakeClient):
        def start(self):
            runs.append(1)
            open(self.filename, "a").close()

    src = LiveSource(recordings_dir=str(tmp_path), timeout=7, client_factory=lambda *a: Silent(*a),
                     retry_delay=0.01, max_failures=3, poll=0.01)
    with pytest.raises(LiveSourceError, match="No data received"):
        asyncio.run(asyncio.wait_for(collect(src), 10))
    assert len(runs) >= 3


def test_livesource_reconnects_after_data_was_seen(tmp_path):
    n = {"runs": 0}

    class Flaky(FakeClient):
        def start(self):
            n["runs"] += 1
            with open(self.filename, "a") as f:
                f.write(line("LapCount", {"CurrentLap": n["runs"]}, float(n["runs"])) + "\n")
            # returns immediately -> supervisor "timeout"; LiveSource must start a new client in the same file

    src = LiveSource(recordings_dir=str(tmp_path), client_factory=lambda *a: Flaky(*a), retry_delay=0.05,
                     max_failures=1, poll=0.01)

    async def go():
        got = []
        async for m in src.messages():
            got.append(m.data["CurrentLap"])
            if len(got) == 3:
                return got

    assert asyncio.run(asyncio.wait_for(go(), 10)) == [1, 2, 3]


def test_livesource_missing_fastf1_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "fastf1", None)
    monkeypatch.setitem(sys.modules, "fastf1.livetiming", None)
    monkeypatch.setitem(sys.modules, "fastf1.livetiming.client", None)
    src = LiveSource(recordings_dir=str(tmp_path), retry_delay=0.01, poll=0.01)
    with pytest.raises(LiveSourceError, match="fastf1"):
        asyncio.run(asyncio.wait_for(collect(src), 10))


def test_import_does_not_require_fastf1():
    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; sys.modules['fastf1'] = None; sys.path.insert(0, %r);"
        "import backend.ingest.sources as s, backend.ingest.messages, backend.sim;"
        "assert s.LiveSource(no_auth=True).mode == 'live'; print('ok')" % str(root)
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_default_factory_builds_real_client_without_network(tmp_path):
    """fastf1 3.x is installed here: constructing (not starting) the real client must work offline."""
    pytest.importorskip("fastf1")
    from backend.ingest.sources import _default_client_factory

    c = _default_client_factory(str(tmp_path / "x.txt"), "a", 5, True)
    assert type(c).__name__ == "SignalRClient" and c.timeout == 5
