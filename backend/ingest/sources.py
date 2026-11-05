"""Message sources: ``LiveSource`` (F1 SignalR via fastf1), ``FileSource`` (recording replay) and
``SimSource`` (synthetic race).

All sources expose ``async def messages(self) -> AsyncIterator[Msg]`` and a ``.control`` object
(:class:`Control`) with ``speed``, ``paused``, ``async seek(utc)`` and ``info()``.

Replay semantics (File/Sim)
---------------------------
* Messages are released when the *media clock* reaches their timestamp.  The media clock advances at
  ``control.speed`` x real time and stops while ``control.paused``.
* Leading un-timestamped snapshot lines plus the first ``DriverList``/``SessionInfo`` are released
  immediately at start (no delay) so the engine has a roster before anything else.
* ``await control.seek(utc)``: consumers (the server) reset their state first, so the source re-sends
  everything from the beginning up to ``utc`` instantly (high-rate ``CarData``/``Position`` older than 3 s
  before the target are skipped), then continues in real time from ``utc``.
* ``messages()`` may be called again after it finished (the server does that after a seek); it starts
  at a pending seek target, else at the beginning.
"""
from __future__ import annotations

import asyncio
import bisect
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from .messages import Msg, parse_line, parse_ts

__all__ = ["Control", "Source", "FileSource", "LiveSource", "SimSource", "LiveSourceError"]

log = logging.getLogger(__name__)

#: high-rate topics that need not be replayed when fast-forwarding on a seek
_BULK_TOPICS = frozenset({"CarData", "Position", "Heartbeat"})
_SEEK_TELEMETRY_WINDOW = 3.0  # seconds of CarData/Position kept before a seek target
_POLL = 0.05  # control polling slice while waiting for the next message


# =============================================================================== control
class Control:
    """Playback control shared between a source and the server (``speed``/``paused`` are plain
    attributes the server may set directly)."""

    def __init__(self, speed: float = 1.0, paused: bool = False, *, replay: bool = True, loop: bool = False) -> None:
        self._speed = 1.0
        self.speed = speed
        self.paused = bool(paused)
        self.replay = replay
        self.loop = loop
        self.start_utc: Optional[float] = None
        self.end_utc: Optional[float] = None
        self.utc: Optional[float] = None  # current media time
        self._seek_target: Optional[float] = None
        self._extra: Callable[[], Dict[str, Any]] = dict

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        v = float(value)
        if not v > 0:
            raise ValueError("speed must be > 0")
        self._speed = min(v, 10000.0)

    @property
    def seek_pending(self) -> bool:
        return self._seek_target is not None

    async def seek(self, utc: float) -> None:
        """Request a jump to ``utc`` (unix seconds). Clamped to the recording; no-op for live."""
        if not self.replay:
            return
        self._seek_target = float(utc)

    def take_seek(self) -> Optional[float]:
        t, self._seek_target = self._seek_target, None
        return t

    def info(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "speed": self._speed,
            "paused": self.paused,
            "replay": self.replay,
            "seekable": self.replay,
            "loop": self.loop,
            "start": self.start_utc,
            "end": self.end_utc,
            "utc": self.utc,
            "duration": (self.end_utc - self.start_utc) if (self.start_utc is not None and self.end_utc is not None) else None,
        }
        try:
            d.update(self._extra())
        except Exception:  # noqa: BLE001
            log.exception("control info provider failed")
        return d


class Source:
    """Base class: ``mode`` ('live'|'file'|'sim'), ``control`` and ``messages()``."""

    mode = "file"
    control: Control

    async def messages(self) -> AsyncIterator[Msg]:  # pragma: no cover - interface
        raise NotImplementedError
        yield  # type: ignore[unreachable]


# =============================================================================== timelines
class _Timeline:
    """Time-sorted message index.  ``times``/``topics`` are parallel lists; ``get(i)`` materialises."""

    times: List[float]
    topics: List[str]
    initial: List[int]

    def __len__(self) -> int:
        return len(self.times)

    @property
    def start(self) -> float:
        return self.times[0] if self.times else 0.0

    @property
    def end(self) -> float:
        return self.times[-1] if self.times else 0.0

    def get(self, i: int) -> Optional[Msg]:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass


class _ListTimeline(_Timeline):
    def __init__(self, msgs: List[Msg]) -> None:
        order = sorted(range(len(msgs)), key=lambda i: msgs[i].utc)
        self._msgs = [msgs[i] for i in order]
        self.times = [m.utc for m in self._msgs]
        self.topics = [m.topic for m in self._msgs]
        self.initial = []

    def get(self, i: int) -> Optional[Msg]:
        return self._msgs[i]


_TS_TAIL = re.compile(rb"""['"](\d{4}-\d\d-\d\d[T ][0-9:.,]+(?:Z|[+-]\d\d:?\d\d)?)['"]\s*\]\s*$""")
_TOPIC_HEAD = re.compile(rb"""^\[\s*['"]([^'"]+)['"]""")
_INDEX_CACHE: Dict[Tuple[str, int, int], "_FileTimeline"] = {}


class _FileTimeline(_Timeline):
    """Index of a fastf1 recording: byte offset, timestamp and topic per line; lines are parsed lazily."""

    def __init__(self, path: str, offsets: List[int], times: List[float], topics: List[str], initial: List[int]) -> None:
        self.path = path
        self._offsets = offsets
        self.times = times
        self.topics = topics
        self.initial = initial
        self._fh: Optional[Any] = None

    @classmethod
    def build(cls, path: str) -> "_FileTimeline":
        st = os.stat(path)
        key = (os.path.abspath(path), st.st_size, int(st.st_mtime))
        hit = _INDEX_CACHE.get(key)
        if hit is not None:
            return hit
        offs: List[int] = []
        tss: List[Optional[float]] = []
        tops: List[str] = []
        pos = 0
        with open(path, "rb") as f:
            for raw in f:
                off = pos
                pos += len(raw)
                s = raw.strip()
                if not s:
                    continue
                c = s[:1]
                topic: Optional[str] = None
                ts: Optional[float] = None
                if c == b"[":
                    mt = _TOPIC_HEAD.match(s)
                    if not mt:
                        continue
                    topic = mt.group(1).decode("utf-8", "replace")
                    tail = _TS_TAIL.search(s[-64:])
                    if tail:
                        ts = parse_ts(tail.group(1).decode("ascii", "replace"))
                elif c == b"{":
                    m = parse_line(s)
                    if m is None:
                        continue
                    topic, ts = m.topic, (m.utc or None)
                else:
                    continue
                if topic.endswith(".z"):
                    topic = topic[:-2]
                offs.append(off)
                tops.append(topic)
                tss.append(ts)
        n = len(offs)
        first = next((t for t in tss if t is not None), 0.0)
        lead = 0
        while lead < n and tss[lead] is None:
            lead += 1
        times: List[float] = []
        last = first
        for t in tss:
            if t is not None:
                last = t
            times.append(last)
        is_lead = [i < lead for i in range(n)]
        # stable sort if the recording is not chronological
        if any(times[i] > times[i + 1] for i in range(n - 1)):
            order = sorted(range(n), key=times.__getitem__)
            offs = [offs[i] for i in order]
            tops = [tops[i] for i in order]
            is_lead = [is_lead[i] for i in order]
            times = [times[i] for i in order]
        initial = [i for i in range(n) if is_lead[i]]
        seen = set()
        for i, t in enumerate(tops):
            if t in ("DriverList", "SessionInfo") and t not in seen:
                seen.add(t)
                if i not in initial:
                    initial.append(i)
        initial.sort()
        tl = cls(path, offs, times, tops, initial)
        _INDEX_CACHE.clear()  # keep at most one index alive
        _INDEX_CACHE[key] = tl
        return tl

    def get(self, i: int) -> Optional[Msg]:
        if self._fh is None:
            self._fh = open(self.path, "rb")
        self._fh.seek(self._offsets[i])
        return parse_line(self._fh.readline(), default_utc=self.times[i])

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None


# =============================================================================== player
class _Player:
    def __init__(self, tl: _Timeline, control: Control, loop: bool) -> None:
        self.tl = tl
        self.ctl = control
        self.loop = loop
        self.i = 0
        self.media = tl.start

    async def _seek_to(self, target: float) -> AsyncIterator[Msg]:
        tl, times, topics = self.tl, self.tl.times, self.tl.topics
        target = min(max(target, tl.start), tl.end)
        init = set(tl.initial)
        for idx in tl.initial:
            m = tl.get(idx)
            if m is not None:
                yield m
        hi = bisect.bisect_right(times, target)
        cutoff = target - _SEEK_TELEMETRY_WINDOW
        cnt = 0
        for idx in range(hi):
            if idx in init:
                continue
            if topics[idx] in _BULK_TOPICS and times[idx] < cutoff:
                continue
            m = tl.get(idx)
            if m is not None:
                yield m
            cnt += 1
            if cnt % 400 == 0:
                await asyncio.sleep(0)
        self.i = hi
        self.media = target

    async def run(self) -> AsyncIterator[Msg]:
        tl, ctl = self.tl, self.ctl
        n = len(tl)
        if n == 0:
            return
        times = tl.times
        init = set(tl.initial)
        ctl.start_utc, ctl.end_utc = tl.start, tl.end
        pending = ctl.take_seek()
        target = tl.start if pending is None else pending
        while True:
            async for m in self._seek_to(target):
                yield m
            ctl.utc = self.media
            last = time.monotonic()
            restart = False
            while not restart:
                tgt = ctl.take_seek()
                if tgt is not None:
                    async for m in self._seek_to(tgt):
                        yield m
                    last = time.monotonic()
                    ctl.utc = self.media
                    continue
                now = time.monotonic()
                if not ctl.paused:
                    self.media += (now - last) * ctl.speed
                last = now
                emitted = 0
                while self.i < n and times[self.i] <= self.media:
                    idx = self.i
                    self.i += 1
                    if idx in init:
                        continue
                    m = tl.get(idx)
                    if m is not None:
                        yield m
                    emitted += 1
                    if emitted % 256 == 0:
                        await asyncio.sleep(0)
                        if ctl.seek_pending:
                            break
                ctl.utc = min(self.media, tl.end)
                if self.i >= n and not ctl.seek_pending:
                    if self.loop or ctl.loop:
                        target = tl.start
                        restart = True
                        continue
                    ctl.utc = tl.end
                    return
                if ctl.paused or self.i >= n:
                    wait = _POLL
                else:
                    wait = min(_POLL, max((times[self.i] - self.media) / ctl.speed, 0.0))
                await asyncio.sleep(wait)


# =============================================================================== FileSource
class FileSource(Source):
    """Replay a fastf1 recording (python-repr lines or JSON lines) honouring original spacing / ``speed``."""

    mode = "file"

    def __init__(self, path: str, speed: float = 1.0, *, loop: bool = False, paused: bool = False) -> None:
        self.path = str(path)
        self.control = Control(speed, paused, replay=True, loop=loop)
        self.control._extra = lambda: {"path": self.path, "messages": len(self._tl) if self._tl is not None else None}
        self._tl: Optional[_Timeline] = None

    async def load(self) -> _Timeline:
        if self._tl is None:
            if not os.path.exists(self.path):
                raise FileNotFoundError("recording not found: %s" % self.path)
            self._tl = await asyncio.to_thread(_FileTimeline.build, self.path)
        return self._tl

    async def messages(self) -> AsyncIterator[Msg]:
        tl = await self.load()
        try:
            async for m in _Player(tl, self.control, self.control.loop).run():
                yield m
        finally:
            tl.close()


# =============================================================================== SimSource
class SimSource(Source):
    """Synthetic race from :mod:`backend.sim` (pre-generated once, then streamed like a recording)."""

    mode = "sim"

    def __init__(self, seed: int = 1, speed: float = 1.0, laps: int = 20, *, track: str = "bahrain-like",
                 loop: bool = False, roundtrip: bool = False, paused: bool = False, **sim_kw: Any) -> None:
        self.seed, self.laps, self.track_name = seed, laps, track
        self.roundtrip = roundtrip
        self._sim_kw = sim_kw
        self.control = Control(speed, paused, replay=True, loop=loop)
        self.control._extra = lambda: {"seed": self.seed, "laps": self.laps, "track": self.track_name}
        self._race: Any = None
        self._tl: Optional[_Timeline] = None

    def _generate(self) -> Any:
        if self._race is None:
            from ..sim import SimConfig, generate

            self._race = generate(SimConfig(seed=self.seed, laps=self.laps, track=self.track_name, **self._sim_kw))
        return self._race

    @property
    def race(self) -> Any:
        """The generated :class:`backend.sim.SimRace` (generated on first access)."""
        return self._generate()

    def all_messages(self) -> List[Msg]:
        """The whole race as ``Msg`` objects, no pacing (used by ``record_to_file``)."""
        return self._generate().messages()

    async def messages(self) -> AsyncIterator[Msg]:
        if self._tl is None:
            race = await asyncio.to_thread(self._generate)
            self._tl = _SimTimeline(race, self.roundtrip)
        async for m in _Player(self._tl, self.control, self.control.loop).run():
            yield m


class _SimTimeline(_Timeline):
    def __init__(self, race: Any, roundtrip: bool) -> None:
        self._race = race
        self._rt = roundtrip
        self.times = race.times
        self.topics = race.topics
        self.initial = []

    def get(self, i: int) -> Optional[Msg]:
        m = self._race.get(i)
        if self._rt:  # exercise the raw-line encode/decode path
            from .messages import to_raw_line

            return parse_line(to_raw_line(m, True))
        return m


# =============================================================================== LiveSource
class LiveSourceError(RuntimeError):
    """The live feed cannot be reached or yields nothing (message is meant for the user)."""


def _default_client_factory(filename: str, filemode: str, timeout: int, no_auth: bool) -> Any:
    try:
        from fastf1.livetiming.client import SignalRClient
    except ImportError as e:  # pragma: no cover - depends on environment
        raise LiveSourceError("LiveSource needs the 'fastf1' package (pip install fastf1): %s" % e) from e
    return SignalRClient(filename, filemode=filemode, timeout=timeout, no_auth=no_auth)


def _explain(exc: BaseException, no_auth: bool) -> str:
    name = type(exc).__name__
    text = "%s: %s" % (name, exc)
    low = text.lower()
    if isinstance(exc, LiveSourceError):
        return str(exc)
    if "auth" in name.lower() or "401" in low or "403" in low or "unauthor" in low or "forbidden" in low:
        hint = "" if no_auth else (" Set F1_USERNAME/F1_PASSWORD (F1 TV subscription) or run `python -m fastf1 auth login`,"
                                   " or start with no_auth=True (--no-auth) for the unauthenticated feed.")
        return "F1 live timing rejected the connection (authentication failed: %s).%s" % (exc, hint)
    if "awsalbcors" in low:
        return ("F1 live timing negotiation failed (no AWSALBCORS cookie in the response). The host is probably "
                "unreachable through a proxy/firewall, or the service rejected the request: %s" % text)
    net = ("connection", "timeout", "timed out", "resolve", "unreachable", "refused", "proxy", "ssl", "name or service",
           "network", "max retries")
    if isinstance(exc, (OSError, ConnectionError, TimeoutError)) or any(w in low for w in net):
        return ("Cannot reach livetiming.formula1.com (network unreachable, blocked or timed out): %s. "
                "Check your internet connection / firewall / proxy settings." % text)
    return "Live timing client failed: %s" % text


class _ClientRunner(threading.Thread):
    """Runs the (blocking) fastf1 SignalR client, restarting it when it exits (timeout / drop)."""

    def __init__(self, factory: Callable[..., Any], path: str, timeout: int, no_auth: bool, retry_delay: float) -> None:
        super().__init__(name="f1-live-client", daemon=True)
        self.factory, self.path, self.timeout, self.no_auth, self.retry_delay = factory, path, timeout, no_auth, retry_delay
        self.stop_event = threading.Event()
        self.client: Any = None
        self.status = "starting"
        self.error: Optional[str] = None
        self.failures = 0  # consecutive runs that produced no data
        self.runs = 0
        self.data_seen = 0.0  # monotonic time of the last data line (set by the tailer)
        self.ever_data = False

    def _size(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def note_data(self) -> None:
        self.data_seen = time.monotonic()
        self.ever_data = True
        self.failures = 0
        self.error = None

    def run(self) -> None:
        while not self.stop_event.is_set():
            started = time.monotonic()
            size_before = self._size()
            self.runs += 1
            try:
                self.status = "connecting"
                self.client = self.factory(self.path, "a", self.timeout, self.no_auth)
                self.status = "connected"
                self.client.start()  # blocks until timeout/stop
                if self.stop_event.is_set():
                    break
                got = self._size() > size_before  # did this run append anything to the recording?
                self.status = "reconnecting"
                if got:
                    self.failures = 0
                else:
                    self.failures += 1
                    self.error = ("No data received from F1 live timing for %ds. Is a session live? If it is, the "
                                  "connection may be unauthenticated (see no_auth / F1 login)." % self.timeout)
            except BaseException as e:  # noqa: BLE001 - includes SystemExit from the client
                if self.stop_event.is_set():
                    break
                self.failures += 1
                self.error = _explain(e, self.no_auth)
                self.status = "error"
                log.warning("live client error: %s", self.error)
                if isinstance(e, LiveSourceError):
                    self.failures = 10 ** 6  # unrecoverable (e.g. fastf1 missing)
                    return
            if time.monotonic() - started < 1.0:
                self.stop_event.wait(self.retry_delay)
            else:
                self.stop_event.wait(min(self.retry_delay, 2.0))

    def stop(self) -> None:
        self.stop_event.set()
        c = self.client
        if c is not None:
            try:  # make fastf1's supervise loop exit on its next 1 s tick
                c.timeout = 1
                c._t_last_message = 0
            except Exception:  # noqa: BLE001
                pass
            try:
                conn = getattr(c, "_connection", None)
                if conn is not None:
                    conn.stop()
            except Exception:  # noqa: BLE001
                pass


class LiveSource(Source):
    """Live F1 timing via ``fastf1.livetiming.client.SignalRClient``.

    The client runs in a daemon thread and appends raw lines to ``recordings/<timestamp>.txt`` (so every live
    session is also a replayable recording); this class tails that file and yields :class:`Msg`.
    ``messages()`` raises :class:`LiveSourceError` (with a human-readable reason) when the feed is unreachable
    or never delivers data after ``max_failures`` attempts; once data has flowed it keeps reconnecting.
    """

    mode = "live"

    def __init__(self, *, no_auth: bool = False, recordings_dir: str = "recordings", filename: Optional[str] = None,
                 timeout: int = 60, max_failures: int = 3, retry_delay: float = 3.0, poll: float = 0.05,
                 client_factory: Optional[Callable[..., Any]] = None) -> None:
        self.no_auth = no_auth
        self.recordings_dir = recordings_dir
        self.filename = filename
        self.timeout = timeout
        self.max_failures = max_failures
        self.retry_delay = retry_delay
        self.poll = poll
        self._factory = client_factory or _default_client_factory
        self.control = Control(1.0, False, replay=False)
        self.control._extra = self._info
        self.runner: Optional[_ClientRunner] = None
        self.path: Optional[str] = None
        self.lines = 0

    def _info(self) -> Dict[str, Any]:
        r = self.runner
        return {"recording": self.path, "status": r.status if r else "idle", "error": r.error if r else None,
                "lines": self.lines, "no_auth": self.no_auth}

    def _new_path(self) -> str:
        if self.filename:
            d = os.path.dirname(self.filename)
            if d:
                os.makedirs(d, exist_ok=True)
            return self.filename
        os.makedirs(self.recordings_dir, exist_ok=True)
        return os.path.join(self.recordings_dir, datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".txt")

    async def messages(self) -> AsyncIterator[Msg]:
        path = self.path = self._new_path()
        runner = self.runner = _ClientRunner(self._factory, path, self.timeout, self.no_auth, self.retry_delay)
        open(path, "a").close()  # the tailer can open it right away
        runner.start()
        buf = b""
        last_utc = time.time()
        try:
            with open(path, "rb") as f:
                while True:
                    chunk = f.readline()
                    if chunk:
                        buf += chunk
                        if not buf.endswith(b"\n"):
                            continue  # partial line, wait for the rest
                        line, buf = buf, b""
                        m = parse_line(line, default_utc=time.time())
                        if m is None:
                            continue
                        self.lines += 1
                        runner.note_data()
                        last_utc = m.utc or last_utc
                        yield m
                        continue
                    if runner.failures >= self.max_failures and not runner.ever_data:
                        raise LiveSourceError(runner.error or "Live timing feed unavailable")
                    if not runner.is_alive() and runner.failures:
                        raise LiveSourceError(runner.error or "Live timing client stopped")
                    await asyncio.sleep(self.poll)
        finally:
            runner.stop()
