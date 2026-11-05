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



# The live source never gives up while the process runs: the F1 feed is simply silent outside of
# sessions, so "no data yet" is a normal state ("waiting"), not an error.  See docs/DEPLOY.md.
class LiveSourceError(RuntimeError):
    """Unrecoverable live-source problem (e.g. fastf1 is not installed); message is meant for the user."""


_LIVE_STATUSES = frozenset({"Started", "Aborted"})  # Aborted = red flag, session still running
_END_STATUSES = frozenset({"Finished", "Finalised", "Ends"})
_FEED_FRESH = 60.0  # s without any feed line before "live" falls back to "waiting"
CONNECT_TIMEOUT = 45.0  # s the fastf1 client may take to open the websocket before it is recycled
KICK_GRACE = 15.0  # s a client thread gets to exit after being asked to stop before it is abandoned


def _format_feed_message(msg: Any) -> Optional[str]:
    """Exactly what ``fastf1.livetiming.client.SignalRClient._on_message`` writes for ``msg`` (3.9.x):
    ``str(list)`` for ``feed`` invocations and, for the ``Subscribe`` CompletionMessage (initial snapshot),
    one ``str([topic, json.dumps(result[topic]), ''])`` per topic joined by newlines."""
    import json

    try:
        from signalrcore.messages.completion_message import CompletionMessage
    except ImportError:  # pragma: no cover - signalrcore ships with fastf1
        CompletionMessage = ()  # type: ignore[assignment]
    if CompletionMessage and isinstance(msg, CompletionMessage):
        data = [[key, json.dumps(msg.result[key]), ""] for key in msg.result.keys()]  # noqa: SIM118
        return "\n".join(map(str, data))
    if isinstance(msg, list):
        return str(msg)
    return None


_RC_CLASS: Any = None


def _recording_client_class(base: Any) -> Any:
    """``base`` (fastf1's SignalRClient) with a UTF-8, newline-safe ``_on_message``.

    The stock client writes through a text file opened with the *locale* encoding (cp1252 on Windows, ASCII
    in some containers): a driver/steward message with an unencodable character is logged and silently lost.
    The override writes UTF-8 bytes straight to the file's binary layer, flushes every message and records
    a rejected ``Subscribe`` (``feed_error``) so the runner can show it.  Formatting is identical to the stock
    client (``_format_feed_message``; covered by a unit test comparing both)."""
    global _RC_CLASS
    if _RC_CLASS is not None and issubclass(_RC_CLASS, base):
        return _RC_CLASS

    class RecordingSignalRClient(base):  # type: ignore[misc, valid-type]
        feed_error: Optional[str] = None
        lines_written = 0

        def _on_message(self, msg):  # noqa: D401
            self._t_last_message = time.time()
            err = getattr(msg, "error", None)
            if err:
                self.feed_error = "F1 live timing rejected the subscription: %s" % (err,)
                self.logger.error(self.feed_error)
                return
            try:
                formatted = _format_feed_message(msg)
            except Exception:  # noqa: BLE001 - e.g. a CompletionMessage without a result
                self.logger.exception("Unexpected feed message %r", type(msg))
                return
            if formatted is None:
                self.logger.error("Unknown message type: %s", type(msg))
                return
            try:
                f = self._output_file
                data = (formatted + "\n").encode("utf-8", "replace")
                buf = getattr(f, "buffer", None)
                if buf is not None:
                    buf.write(data)
                    buf.flush()
                else:  # pragma: no cover - always a text file in fastf1 3.x
                    f.write(formatted + "\n")
                    f.flush()
                self.lines_written += formatted.count("\n") + 1
            except Exception:  # noqa: BLE001
                self.logger.exception("Exception while writing message to file")

    _RC_CLASS = RecordingSignalRClient
    return _RC_CLASS


def _default_client_factory(filename: str, filemode: str, timeout: int, no_auth: bool) -> Any:
    try:
        from fastf1.livetiming.client import SignalRClient
    except ImportError as e:  # pragma: no cover - depends on environment
        raise LiveSourceError("LiveSource needs the 'fastf1' package (pip install fastf1): %s" % e) from e
    cls = _recording_client_class(SignalRClient)
    # own logger: the stock client calls logging.basicConfig() when none is given
    return cls(filename, filemode=filemode, timeout=timeout, no_auth=no_auth, logger=logging.getLogger("backend.signalr"))


def _f1_account() -> Any:
    """The fastf1 module that owns the F1 token (name differs between fastf1 releases)."""
    import importlib

    for name in ("fastf1.internals.f1account", "fastf1.internals.f1auth"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


def _default_auth_check() -> Optional[str]:
    """Make sure a valid F1 TV token exists *without ever prompting* (a prompt inside the server's worker
    thread would hang it).  Returns ``None`` when a token is available, else a user-facing problem text.

    Sign-in itself is delegated to fastf1 (``get_token``) which stores the token on disk, so the
    client started afterwards reuses it without signing in again."""
    acct = _f1_account()
    if acct is None or not hasattr(acct, "get_token"):
        return None  # unknown fastf1 layout: let the client authenticate by itself
    try:
        tm = getattr(acct, "token_manager", None)
        tok = tm.get_stored() if tm is not None else None
        if tok is not None and tok.is_valid():
            return None
        method = acct.get_auth_method() if hasattr(acct, "get_auth_method") else "password"
        if method == "password" and not (os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD")):
            return ("No valid F1 TV token and F1_USERNAME / F1_PASSWORD are not set. Set both, or run "
                    "`python -m fastf1 auth f1tv --authenticate` once (the server picks the token up automatically), "
                    "or restart with --no-auth.")
        if method != "password":
            return ("No valid F1 TV token (sign-in method %r). Run `python -m fastf1 auth f1tv --authenticate`, "
                    "or restart with --no-auth." % method)
        acct.get_token()  # signs in with F1_USERNAME / F1_PASSWORD and stores the token
        return None
    except Exception as e:  # noqa: BLE001 - F1AuthError, requests errors, ...
        return "F1 sign-in failed (%s: %s). Check F1_USERNAME / F1_PASSWORD and your F1 TV subscription." % (type(e).__name__, e)


def _explain(exc: BaseException, no_auth: bool) -> str:
    name = type(exc).__name__
    text = "%s: %s" % (name, exc)
    low = text.lower()
    if isinstance(exc, LiveSourceError):
        return str(exc)
    if "auth" in name.lower() or "401" in low or "403" in low or "unauthor" in low or "forbidden" in low:
        hint = "" if no_auth else (" Set F1_USERNAME/F1_PASSWORD (F1 TV subscription) or run `python -m fastf1 auth f1tv --authenticate`,"
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


def _ensure_trailing_newline(path: str) -> None:
    """A previous client killed mid-write may leave a partial last line; terminate it so the next client's
    first message does not get glued onto it (the garbage line is skipped by the parser)."""
    try:
        with open(path, "rb+") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                return
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
    except OSError:
        pass


class _ClientRunner(threading.Thread):
    """Keeps a fastf1 SignalR client alive for the whole life of the process.

    One *run* = auth pre-flight, build a client, call its blocking ``start()`` in a helper thread, and
    supervise it from here.  The stock client has three failure modes this guards against:

    * ``start()`` can block forever before its own supervisor starts (it spins in
      ``while not self._is_connected`` when the websocket never opens)  -> ``connect_timeout``;
    * after a drop it only notices once its ``timeout`` of silence has passed        -> ``_is_connected`` is polled;
    * ``timeout=0`` (we disable the client's own exit) makes it wait forever on a half-open socket
      -> we recycle when the recording file has not grown for ``idle_timeout`` (a live feed always grows it:
      Heartbeat, CarData, Position) or ``waiting_timeout`` (nothing but the initial snapshot: no session running).

    Between runs it waits with exponential back-off (``retry_delay`` .. ``max_delay``) that is reset by data.
    It never stops by itself; only :meth:`stop` ends it.
    """

    def __init__(self, factory: Callable[..., Any], path: str, *, no_auth: bool, retry_delay: float,
                 idle_timeout: float, waiting_timeout: float, connect_timeout: float, max_delay: float,
                 auth_check: Optional[Callable[[], Optional[str]]], watch_interval: float, drop_grace: float = 3.0) -> None:
        super().__init__(name="f1-live-client", daemon=True)
        self.factory, self.path, self.no_auth, self.retry_delay = factory, path, no_auth, retry_delay
        self.idle_timeout, self.waiting_timeout, self.connect_timeout = idle_timeout, waiting_timeout, connect_timeout
        self.max_delay, self.auth_check, self.watch_interval = max_delay, auth_check, watch_interval
        self.drop_grace = drop_grace
        self.stop_event = threading.Event()
        self.client: Any = None
        self.status = "starting"
        self.error: Optional[str] = None
        self.note: Optional[str] = None
        self.fatal = False
        self.auth_state = "skipped" if (no_auth or auth_check is None) else "pending"
        self.connected = False
        self.runs = 0
        self.failures = 0  # consecutive failed runs (exception / nothing received)
        self.data_seen = 0.0  # monotonic time of the last feed (non-snapshot) line, set by the tailer
        self.line_seen = 0.0  # ... of the last line of any kind
        self.ever_data = False
        self.live_run = False  # this run received feed lines after the initial snapshot
        self.connected_since: Optional[float] = None

    # --- tailer callbacks (event-loop thread) ---------------------------------------------------------------
    def note_line(self, snapshot: bool) -> None:
        now = time.monotonic()
        self.line_seen = now
        self.error = None
        if not snapshot:
            self.data_seen = now
            self.live_run = True
            self.ever_data = True

    def _size(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    # --- client control ---------------------------------------------------------------------------------------
    @staticmethod
    def _client_connected(client: Any, age: float) -> bool:
        v = getattr(client, "_is_connected", None)
        return v if isinstance(v, bool) else age > 0.3  # fakes have no flag: treat as up shortly after start()

    def _kick(self, client: Any, release_connect_wait: bool = True) -> None:
        """Make ``client.start()`` return: lower its timeout so its supervise loop exits on the next 1 s tick,
        close the connection and release the connect-wait spin."""
        if client is None:
            return
        try:
            client.timeout = 1
            client._t_last_message = 0
        except Exception:  # noqa: BLE001
            pass
        try:
            conn = getattr(client, "_connection", None)
            if conn is not None:
                conn.stop()
        except Exception:  # noqa: BLE001
            pass
        if release_connect_wait:
            try:
                if getattr(client, "_is_connected", True) is False:
                    client._is_connected = True  # lets `while not self._is_connected` in _run() fall through
            except Exception:  # noqa: BLE001
                pass

    def stop(self) -> None:
        self.stop_event.set()
        self._kick(self.client)

    @staticmethod
    def _close_client_file(client: Any) -> None:
        try:
            f = getattr(client, "_output_file", None)
            if f is not None and not f.closed:
                f.close()
        except Exception:  # noqa: BLE001
            pass

    # --- main loop --------------------------------------------------------------------------------------------
    def run(self) -> None:
        while not self.stop_event.is_set():
            self.runs += 1
            try:
                outcome = self._one_run()
            except LiveSourceError as e:
                self.error, self.status, self.fatal = str(e), "error", True
                log.error("live source cannot run: %s", e)
                return
            except BaseException as e:  # noqa: BLE001 - defensive: the supervisor itself must never die
                log.exception("live runner crashed; restarting")
                self.error, self.status, outcome = _explain(e, self.no_auth), "error", "error"
            if self.stop_event.is_set():
                break
            if outcome == "ok":
                self.failures = 0
                delay = self.retry_delay
                self.status = "reconnecting"
            else:
                self.failures += 1
                delay = min(self.max_delay, self.retry_delay * (2 ** min(self.failures - 1, 6)))
                if outcome == "auth":
                    delay = max(delay, min(30.0, self.max_delay))  # do not hammer the F1 sign-in
                self.status = "error" if self.error else "reconnecting"
            log.info("live client run %d ended (%s); next attempt in %.1fs", self.runs, outcome, delay)
            self.stop_event.wait(delay)

    def _one_run(self) -> str:
        """Returns 'ok' (data flowed), 'quiet' (connected, nothing received), 'error' or 'auth'."""
        self.live_run = False
        self.connected = False
        self.connected_since = None
        # 1. auth pre-flight (never prompts; see _default_auth_check)
        if self.auth_check is not None and not self.no_auth:
            self.status = "authenticating"
            try:
                problem = self.auth_check()
            except Exception as e:  # noqa: BLE001
                problem = "F1 sign-in check failed: %s: %s" % (type(e).__name__, e)
            if problem:
                self.auth_state = "failed"
                self.error = problem
                self.status = "error"
                log.warning("live auth: %s", problem)
                return "auth"
            self.auth_state = "ok"
        # 2. build the client
        _ensure_trailing_newline(self.path)
        self.status = "connecting"
        try:
            client = self.factory(self.path, "a", 0, self.no_auth)  # timeout=0: we own the idle policy
        except LiveSourceError:
            raise
        except BaseException as e:  # noqa: BLE001
            self.error = _explain(e, self.no_auth)
            self.status = "error"
            log.warning("live client could not be created: %s", self.error)
            return "error"
        self.client = client
        exc_box: List[BaseException] = []

        def call_start() -> None:
            try:
                client.start()  # blocks until the supervisor exits (we make it exit) or an exception
            except BaseException as e:  # noqa: BLE001 - includes SystemExit
                exc_box.append(e)

        th = threading.Thread(target=call_start, name="f1-live-client-start", daemon=True)
        started = time.monotonic()
        size0 = last_size = self._size()
        activity = started
        was_conn = False
        disc_since: Optional[float] = None
        kicked_at: Optional[float] = None
        kick_reason: Optional[str] = None
        th.start()
        # 3. supervise
        while th.is_alive():
            th.join(self.watch_interval)
            now = time.monotonic()
            sz = self._size()
            if sz > last_size:
                activity = now
            last_size = sz
            if kicked_at is not None:
                if now - kicked_at > KICK_GRACE:
                    log.warning("live client thread did not exit after %.0fs; abandoning it", KICK_GRACE)
                    break
                continue
            conn = self._client_connected(client, now - started)
            self.connected = conn
            if conn:
                was_conn, disc_since = True, None
                if self.connected_since is None:
                    self.connected_since = now
                if self.error and now - self.connected_since > 5.0:
                    self.error = None  # connected steadily again: an old connect error is stale
                if self.status in ("connecting", "reconnecting", "authenticating", "error"):
                    self.status = "connected"
            else:
                self.connected_since = None
                if was_conn and disc_since is None:
                    disc_since = now
            fe = getattr(client, "feed_error", None)
            reason = None
            if self.stop_event.is_set():
                reason = "stop"
            elif fe:
                reason, self.error = "subscription rejected", fe
            elif not conn and not was_conn and now - started > self.connect_timeout:
                reason = "connect timeout"
                self.error = ("Could not open the F1 live timing websocket within %ds (blocked network, bad token or "
                              "service problem); retrying." % self.connect_timeout)
            elif was_conn and disc_since is not None and now - disc_since > self.drop_grace:
                reason = "connection closed"
                self.note = "connection to F1 live timing was closed; reconnecting"
            elif conn:
                limit = self.idle_timeout if self.live_run else self.waiting_timeout
                if now - activity > limit:
                    reason = "silent for %ds" % (now - activity)
                    self.note = ("feed silent for %ds%s; reconnecting" %
                                 (now - activity, "" if self.live_run else " (no session running?)"))
            if reason:
                log.info("recycling live client: %s", reason)
                self._kick(client, release_connect_wait=not conn)
                kicked_at, kick_reason = now, reason
        self._close_client_file(client)
        self.connected = False
        if exc_box and not self.stop_event.is_set():
            self.error = _explain(exc_box[0], self.no_auth)
            self.status = "error"
            log.warning("live client error: %s", self.error)
            return "error"
        grew = self._size() > size0
        if grew:
            return "ok"
        if kick_reason == "subscription rejected":
            return "error"
        if kicked_at is not None and kick_reason == "connect timeout":
            return "error"
        self.note = "connected but the feed sent nothing (normal when no session is running)"
        return "quiet"


class _Tail:
    """Incremental line reader for a file another thread/process appends to: binary chunked reads, only complete
    ``\\n``-terminated lines are returned, the partial last line is kept until its remainder arrives."""

    CHUNK = 1 << 20
    MAX_PARTIAL = 64 << 20  # a "line" longer than this is garbage: drop it rather than grow without bound

    def __init__(self, path: str) -> None:
        self.path = path
        self.fh = open(path, "rb")
        self.buf = b""
        self.pos = 0

    def read(self) -> List[bytes]:
        data = self.fh.read(self.CHUNK)
        if not data:
            try:
                if os.path.getsize(self.path) < self.pos:  # truncated / replaced: start over
                    self.fh.seek(0)
                    self.pos, self.buf = 0, b""
            except OSError:
                pass
            return []
        self.pos += len(data)
        self.buf += data
        parts = self.buf.split(b"\n")
        self.buf = parts.pop()
        if len(self.buf) > self.MAX_PARTIAL:
            self.buf = b""
        return parts

    def close(self) -> None:
        try:
            self.fh.close()
        except OSError:
            pass


class LiveSource(Source):
    """Live F1 timing via ``fastf1.livetiming.client.SignalRClient``.

    The client runs in a supervised daemon thread (see :class:`_ClientRunner`) and appends raw lines to
    ``<recordings_dir>/<timestamp>.txt`` (so every live session is also a replayable recording); this class tails
    that file and yields :class:`Msg`.

    ``messages()`` **never ends or raises because of the network or the feed**: before a session starts the feed
    is silent, between sessions it goes quiet, connections drop, tokens expire - the runner keeps (re)connecting with
    back-off for as long as the process lives and ``control.info()`` reports what is going on (``phase`` =
    ``waiting|live|ended|error``, ``message``, ``error``).  Only a missing fastf1 install is fatal (``error`` is set,
    ``messages()`` then idles).
    """

    mode = "live"

    def __init__(self, *, no_auth: bool = False, recordings_dir: str = "recordings", filename: Optional[str] = None,
                 timeout: int = 60, waiting_timeout: float = 300.0, connect_timeout: float = CONNECT_TIMEOUT,
                 max_failures: int = 0, retry_delay: float = 3.0, max_delay: float = 60.0, poll: float = 0.05,
                 watch_interval: float = 0.5, drop_grace: float = 3.0, client_factory: Optional[Callable[..., Any]] = None,
                 auth_check: Optional[Callable[[], Optional[str]]] = None) -> None:
        """``timeout``: seconds of total silence (not even Heartbeat) after which a feed that was live is recycled;
        ``waiting_timeout``: same when only the initial snapshot arrived (no session running); ``max_failures`` is
        accepted for backwards compatibility and ignored (the source never gives up)."""
        self.no_auth = no_auth
        self.recordings_dir = recordings_dir
        self.filename = filename
        self.idle_timeout = timeout
        self.waiting_timeout = waiting_timeout
        self.connect_timeout = connect_timeout
        self.retry_delay = retry_delay
        self.max_delay = max_delay
        self.poll = poll
        self.watch_interval = watch_interval
        self.drop_grace = drop_grace
        self._factory = client_factory or _default_client_factory
        self._auth_check = auth_check if auth_check is not None else (_default_auth_check if client_factory is None else None)
        self.control = Control(1.0, False, replay=False)
        self.control._extra = self._info
        self.runner: Optional[_ClientRunner] = None
        self.path: Optional[str] = None
        self.lines = 0
        self.bad_lines = 0
        self.started_at = time.monotonic()
        self.session_status: Optional[str] = None
        self.session: Dict[str, Any] = {}
        self.topics: Dict[str, int] = {}
        self._disk: Tuple[float, Optional[int]] = (0.0, None)

    # --- status ---------------------------------------------------------------------------------------------------
    def _size(self) -> Optional[int]:
        try:
            return os.path.getsize(self.path) if self.path else None
        except OSError:
            return None

    def _disk_free_mb(self) -> Optional[int]:
        now = time.monotonic()
        if now - self._disk[0] > 10.0:
            try:
                import shutil

                free = int(shutil.disk_usage(os.path.dirname(os.path.abspath(self.path or self.recordings_dir))).free // (1 << 20))
            except OSError:
                free = None
            self._disk = (now, free)
        return self._disk[1]

    def phase(self) -> str:
        """``waiting`` (connected / connecting, session not started), ``live``, ``ended`` or ``error``."""
        r = self.runner
        if r is None:
            return "waiting"
        if r.fatal or (r.error and not (r.connected and r.live_run)):
            return "error"
        now = time.monotonic()
        fresh = r.line_seen and now - r.line_seen < _FEED_FRESH
        s = self.session_status
        if s in _END_STATUSES:
            return "ended"
        if s in _LIVE_STATUSES and fresh:
            return "live"
        if s is None and r.live_run and r.data_seen and now - r.data_seen < 15.0 and self.topics.get("Position", 0) + self.topics.get("TimingData", 0) > 0:
            return "live"  # no SessionStatus seen (very unusual) but timing data is flowing
        return "waiting"

    def _message(self, phase: str) -> str:
        r = self.runner
        if r is None:
            return "Live source not started"
        if phase == "error":
            return r.error or "Live timing error"
        sess = ""
        name, meeting = self.session.get("name"), self.session.get("meeting")
        if name or meeting:
            sess = " (feed shows: %s)" % ", ".join(x for x in (meeting, name) if x)
        if phase == "live":
            return "Live"
        if phase == "ended":
            return "Session finished (%s). Leave this running if another session follows." % self.session_status
        if r.status in ("starting", "authenticating", "connecting"):
            return "Connecting to F1 live timing..."
        if r.status == "reconnecting" or not r.connected:
            return "Connection to F1 live timing lost - reconnecting..." if r.ever_data or self.lines else "Connecting to F1 live timing..."
        return "Connected - waiting for the session to start%s" % sess

    def _info(self) -> Dict[str, Any]:
        r = self.runner
        now = time.monotonic()
        ph = self.phase()
        age = lambda t: None if not t else round(now - t, 1)  # noqa: E731
        return {
            "recording": self.path, "recordingBytes": self._size(), "diskFreeMB": self._disk_free_mb(),
            "status": r.status if r else "idle", "error": (r.error if r else None), "note": (r.note if r else None),
            "phase": ph, "waiting": ph == "waiting", "message": self._message(ph),
            "connected": bool(r and r.connected), "runs": r.runs if r else 0, "reconnects": max(0, r.runs - 1) if r else 0,
            "authState": r.auth_state if r else None, "lines": self.lines, "badLines": self.bad_lines, "no_auth": self.no_auth,
            "sessionStatus": self.session_status, "session": dict(self.session),
            "lastFeedAgeSec": age(r.line_seen) if r else None, "lastDataAgeSec": age(r.data_seen) if r else None,
            "sourceUptimeSec": round(now - self.started_at, 1),
        }

    def close(self) -> None:
        r = self.runner
        if r is not None:
            r.stop()
            r.join(timeout=3.0)

    # --- file ---------------------------------------------------------------------------------------------------
    def _new_path(self) -> str:
        if self.filename:
            d = os.path.dirname(self.filename)
            if d:
                os.makedirs(d, exist_ok=True)
            return self.filename
        os.makedirs(self.recordings_dir, exist_ok=True)
        base = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.recordings_dir, base + ".txt")
        n = 1
        while os.path.exists(path):  # never append to (or interleave with) an older recording
            n += 1
            path = os.path.join(self.recordings_dir, "%s_%d.txt" % (base, n))
        return path

    def _track(self, m: Msg) -> None:
        t = m.topic
        self.topics[t] = self.topics.get(t, 0) + 1
        d = m.data
        if not isinstance(d, dict):
            return
        if t == "SessionStatus" and d.get("Status"):
            self.session_status = str(d["Status"])
        elif t == "SessionInfo":
            meet = d.get("Meeting")
            if isinstance(meet, dict) and meet.get("Name"):
                self.session["meeting"] = str(meet["Name"])
            if d.get("Name"):
                self.session["name"] = str(d["Name"])
            if d.get("Type"):
                self.session["type"] = str(d["Type"])

    # --- stream -------------------------------------------------------------------------------------------------
    async def messages(self) -> AsyncIterator[Msg]:
        if self.runner is not None:  # re-entered (server retry): replace the old runner, start a fresh recording
            self.close()
        path = self.path = self._new_path()
        self.session_status = None
        runner = self.runner = _ClientRunner(
            self._factory, path, no_auth=self.no_auth, retry_delay=self.retry_delay, idle_timeout=self.idle_timeout,
            waiting_timeout=self.waiting_timeout, connect_timeout=self.connect_timeout, max_delay=self.max_delay,
            auth_check=self._auth_check, watch_interval=self.watch_interval, drop_grace=self.drop_grace)
        open(path, "ab").close()  # the tailer can open it right away
        tail = _Tail(path)
        runner.start()
        n = 0
        try:
            while True:
                lines = tail.read()
                if not lines:
                    await asyncio.sleep(self.poll)
                    continue
                for raw in lines:
                    if not raw.strip():
                        continue
                    m = parse_line(raw, default_utc=time.time())
                    if m is None:
                        self.bad_lines += 1
                        if self.bad_lines <= 5:
                            log.warning("unparseable line in %s (%d bytes): %r", path, len(raw), raw[:80])
                        continue
                    self.lines += 1
                    self._track(m)
                    runner.note_line(snapshot=raw.rstrip().endswith(b"'']"))
                    yield m
                    n += 1
                    if n % 200 == 0:
                        await asyncio.sleep(0)  # a reconnect snapshot / backlog must not starve the event loop
        finally:
            tail.close()
            runner.stop()
