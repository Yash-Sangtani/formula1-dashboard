"""FastAPI server: pumps a Source into the RaceEngine and broadcasts over WebSocket per docs/CONTRACT.md."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .engine import RaceEngine
from .track import OfficialLayoutFallback

log = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
FAST_HZ = 4.0
STATE_MIN_INTERVAL = 1.0
HISTORY_MIN_INTERVAL = 2.0
MAX_CLIENTS = 100            # concurrent WebSocket viewers; extra ones are told to retry later (close code 1013)
SEND_TIMEOUT = 15.0          # a client that cannot take one frame in this long is dropped
MAX_CLIENT_MSG = 4096        # control messages from browsers are tiny; ignore anything bigger
PHASES = ("waiting", "live", "ended", "error")


def _default(o: Any) -> Any:
    """json ``default=``: numpy scalars/arrays, sets, anything else as text (never raises)."""
    item = getattr(o, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    tolist = getattr(o, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:  # noqa: BLE001
            pass
    return str(o)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False, default=_default)


def _clean(o: Any) -> Any:
    """Replace NaN/inf by None, tuples by lists, non-string keys by strings; numpy values via ``_default``."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else str(k)): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if o is None or isinstance(o, (str, int, bool)):
        return o
    d = _default(o)
    return _clean(d) if d is not o and not isinstance(d, str) else d


def _safe_dumps(obj: Any) -> str:
    """Compact JSON that can never fail on NaN/inf/odd keys (those become null); last resort is an error frame."""
    try:
        return _dumps(obj)
    except (ValueError, TypeError):
        pass
    try:
        return _dumps(_clean(obj))
    except Exception:  # noqa: BLE001 - circular reference etc.
        log.exception("could not serialise frame")
        return _dumps({"type": "error", "message": "frame could not be serialised"})


class NoCacheStatic(StaticFiles):
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        return resp


class Client:
    """Per-client outgoing buffer. Latest-wins slots per frame type; fast frames are merged, so a slow
    client skips stale fast frames instead of building a backlog."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.event = asyncio.Event()
        self.slots: Dict[str, str] = {}          # type -> text (latest wins)
        self.order: deque = deque()              # types in arrival order
        self.fast_cars: Optional[Dict[str, dict]] = None
        self.fast_utc = 0.0
        self.fast_text: Optional[str] = None
        self.dropped_fast = 0
        self.closed = False

    def put(self, kind: str, text: str) -> None:
        if kind not in self.slots:
            self.order.append(kind)
        self.slots[kind] = text
        self.event.set()

    def put_fast(self, frame: dict, text: str) -> None:
        if self.fast_cars is None:
            self.fast_cars = {k: dict(v) for k, v in frame["cars"].items()}
            self.fast_text = text
        else:  # backlog: merge, latest wins per field
            self.dropped_fast += 1
            for k, v in frame["cars"].items():
                self.fast_cars.setdefault(k, {}).update(v)
            self.fast_text = None
        self.fast_utc = frame["utc"]
        self.event.set()

    def take(self):
        out = []
        while self.order:
            k = self.order.popleft()
            out.append(self.slots.pop(k))
        if self.fast_cars is not None:
            txt = self.fast_text or _safe_dumps({"type": "fast", "utc": self.fast_utc, "cars": self.fast_cars})
            out.append(txt)
            self.fast_cars = None
            self.fast_text = None
        self.event.clear()
        return out

    async def writer(self):
        try:
            while not self.closed:
                await self.event.wait()
                for text in self.take():
                    await asyncio.wait_for(self.ws.send_text(text), SEND_TIMEOUT)
        except asyncio.CancelledError:
            raise
        except Exception:  # peer gone or too slow: drop this client, never the server
            pass
        finally:
            self.closed = True
            try:  # unblocks the endpoint's receive loop so the client is cleaned up
                await asyncio.wait_for(self.ws.close(), 2.0)
            except Exception:  # noqa: BLE001
                pass


_NO_CIRCUIT = (None, "", 0, "0")


def session_context(eng: RaceEngine) -> Tuple[Any, Optional[int], Any]:
    """(circuit key, season year, meeting key) from the engine's raw ``SessionInfo`` (any may be None)."""
    try:
        si = (eng.raw or {}).get("SessionInfo")
    except Exception:  # noqa: BLE001
        return None, None, None
    if not isinstance(si, dict):
        return None, None, None
    meeting = si.get("Meeting") if isinstance(si.get("Meeting"), dict) else {}
    circuit = meeting.get("Circuit") if isinstance(meeting.get("Circuit"), dict) else {}
    ckey = circuit.get("Key")
    year: Optional[int] = None
    m = re.match(r"\s*(\d{4})/", str(si.get("Path") or ""))
    if m:
        year = int(m.group(1))
    else:
        for k in ("StartDate", "EndDate"):
            m = re.match(r"\s*(\d{4})-", str(si.get(k) or ""))
            if m:
                year = int(m.group(1))
                break
    return (None if ckey in _NO_CIRCUIT else ckey), year, meeting.get("Key")


class ServerState:
    def __init__(self, source, engine: RaceEngine, mode: str, retry_live: bool, max_clients: int = MAX_CLIENTS,
                 fallback: Optional[OfficialLayoutFallback] = None):
        self.source = source
        self.engine = engine
        self.mode = mode
        self.retry_live = retry_live
        self.max_clients = max_clients
        self.fallback = fallback
        self.clients: set[Client] = set()
        self.error: Optional[str] = None
        self.ended = False
        self.started_at = time.time()
        self.last_msg_mono: Optional[float] = None
        self.msg_times: deque = deque(maxlen=4096)
        self.total_msgs = 0
        self.apply_errors = 0
        self.pump_restarts = 0
        self.rejected_clients = 0
        self.epoch = 0            # bumps on seek / reset to force full resend
        self.tasks: list = []
        self.lock = asyncio.Lock()
        self.seeking = False
        self.restart = asyncio.Event()
        self.circuit: Tuple[Any, Optional[int]] = (None, None)
        self.last_fallback_tick = 0.0
        self.viewers_have_track = False   # some client was sent a (non-null) track; a reset must tell them to drop it

    @property
    def replay(self) -> bool:
        return self.mode != "live"

    def msgs_per_sec(self) -> float:
        now = time.monotonic()
        while self.msg_times and now - self.msg_times[0] > 5.0:
            self.msg_times.popleft()
        return round(len(self.msg_times) / 5.0, 1)

    def control_info(self) -> dict:
        c = getattr(self.source, "control", None)
        info: Dict[str, Any] = {}
        if c is not None:
            try:
                i = c.info() if callable(getattr(c, "info", None)) else {}
                if isinstance(i, dict):
                    info.update(i)
            except Exception:
                log.exception("control.info failed")
            for k in ("speed", "paused"):
                if hasattr(c, k):
                    info[k] = getattr(c, k)
        return info

    def current_error(self, info: Optional[dict] = None) -> Optional[str]:
        info = self.control_info() if info is None else info
        return self.error or info.get("error")

    def phase(self, info: Optional[dict] = None) -> str:
        """``waiting`` (connected / starting, nothing to show yet), ``live``, ``ended`` or ``error``."""
        info = self.control_info() if info is None else info
        p = info.get("phase")
        if p in PHASES:
            return p
        if self.error:
            return "error"
        if self.ended:
            return "ended"
        return "waiting" if self.total_msgs == 0 else "live"

    def status_text(self, info: Optional[dict] = None) -> str:
        info = self.control_info() if info is None else info
        ph = self.phase(info)
        if info.get("message"):
            return str(info["message"])
        if ph == "error":
            return str(self.current_error(info) or "error")
        return {"waiting": "Waiting for data...", "live": "Live" if self.mode == "live" else "Playing",
                "ended": "Finished"}[ph]

    def status_frame(self, info: Optional[dict] = None) -> dict:
        info = self.control_info() if info is None else info
        ph = self.phase(info)
        return {"type": "status", "phase": ph, "waiting": ph == "waiting", "message": self.status_text(info),
                "error": self.current_error(info), "mode": self.mode}

    def hello(self) -> dict:
        info = self.control_info()
        ph = self.phase(info)
        return {"type": "hello", "mode": self.mode, "replay": self.replay, "info": info, "phase": ph,
                "waiting": ph == "waiting", "message": self.status_text(info), "error": self.current_error(info)}

    def broadcast(self, kind: str, text: str) -> None:
        for c in list(self.clients):
            c.put(kind, text)

    def broadcast_fast(self, frame: dict) -> None:
        if not self.clients:
            return
        text = _safe_dumps(frame)
        for c in list(self.clients):
            c.put_fast(frame, text)


def _infer_mode(source) -> str:
    m = getattr(source, "mode", None)
    if m in ("live", "file", "sim"):
        return m
    n = type(source).__name__.lower()
    if "live" in n:
        return "live"
    if "sim" in n:
        return "sim"
    return "file"


def reset_track(st: ServerState, reason: str = "api") -> None:
    """Forget the track (and every car's projection onto it) so a new one is learned / fetched.  Safe to call any time."""
    eng = st.engine
    eng.tb.reset()
    for d in getattr(eng, "drivers", {}).values():
        for attr in ("progress", "hint"):
            if hasattr(d, attr):
                try:
                    setattr(d, attr, None)
                except Exception:  # noqa: BLE001
                    pass
    if st.fallback is not None:
        st.fallback.forget()
    eng.rev += 1
    log.info("track reset (%s)", reason)


def track_tick(st: ServerState) -> None:
    """Called ~4 Hz: keep the TrackBuilder in step with the circuit announced by SessionInfo and, while no
    track exists, let the official-layout fallback try to supply one.  Never raises."""
    try:
        eng = st.engine
        ckey, year, _ = session_context(eng)
        if ckey is None:
            return
        tb = eng.tb
        if tb.set_context(ckey, year):  # a different circuit than the samples/track belong to
            reset_track_side_effects(st)
        st.circuit = (ckey, year)
        now = time.monotonic()
        if st.fallback is not None and tb.track is None and year is not None and now - st.last_fallback_tick >= 1.5:
            st.last_fallback_tick = now
            res = st.fallback.tick(tb, ckey, year)
            if res == "accepted":
                project = getattr(eng, "_project", None)
                for d in getattr(eng, "drivers", {}).values():
                    if callable(project):
                        try:
                            project(d)
                        except Exception:  # noqa: BLE001
                            pass
                eng.rev += 1
    except Exception:  # noqa: BLE001
        log.exception("track tick failed")


def reset_track_side_effects(st: ServerState) -> None:
    eng = st.engine
    for d in getattr(eng, "drivers", {}).values():
        for attr in ("progress", "hint"):
            if hasattr(d, attr):
                try:
                    setattr(d, attr, None)
                except Exception:  # noqa: BLE001
                    pass
    eng.rev += 1


async def _pump(st: ServerState):
    """Source -> engine.  Live: never gives up (re-opens the source with back-off if it ever ends or raises); replay:
    stays idle (serving the last state) when the stream ends.  A bad message never stops the stream."""
    backoff = 2.0
    while True:
        try:
            n = 0
            async for msg in st.source.messages():
                try:
                    st.engine.apply(msg)
                except Exception:  # noqa: BLE001 - the engine is supposed not to raise; belt and braces
                    st.apply_errors += 1
                    if st.apply_errors <= 5:
                        log.exception("engine.apply raised")
                st.total_msgs += 1
                st.msg_times.append(time.monotonic())
                st.last_msg_mono = time.monotonic()
                st.error = None
                backoff = 2.0
                n += 1
                if n % 200 == 0:
                    await asyncio.sleep(0)
            st.ended = True
            log.info("source ended after %d messages", st.total_msgs)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            st.error = "%s: %s" % (type(e).__name__, e)
            log.exception("source error")
        if st.mode == "live" and st.retry_live:
            st.ended = False
            await asyncio.sleep(backoff)
            backoff = min(30.0, backoff * 1.7)
            continue
        # replay finished (or failed): keep serving the last state; a seek restarts the stream
        await st.restart.wait()
        st.restart.clear()
        st.ended = False


async def _broadcaster(st: ServerState):
    eng = st.engine
    last_rev = -1
    last_hist_rev = -1
    last_track_sig: Any = None
    last_state_t = 0.0
    last_hist_t = 0.0
    last_status_sig: Any = None
    tick = 0
    epoch = st.epoch
    period = 1.0 / FAST_HZ
    while True:
        await asyncio.sleep(period)
        tick += 1
        try:
            now = time.monotonic()
            force = epoch != st.epoch
            epoch = st.epoch
            if force:
                last_rev = last_hist_rev = -1
                last_track_sig = None
                last_state_t = last_hist_t = 0.0
                last_status_sig = None
            track_tick(st)
            has_track = eng.tb.track is not None
            sig = (id(eng.tb), eng.track_version, has_track)
            if sig != last_track_sig:  # tracked even with no viewers: a later reset must still reach those who joined after the build
                if st.clients:
                    if has_track:
                        st.broadcast("track", _safe_dumps({"type": "track", "track": eng.track_dict()}))
                        st.viewers_have_track = True
                    elif st.viewers_have_track:  # reset: tell viewers to drop the old outline
                        st.broadcast("track", _safe_dumps({"type": "track", "track": None}))
                        st.viewers_have_track = False
                last_track_sig = sig
            if not st.clients:
                eng.fast_delta()  # keep the dirty set from growing
                continue
            if tick % 4 == 0 or force:
                info = st.control_info()
                sig = (st.phase(info), st.status_text(info), st.current_error(info))
                if sig != last_status_sig:
                    last_status_sig = sig
                    st.broadcast("status", _safe_dumps(st.status_frame(info)))
            f = eng.fast_delta()
            if f is not None:
                st.broadcast_fast(f)
            if eng.rev != last_rev and now - last_state_t >= STATE_MIN_INTERVAL:
                last_rev, last_state_t = eng.rev, now
                st.broadcast("state", _safe_dumps(eng.state()))
            if eng.history_rev != last_hist_rev and now - last_hist_t >= HISTORY_MIN_INTERVAL:
                last_hist_rev, last_hist_t = eng.history_rev, now
                st.broadcast("history", _safe_dumps(eng.history()))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("broadcaster error")


async def _supervised(name: str, st: ServerState, fn):
    """Run ``fn(st)`` forever: if it ever dies with an exception, log and start it again."""
    while True:
        try:
            await fn(st)
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            st.pump_restarts += 1
            log.exception("task %s crashed; restarting in 1s", name)
            await asyncio.sleep(1.0)


async def _handle_control(st: ServerState, data: dict) -> dict:
    """Apply a control request. Returns the resulting control info."""
    c = getattr(st.source, "control", None)
    if c is None or not st.replay:
        return st.control_info()
    if "speed" in data:
        try:
            sp = float(data["speed"])
            if sp > 0:
                c.speed = sp
        except (TypeError, ValueError):
            pass
    if "paused" in data:
        c.paused = bool(data["paused"])
    if "seek" in data:
        try:
            target = float(data["seek"])
        except (TypeError, ValueError):
            target = None
        if target is not None:
            async with st.lock:
                st.seeking = True
                was_idle = bool(st.ended or st.error)
                try:
                    st.engine.reset(keep_track=True)
                    await c.seek(target)
                except Exception as e:
                    log.exception("seek failed")
                    st.error = "seek failed: %s" % e
                finally:
                    st.seeking = False
                if was_idle:
                    st.restart.set()
                st.epoch += 1
    return st.control_info()


def _recording_info(info: dict) -> Optional[dict]:
    path = info.get("recording")
    if not path:
        return None
    size = info.get("recordingBytes")
    if size is None:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = None
    return {"path": path, "bytes": size, "diskFreeMB": info.get("diskFreeMB")}


def create_app(source, *, mode: Optional[str] = None, engine: Optional[RaceEngine] = None,
               frontend_dir: Optional[Path] = None, dev: bool = True, retry_live: bool = True,
               max_clients: int = MAX_CLIENTS, track_fallback: Any = None) -> FastAPI:
    """``track_fallback``: ``None`` = automatic (official circuit layout lookup in live/file mode, unless env
    ``F1_NO_TRACK_FALLBACK`` is set), ``False`` = off, an :class:`OfficialLayoutFallback` = use that (tests)."""
    mode = mode or _infer_mode(source)
    eng = engine or RaceEngine()
    if track_fallback is None:
        track_fallback = (OfficialLayoutFallback() if mode in ("live", "file") and not os.environ.get("F1_NO_TRACK_FALLBACK")
                          else False)
    st = ServerState(source, eng, mode, retry_live, max_clients=max_clients, fallback=track_fallback or None)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        st.tasks = [asyncio.create_task(_supervised("pump", st, _pump), name="pump"),
                    asyncio.create_task(_supervised("broadcast", st, _broadcaster), name="broadcast")]
        try:
            yield
        finally:
            for t in st.tasks:
                t.cancel()
            await asyncio.gather(*st.tasks, return_exceptions=True)
            closer = getattr(source, "close", None)
            if callable(closer):
                try:
                    r = closer()
                    if asyncio.iscoroutine(r):
                        await r
                except Exception:
                    log.exception("source close failed")

    app = FastAPI(title="F1 Live Dashboard", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.server = st
    app.state.engine = eng

    # ---------------------------------------------------------------- REST
    def json_resp(obj):
        return Response(content=_safe_dumps(obj), media_type="application/json")

    @app.get("/api/state")
    async def api_state():
        return json_resp(eng.state())

    @app.get("/api/history")
    async def api_history():
        return json_resp(eng.history())

    @app.get("/api/track")
    async def api_track():
        return json_resp({"type": "track", "track": eng.track_dict()})

    @app.post("/api/track/reset")
    async def api_track_reset():
        """Forget the current track outline (wrong/odd track, or a new circuit) and learn/fetch a fresh one."""
        reset_track(st, "api")
        st.epoch += 1
        return json_resp({"ok": True, "trackVersion": eng.track_version})

    @app.get("/healthz")
    async def healthz():
        return json_resp({"ok": True})

    @app.get("/api/status")
    async def api_status():
        info = st.control_info()
        age = None if st.last_msg_mono is None else round(time.monotonic() - st.last_msg_mono, 2)
        ph = st.phase(info)
        tr = eng.tb.track
        return json_resp({
            "mode": st.mode, "replay": st.replay, "phase": ph, "waiting": ph == "waiting", "message": st.status_text(info),
            "clients": len(st.clients), "maxClients": st.max_clients, "rejectedClients": st.rejected_clients,
            "msgsPerSec": st.msgs_per_sec(), "totalMsgs": st.total_msgs, "lastMessageAgeSec": age,
            "ended": st.ended, "error": st.current_error(info), "uptimeSec": round(time.time() - st.started_at, 1),
            "recording": _recording_info(info),
            "control": info, "engine": eng.stats(), "trackVersion": eng.track_version,
            "track": {"built": tr is not None, "source": getattr(tr, "source", None), "resets": getattr(eng.tb, "resets", 0),
                      "circuitKey": st.circuit[0], "year": st.circuit[1]},
            "applyErrors": st.apply_errors, "taskRestarts": st.pump_restarts, "utc": eng.utc,
        })

    @app.post("/api/control")
    async def api_control(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        info = await _handle_control(st, body)
        return json_resp({"ok": st.replay, "control": info})

    # ---------------------------------------------------------------- WebSocket
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        if len(st.clients) >= st.max_clients:
            st.rejected_clients += 1
            try:
                await ws.send_text(_safe_dumps({"type": "error", "message": "too many viewers, try again later"}))
                await ws.close(code=1013)
            except Exception:  # noqa: BLE001
                pass
            return
        client = Client(ws)
        try:
            # Build the whole first snapshot synchronously and register the client in the same step, so nothing
            # broadcast in between can be missed; the writer task then sends it in order.
            client.put("hello", _safe_dumps(st.hello()))
            client.put("state", _safe_dumps(eng.state()))
            if eng.tb.track is not None:
                client.put("track", _safe_dumps({"type": "track", "track": eng.track_dict()}))
                st.viewers_have_track = True
            client.put("history", _safe_dumps(eng.history()))
        except Exception:
            log.exception("could not build the initial snapshot")
            return
        st.clients.add(client)
        wtask = asyncio.create_task(client.writer())
        try:
            while True:
                raw = await ws.receive_text()
                if len(raw) > MAX_CLIENT_MSG:
                    continue
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(msg, dict) and msg.get("type") == "control":
                    info = await _handle_control(st, msg)
                    st.broadcast("control", _safe_dumps({"type": "control", **{k: info[k] for k in ("speed", "paused") if k in info}}))
        except WebSocketDisconnect:
            pass
        except Exception:
            log.debug("ws closed", exc_info=True)
        finally:
            client.closed = True
            st.clients.discard(client)
            wtask.cancel()

    # ---------------------------------------------------------------- static frontend
    fdir = Path(frontend_dir) if frontend_dir else FRONTEND_DIR
    if fdir.is_dir():
        cls = NoCacheStatic if dev else StaticFiles
        app.mount("/", cls(directory=str(fdir), html=True), name="frontend")
    else:
        log.warning("frontend dir %s not found; static files disabled", fdir)
    return app
