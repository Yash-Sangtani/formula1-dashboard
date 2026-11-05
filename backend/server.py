"""FastAPI server: pumps a Source into the RaceEngine and broadcasts over WebSocket per docs/CONTRACT.md."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .engine import RaceEngine

log = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
FAST_HZ = 4.0
STATE_MIN_INTERVAL = 1.0
HISTORY_MIN_INTERVAL = 2.0


def _dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False, default=str)


def _safe_dumps(obj: Any) -> str:
    try:
        return _dumps(obj)
    except ValueError:  # NaN/inf sneaked in: sanitise
        def clean(o):
            if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
                return None
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            return o
        return _dumps(clean(obj))


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
                    await self.ws.send_text(text)
        except Exception:
            self.closed = True


class ServerState:
    def __init__(self, source, engine: RaceEngine, mode: str, retry_live: bool):
        self.source = source
        self.engine = engine
        self.mode = mode
        self.retry_live = retry_live
        self.clients: set[Client] = set()
        self.error: Optional[str] = None
        self.ended = False
        self.started_at = time.time()
        self.last_msg_mono: Optional[float] = None
        self.msg_times: deque = deque(maxlen=4096)
        self.total_msgs = 0
        self.epoch = 0            # bumps on seek / reset to force full resend
        self.tasks: list = []
        self.lock = asyncio.Lock()
        self.seeking = False
        self.restart = asyncio.Event()

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

    def hello(self) -> dict:
        return {"type": "hello", "mode": self.mode, "replay": self.replay, "info": self.control_info()}

    def broadcast(self, kind: str, text: str) -> None:
        for c in self.clients:
            c.put(kind, text)

    def broadcast_fast(self, frame: dict) -> None:
        if not self.clients:
            return
        text = _safe_dumps(frame)
        for c in self.clients:
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


async def _pump(st: ServerState):
    """Source -> engine. Retries for live sources; stays idle (serving last state) when a replay ends."""
    backoff = 2.0
    while True:
        try:
            n = 0
            async for msg in st.source.messages():
                st.engine.apply(msg)
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
    last_track_ver = -1
    last_state_t = 0.0
    last_hist_t = 0.0
    epoch = st.epoch
    period = 1.0 / FAST_HZ
    while True:
        await asyncio.sleep(period)
        try:
            now = time.monotonic()
            force = epoch != st.epoch
            epoch = st.epoch
            if force:
                last_rev = last_hist_rev = last_track_ver = -1
                last_state_t = last_hist_t = 0.0
            if not st.clients:
                eng.fast_delta()  # keep the dirty set from growing
                continue
            f = eng.fast_delta()
            if f is not None:
                st.broadcast_fast(f)
            if eng.track_version != last_track_ver and eng.tb.track is not None:
                last_track_ver = eng.track_version
                st.broadcast("track", _safe_dumps({"type": "track", "track": eng.track_dict()}))
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


def create_app(source, *, mode: Optional[str] = None, engine: Optional[RaceEngine] = None,
               frontend_dir: Optional[Path] = None, dev: bool = True, retry_live: bool = True) -> FastAPI:
    mode = mode or _infer_mode(source)
    eng = engine or RaceEngine()
    st = ServerState(source, eng, mode, retry_live)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        st.tasks = [asyncio.create_task(_pump(st), name="pump"), asyncio.create_task(_broadcaster(st), name="broadcast")]
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
        return JSONResponse(content=json.loads(_safe_dumps(obj)))

    @app.get("/api/state")
    async def api_state():
        return json_resp(eng.state())

    @app.get("/api/history")
    async def api_history():
        return json_resp(eng.history())

    @app.get("/api/track")
    async def api_track():
        return json_resp({"type": "track", "track": eng.track_dict()})

    @app.get("/api/status")
    async def api_status():
        age = None if st.last_msg_mono is None else round(time.monotonic() - st.last_msg_mono, 2)
        return json_resp({
            "mode": st.mode, "replay": st.replay, "clients": len(st.clients),
            "msgsPerSec": st.msgs_per_sec(), "totalMsgs": st.total_msgs, "lastMessageAgeSec": age,
            "ended": st.ended, "error": st.error, "uptimeSec": round(time.time() - st.started_at, 1),
            "control": st.control_info(), "engine": eng.stats(), "trackVersion": eng.track_version,
            "utc": eng.utc,
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
        client = Client(ws)
        try:
            await ws.send_text(_safe_dumps(st.hello()))
            await ws.send_text(_safe_dumps(eng.state()))
            if eng.tb.track is not None:
                await ws.send_text(_safe_dumps({"type": "track", "track": eng.track_dict()}))
            await ws.send_text(_safe_dumps(eng.history()))
        except Exception:
            return
        st.clients.add(client)
        wtask = asyncio.create_task(client.writer())
        try:
            while True:
                raw = await ws.receive_text()
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
