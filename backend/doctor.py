"""``python -m backend doctor`` - pre-flight checks for a live race weekend.

    python -m backend doctor                 environment + credentials + network + a short real live-timing probe
    python -m backend doctor --selftest      offline: prove the install works (sim -> engine -> server, fake live feed,
                                              the real fastf1 client's file format) - no network, no credentials
    python -m backend doctor --probe-seconds 30 --no-login --skip-network --recordings-dir D --port P --no-auth

Each line is ``[PASS]`` / ``[WARN]`` / ``[FAIL]`` / ``[INFO]`` with a fix hint under failures.  The exit code is 1 when
anything FAILed (warnings do not fail).  No secret is ever printed: credentials are reported as present/absent and the
token only by expiry and subscription status.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import os
import shutil
import socket
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

HOST = "livetiming.formula1.com"
NEGOTIATE_URL = "https://livetiming.formula1.com/signalrcore/negotiate"
TESTED_FASTF1 = "3.9"          # major.minor this code was reviewed against
MIN_DISK_FAIL_MB = 200
MIN_DISK_WARN_MB = 2000


@dataclass
class Result:
    status: str  # PASS | WARN | FAIL | INFO
    name: str
    detail: str = ""
    hint: str = ""

    def line(self) -> str:
        s = "[%s] %s" % (self.status, self.name)
        if self.detail:
            s += ": " + self.detail
        if self.hint and self.status in ("FAIL", "WARN"):
            s += "\n         fix: " + self.hint
        return s


def P(name, detail="", hint=""):
    return Result("PASS", name, detail, hint)


def W(name, detail="", hint=""):
    return Result("WARN", name, detail, hint)


def F(name, detail="", hint=""):
    return Result("FAIL", name, detail, hint)


def I(name, detail=""):  # noqa: E741
    return Result("INFO", name, detail)


# =============================================================================== dependency injection point (tests)
class Deps:
    """Everything that touches the outside world, so tests can fake it."""

    def __init__(self, env: Optional[Dict[str, str]] = None) -> None:
        self.env = os.environ if env is None else env

    def module_version(self, name: str) -> Optional[str]:
        try:
            mod = importlib.import_module(name)
        except Exception:  # noqa: BLE001
            return None
        return str(getattr(mod, "__version__", "?"))

    def has_module(self, name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            return False

    def python_version(self) -> Tuple[int, int, int]:
        return tuple(sys.version_info[:3])  # type: ignore[return-value]

    def disk_free_mb(self, path: str) -> int:
        return int(shutil.disk_usage(path).free // (1 << 20))

    def port_in_use(self, host: str, port: int) -> bool:
        from .__main__ import _port_in_use

        return _port_in_use(host, port)

    def resolve(self, host: str) -> List[str]:
        return sorted({ai[4][0] for ai in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)})

    def negotiate(self, timeout: float = 10.0) -> Tuple[int, Dict[str, str]]:
        """OPTIONS the negotiate URL exactly as fastf1's client does; returns (status, cookies)."""
        import requests

        r = requests.options(NEGOTIATE_URL, timeout=timeout)
        return r.status_code, dict(r.cookies)

    def token_state(self) -> Optional[Dict[str, Any]]:
        """Stored fastf1 token (no network, no prompt): {'valid', 'expires_at', 'status', 'product', 'method'} or None."""
        from .ingest.sources import _f1_account

        acct = _f1_account()
        if acct is None or not hasattr(acct, "token_manager"):
            return {"unsupported": True}
        tok = acct.token_manager.get_stored()
        method = acct.get_auth_method() if hasattr(acct, "get_auth_method") else "password"
        if tok is None:
            return {"none": True, "method": method}
        try:
            claims = tok.claims
        except Exception:  # noqa: BLE001
            claims = {}
        return {"valid": tok.is_valid(), "expires_at": tok.expires_at, "method": method,
                "status": claims.get("SubscriptionStatus"), "product": claims.get("SubscribedProduct")}

    def login(self) -> Optional[str]:
        """Sign in with the environment credentials via fastf1 (stores the token). None = ok, else the problem."""
        from .ingest.sources import _f1_account

        acct = _f1_account()
        if acct is None:
            return "fastf1 sign-in API not found"
        try:
            acct.get_token(force_renew=False)
            return None
        except Exception as e:  # noqa: BLE001
            return "%s: %s" % (type(e).__name__, e)

    def probe(self, seconds: float, no_auth: bool) -> Dict[str, Any]:
        """Real SignalR connect + subscribe for ``seconds`` using the production LiveSource into a temp dir."""
        from .ingest.sources import LiveSource

        async def go() -> Dict[str, Any]:
            topics: Counter = Counter()
            seen = {"connected": False, "error": None, "info": {}}
            with tempfile.TemporaryDirectory(prefix="f1doctor_") as d:
                src = LiveSource(no_auth=no_auth, recordings_dir=d, connect_timeout=min(30.0, max(5.0, seconds - 1)),
                                 retry_delay=2.0, waiting_timeout=3600.0)

                async def run() -> None:
                    async for m in src.messages():
                        topics[m.topic] += 1

                task = asyncio.create_task(run())
                t0 = time.monotonic()
                while time.monotonic() - t0 < seconds:
                    await asyncio.sleep(0.25)
                    info = src.control.info()
                    seen["connected"] = seen["connected"] or bool(info.get("connected"))
                    seen["error"] = info.get("error") or seen["error"]
                    seen["info"] = info
                task.cancel()
                try:
                    await task
                except BaseException:  # noqa: BLE001
                    pass
                await asyncio.to_thread(src.close)
            return {"topics": dict(topics), **seen}

        return asyncio.run(go())


# =============================================================================== checks
def check_python(d: Deps) -> List[Result]:
    v = d.python_version()
    s = "%d.%d.%d" % v
    if v < (3, 10):
        return [F("Python", s, "install Python 3.11+ (3.13 is what this was tested on) and recreate the virtualenv")]
    if v < (3, 11):
        return [W("Python", s + " (3.11+ recommended, 3.13 tested)", "upgrade Python if you see odd errors")]
    return [P("Python", s)]


def _parse_ver(s: str) -> Tuple[int, ...]:
    out = []
    for part in s.split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def check_packages(d: Deps) -> List[Result]:
    out: List[Result] = []
    ff = d.module_version("fastf1")
    if ff is None:
        out.append(F("fastf1", "not installed", "pip install -r requirements.txt   (live mode needs fastf1 for the F1 feed client)"))
    else:
        if _parse_ver(ff)[:2] < (3, 5):
            out.append(F("fastf1", ff + " is too old for the SignalR-Core live client", "pip install -U fastf1"))
        elif ff.split(".")[:2] != TESTED_FASTF1.split("."):
            out.append(W("fastf1", "%s (reviewed against %s.x; the live client internals may differ)" % (ff, TESTED_FASTF1),
                         "if live mode misbehaves: pip install 'fastf1==3.9.2'"))
        else:
            out.append(P("fastf1", ff))
    for name, why in (("fastapi", "web server"), ("uvicorn", "web server"), ("numpy", "track geometry")):
        v = d.module_version(name)
        out.append(P(name, v) if v else F(name, "not installed (%s)" % why, "pip install -r requirements.txt"))
    if not (d.has_module("websockets") or d.has_module("wsproto")):
        out.append(F("websocket library", "neither 'websockets' nor 'wsproto' installed - the browser would get no live updates",
                     "pip install websockets"))
    else:
        out.append(P("websocket library", "websockets" if d.has_module("websockets") else "wsproto"))
    if ff is not None:
        missing = [m for m in ("signalrcore", "requests", "jwt") if not d.has_module(m)]
        if missing:
            out.append(F("fastf1 dependencies", "missing " + ", ".join(missing), "pip install -U fastf1"))
    return out


def check_recordings(d: Deps, path: str) -> List[Result]:
    try:
        os.makedirs(path, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".doctor_", delete=True) as f:
            f.write(b"x")
            f.flush()
    except OSError as e:
        return [F("recordings dir", "%s is not writable (%s)" % (os.path.abspath(path), e),
                  "choose another folder with --recordings-dir, or fix its permissions")]
    out = [P("recordings dir", "%s is writable" % os.path.abspath(path))]
    try:
        free = d.disk_free_mb(path)
    except OSError:
        return out
    msg = "%.1f GB free (a session writes tens to a couple of hundred MB; not measured here)" % (free / 1024)
    if free < MIN_DISK_FAIL_MB:
        out.append(F("disk space", "only %d MB free" % free, "free some disk space or point --recordings-dir at a bigger drive"))
    elif free < MIN_DISK_WARN_MB:
        out.append(W("disk space", "%d MB free" % free, "free up space before a long session"))
    else:
        out.append(P("disk space", msg))
    return out


def check_port(d: Deps, host: str, port: int) -> List[Result]:
    if d.port_in_use(host, port):
        return [F("port %d" % port, "already in use on %s" % host,
                  "stop the other process (an old `python -m backend`?) or use --port N")]
    return [P("port %d" % port, "free")]


def check_credentials(d: Deps, no_auth: bool, login: bool = True) -> List[Result]:
    if no_auth:
        return [I("F1 credentials", "skipped (--no-auth: the feed is tried unauthenticated; F1 serves that only intermittently)")]
    have_env = bool(d.env.get("F1_USERNAME") and d.env.get("F1_PASSWORD"))
    try:
        ts = d.token_state()
    except Exception as e:  # noqa: BLE001
        ts = {"unsupported": True}
        err = "%s: %s" % (type(e).__name__, e)
        return [W("F1 token", "could not read the stored token (%s)" % err, "run `python -m fastf1 auth f1tv --status`")]
    out: List[Result] = []
    assert ts is not None
    if ts.get("valid"):
        exp = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts["expires_at"]))
        left_h = (ts["expires_at"] - time.time()) / 3600
        status = ts.get("status")
        detail = "stored token valid until %s (%.0f h left); subscription status: %s; product: %s" % (
            exp, left_h, status, ts.get("product"))
        if status is not None and str(status).lower() not in ("active", "enabled"):
            out.append(W("F1 token", detail, "your F1 TV subscription may not be active - check your F1 account"))
        elif left_h < 6:
            out.append(W("F1 token", detail, "token expires soon; the server renews it with F1_USERNAME/F1_PASSWORD if set, "
                                           "otherwise run `python -m fastf1 auth f1tv --authenticate` again"))
        else:
            out.append(P("F1 token", detail))
        out.append(I("F1_USERNAME/F1_PASSWORD", "set (will renew the token automatically)" if have_env else
                     "not set - fine now, but the token cannot be renewed automatically when it expires"))
        return out
    method = ts.get("method", "password")
    if ts.get("unsupported"):
        out.append(W("F1 token", "this fastf1 version has no token API I know of", "check `python -m fastf1 auth f1tv --status`"))
    elif ts.get("none"):
        out.append(I("F1 token", "none stored yet"))
    else:
        out.append(I("F1 token", "stored token is expired"))
    if method == "password" and have_env:
        out.append(P("F1_USERNAME/F1_PASSWORD", "both set"))
        if login:
            problem = d.login()
            if problem is None:
                out.append(P("F1 sign-in", "succeeded; token stored for the server"))
            else:
                out.append(F("F1 sign-in", problem, "check username/password and that your F1 TV subscription is active; "
                                                    "if the network blocks account.formula1.com the server cannot sign in either"))
        else:
            out.append(I("F1 sign-in", "not attempted (--no-login)"))
        return out
    if method == "password":
        out.append(F("F1 credentials", "no valid token and F1_USERNAME / F1_PASSWORD are not both set",
                     "export F1_USERNAME=... F1_PASSWORD=...  (or run `python -m fastf1 auth f1tv --authenticate` once), "
                     "or start the server with --no-auth"))
    else:
        out.append(F("F1 credentials", "sign-in method %r needs an interactive browser sign-in" % method,
                     "run `python -m fastf1 auth f1tv --authenticate`"))
    return out


def _proxy_env(d: Deps) -> Optional[str]:
    for k in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        if d.env.get(k):
            return k
    return None


def check_network(d: Deps) -> Tuple[List[Result], bool]:
    """Returns (results, negotiate_ok)."""
    out: List[Result] = []
    pk = _proxy_env(d)
    if pk:
        out.append(W("proxy", "%s is set: the HTTPS negotiate request uses it, but the websocket library (signalrcore) may "
                             "connect directly" % pk, "if the probe below fails while negotiate passes, unset the proxy "
                                                       "variable or allow outbound wss://%s" % HOST))
    try:
        ips = d.resolve(HOST)
        out.append(P("DNS %s" % HOST, ", ".join(ips[:3])))
    except OSError as e:
        if pk:  # a proxy resolves names itself; local DNS failing is expected behind some proxies
            out.append(W("DNS %s" % HOST, "not resolvable locally (%s) - the proxy may still resolve it" % e,
                         "if the next check fails too, check the proxy / your internet connection"))
        else:
            out.append(F("DNS %s" % HOST, "cannot resolve (%s)" % e,
                         "check your internet connection / DNS / VPN; corporate or hotel networks sometimes block it"))
            return out, False
    try:
        status, cookies = d.negotiate()
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        low = ("%s %s" % (name, e)).lower()
        if "ssl" in low or "certificate" in low:
            hint = "TLS verification failed - a proxy/antivirus is probably intercepting HTTPS; set REQUESTS_CA_BUNDLE to its CA file"
        elif "proxy" in low:
            hint = "check HTTPS_PROXY / your proxy settings"
        elif "timeout" in low or "timed out" in low:
            hint = "request timed out - firewall or very slow link; try another network"
        else:
            hint = "check your internet connection / firewall (outbound HTTPS to %s must be allowed)" % HOST
        out.append(F("HTTPS %s" % NEGOTIATE_URL.replace("https://", ""), "%s: %s" % (name, str(e)[:160]), hint))
        return out, False
    if "AWSALBCORS" in cookies:
        out.append(P("negotiate (OPTIONS)", "HTTP %d, AWSALBCORS cookie received" % status))
        return out, True
    out.append(F("negotiate (OPTIONS)", "HTTP %d but no AWSALBCORS cookie - fastf1's client needs it (KeyError otherwise)" % status,
                 "usually a blocking proxy/captive portal or F1 changed the service; try another network, then update fastf1"))
    return out, False


def check_probe(d: Deps, seconds: float, no_auth: bool) -> List[Result]:
    try:
        r = d.probe(seconds, no_auth)
    except Exception as e:  # noqa: BLE001
        return [F("live probe", "%s: %s" % (type(e).__name__, e), "see the checks above; rerun with --log-level debug elsewhere")]
    topics: Dict[str, int] = r.get("topics") or {}
    info: Dict[str, Any] = r.get("info") or {}
    err = r.get("error")
    session = info.get("session") or {}
    label = ", ".join(x for x in (session.get("meeting"), session.get("name")) if x)
    top = ", ".join("%s x%d" % kv for kv in sorted(topics.items(), key=lambda kv: -kv[1])[:8])
    live_topics = {t for t in topics if t not in ("Heartbeat",)}
    if not r.get("connected"):
        if err:
            hint = ("sign-in problem: fix the F1 credentials above" if "sign-in" in err.lower() or "token" in err.lower() or
                    "authentication" in err.lower() else
                    "network problem: fix the DNS/HTTPS checks above (a proxy often allows HTTPS but blocks websockets)")
            return [F("live probe", "could not connect in %ds: %s" % (seconds, err), hint)]
        return [W("live probe", "still connecting after %ds, no error reported" % seconds, "rerun with --probe-seconds 45")]
    if info.get("sessionStatus") in ("Started", "Aborted") and live_topics:
        return [P("live probe", "connected and RECEIVING a live session%s: %s" % (" (%s)" % label if label else "", top))]
    if live_topics:
        return [P("live probe", "connected and subscribed; got %s. Session status: %s%s - no session is running right now "
                                "(that is normal outside sessions; the server will show 'waiting for session')"
                  % (top, info.get("sessionStatus"), " [feed shows %s]" % label if label else ""))]
    return [P("live probe", "connected; the feed sent nothing in %ds - no session is running right now (normal; the server will "
                            "show 'waiting for session' and start as soon as data flows)" % seconds)]


# =============================================================================== selftest
def _selftest_sim(laps: int = 4) -> List[Result]:
    try:
        from starlette.testclient import TestClient
    except ImportError as e:
        return [W("selftest server", "starlette TestClient needs httpx (%s)" % e, "pip install httpx")]
    import json

    from .ingest.sources import SimSource
    from .server import create_app

    app = create_app(SimSource(seed=1, speed=100, laps=laps), mode="sim", frontend_dir=None, track_fallback=False)
    out: List[Result] = []
    t0 = time.time()
    try:
        with TestClient(app) as c:
            kinds: Counter = Counter()
            with c.websocket_connect("/ws") as ws:  # live viewer while the sim runs (bounded: sim keeps producing frames)
                first = json.loads(ws.receive_text())
                kinds[first["type"]] += 1
                for _ in range(12):
                    kinds[json.loads(ws.receive_text())["type"]] += 1
            if first.get("type") != "hello":
                out.append(F("selftest sim->server", "first WebSocket frame was %r, expected hello" % first.get("type")))
                return out
            end = time.time() + 60
            status: Dict[str, Any] = {}
            while time.time() < end:
                status = c.get("/api/status").json()
                if status.get("ended"):
                    break
                time.sleep(0.2)
            with c.websocket_connect("/ws") as ws:  # late joiner gets the full snapshot
                late = [json.loads(ws.receive_text())["type"] for _ in range(4)]
            state = c.get("/api/state").json()
            hist = c.get("/api/history").json()
            track = c.get("/api/track").json().get("track")
            status = c.get("/api/status").json()
        if late[:2] != ["hello", "state"] or "history" not in late:
            kinds["late-join-snapshot-bad"] += 1
        for k in late:
            kinds[k] += 1
        probs = []
        if not status.get("ended"):
            probs.append("sim did not finish in 60 s")
        if len(state.get("cars") or {}) < 10:
            probs.append("only %d cars in state" % len(state.get("cars") or {}))
        if not track or len(track.get("points") or []) < 100:
            probs.append("no track outline was learned")
        if not hist.get("laps"):
            probs.append("no lap history")
        if status.get("applyErrors") or (status.get("engine") or {}).get("errors"):
            probs.append("engine errors: %s/%s" % (status.get("applyErrors"), (status.get("engine") or {}).get("errors")))
        for k in ("fast", "track", "history", "state"):
            if not kinds[k]:
                probs.append("never received a %r frame" % k)
        if kinds["late-join-snapshot-bad"]:
            probs.append("a late-joining viewer did not get hello/state/history first")
        if probs:
            out.append(F("selftest sim->engine->server", "; ".join(probs), "run `pip install -r requirements.txt` again; if it persists, "
                                                                           "run `python -m pytest backend/tests -q`"))
        else:
            out.append(P("selftest sim->engine->server", "%d msgs -> %d cars, %d laps of history, track %d pts, frames %s in %.1fs" % (
                status["totalMsgs"], len(state["cars"]), hist.get("maxLap", 0), len(track["points"]), dict(kinds), time.time() - t0)))
    except Exception as e:  # noqa: BLE001
        out.append(F("selftest sim->engine->server", "%s: %s" % (type(e).__name__, e), "reinstall requirements; run the test-suite"))
    return out


def _selftest_live_pipeline(laps: int = 3) -> List[Result]:
    """Fake fastf1 client replays a simulated race in the exact on-disk format -> LiveSource tails it -> engine; then the
    recording is replayed with FileSource into a second engine: both must agree."""
    from .engine import RaceEngine
    from .ingest.messages import to_raw_line
    from .ingest.sources import FileSource, LiveSource
    from .sim import SimConfig, generate

    msgs = generate(SimConfig(seed=2, laps=laps)).messages()
    lines = []
    first = {}
    for m in msgs:
        if m.topic in ("DriverList", "SessionInfo") and m.topic not in first:
            first[m.topic] = m
    import json

    for m in first.values():  # CompletionMessage-style snapshot lines: JSON string payload, empty timestamp
        lines.append(str([m.topic, json.dumps(m.data), ""]))
    lines += [to_raw_line(m, True) for m in msgs]
    ref = RaceEngine()
    for m in msgs:
        ref.apply(m)

    class FakeClient:
        def __init__(self, filename, filemode, timeout, no_auth):
            self.filename, self.filemode, self.timeout = filename, filemode, timeout

        def start(self):
            with open(self.filename, self.filemode, encoding="utf-8") as f:
                for i in range(0, len(lines), 500):
                    f.write("\n".join(lines[i:i + 500]) + "\n")
                    f.flush()
                    time.sleep(0.001)
            while self.timeout != 1:
                time.sleep(0.02)

    async def run_live(rec_dir: str) -> Tuple[RaceEngine, str, float]:
        src = LiveSource(recordings_dir=rec_dir, client_factory=FakeClient, poll=0.01, watch_interval=0.05)
        eng = RaceEngine()
        want = len(lines)
        n = 0
        t0 = time.time()
        async for m in src.messages():
            eng.apply(m)
            n += 1
            if n >= want:
                break
        path = src.path or ""
        await asyncio.to_thread(src.close)
        return eng, path, time.time() - t0

    async def run_file(path: str) -> RaceEngine:
        eng = RaceEngine()
        async for m in FileSource(path, speed=10000).messages():
            eng.apply(m)
        return eng

    out: List[Result] = []
    try:
        with tempfile.TemporaryDirectory(prefix="f1doctor_") as d:
            live_eng, path, dt = asyncio.run(asyncio.wait_for(run_live(d), 120))
            replay_eng = asyncio.run(asyncio.wait_for(run_file(path), 120))
            size = os.path.getsize(path)
        s_ref, s_live, s_rep = ref.state(), live_eng.state(), replay_eng.state()
        probs = []
        for name, s in (("live tail", s_live), ("recording replay", s_rep)):
            if s["order"] != s_ref["order"]:
                probs.append("%s: running order differs from the direct run" % name)
            if len(s["cars"]) != len(s_ref["cars"]):
                probs.append("%s: %d cars vs %d" % (name, len(s["cars"]), len(s_ref["cars"])))
        if live_eng.stats()["errors"]:
            probs.append("engine errors during live tail")
        if probs:
            out.append(F("selftest live pipeline", "; ".join(probs), "this is a bug in the ingest code - run the test-suite and report it"))
        else:
            out.append(P("selftest live pipeline", "%d raw lines tailed in %.1fs and replayed from the %.1f MB recording with identical results"
                         % (len(lines), dt, size / 1e6)))
    except Exception as e:  # noqa: BLE001
        out.append(F("selftest live pipeline", "%s: %s" % (type(e).__name__, e), "run the test-suite and report it"))
    return out


def _selftest_fastf1_format() -> List[Result]:
    """The REAL fastf1 client class (offline): feed one list message and one Subscribe CompletionMessage through
    ``_on_message`` and check our parser reads what it wrote."""
    try:
        from signalrcore.messages.completion_message import CompletionMessage  # noqa: F401
        import fastf1  # noqa: F401
    except ImportError as e:
        return [W("selftest fastf1 client format", "skipped (%s)" % e, "pip install fastf1")]
    from .ingest.messages import encode_z, parse_line
    from .ingest.sources import _default_client_factory
    from signalrcore.messages.completion_message import CompletionMessage as CM

    try:
        with tempfile.TemporaryDirectory(prefix="f1doctor_") as d:
            p = os.path.join(d, "x.txt")
            c = _default_client_factory(p, "a", 0, True)
            c._output_file = open(p, "a")
            c._on_message(["TimingData", {"Lines": {"1": {"Position": "1"}}}, "2025-03-16T04:00:01.000Z"])
            c._on_message(CM("1", {"CarData.z": encode_z({"Entries": []}), "DriverList": {"1": {"Tla": "VER"}}}, None))
            c._output_file.close()
            ms = [parse_line(x) for x in open(p, "rb").read().splitlines()]
        if [m.topic if m else None for m in ms] != ["TimingData", "CarData", "DriverList"]:
            return [F("selftest fastf1 client format", "unexpected parse result %r" % [m.topic if m else None for m in ms],
                      "fastf1's recording format changed? pip install 'fastf1==3.9.2'")]
        return [P("selftest fastf1 client format", "installed fastf1 %s writes lines this parser reads" % fastf1.__version__)]
    except Exception as e:  # noqa: BLE001
        return [F("selftest fastf1 client format", "%s: %s" % (type(e).__name__, e), "pip install 'fastf1==3.9.2'")]


def run_selftest() -> List[Result]:
    return _selftest_sim() + _selftest_live_pipeline() + _selftest_fastf1_format()


# =============================================================================== driver
def run_checks(opts: argparse.Namespace, deps: Optional[Deps] = None,
               selftest: Optional[Callable[[], List[Result]]] = None) -> List[Result]:
    d = deps or Deps()
    out: List[Result] = []
    out += check_python(d)
    out += check_packages(d)
    if opts.selftest:
        out += (selftest or run_selftest)()
        return out
    out += check_recordings(d, opts.recordings_dir)
    out += check_port(d, opts.host if opts.host not in ("0.0.0.0", "") else "127.0.0.1", opts.port)
    out += check_credentials(d, opts.no_auth, login=not opts.no_login and not opts.skip_network)
    if opts.skip_network:
        out.append(I("network checks", "skipped (--skip-network)"))
    else:
        net, ok = check_network(d)
        out += net
        if opts.probe_seconds <= 0:
            out.append(I("live probe", "skipped (--probe-seconds 0)"))
        elif not ok:
            out.append(W("live probe", "skipped because the negotiate check failed", "fix the network checks above first"))
        else:
            out += check_probe(d, opts.probe_seconds, opts.no_auth)
    return out


def main(argv: Optional[List[str]] = None, deps: Optional[Deps] = None, out=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backend doctor", description="Pre-flight checks for live mode.")
    env = os.environ.get
    ap.add_argument("--selftest", action="store_true", help="offline: run the simulator through engine+server and a fake live feed")
    ap.add_argument("--probe-seconds", type=float, default=15.0, help="seconds of real live-timing connection to try (0 = skip)")
    ap.add_argument("--skip-network", action="store_true", help="skip DNS/HTTPS/sign-in/probe")
    ap.add_argument("--no-login", action="store_true", help="do not attempt an F1 sign-in even if credentials are in the environment")
    ap.add_argument("--no-auth", action="store_true", default=env("F1_NO_AUTH", "") in ("1", "true", "yes"))
    ap.add_argument("--recordings-dir", default=env("F1_RECORDINGS_DIR", "recordings"))
    ap.add_argument("--port", type=int, default=int(env("PORT", "8000")))
    ap.add_argument("--host", default=env("HOST", "0.0.0.0"))
    opts = ap.parse_args(argv)
    results = run_checks(opts, deps)
    w = out or sys.stdout
    for r in results:
        print(r.line(), file=w)
    n = Counter(r.status for r in results)
    print("\n%d passed, %d warnings, %d failed" % (n["PASS"], n["WARN"], n["FAIL"]), file=w)
    if n["FAIL"]:
        print("NOT READY: fix the FAIL lines above.", file=w)
        return 1
    print("READY" + (" (with warnings)" if n["WARN"] else "") + (" - this was the offline self-test only" if opts.selftest else ""), file=w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
