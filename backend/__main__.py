"""python -m backend --mode sim|file|live [options]      |      python -m backend doctor [--selftest] [...]"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import sys


def build_source(args):
    if args.mode == "sim":
        from .ingest.sources import SimSource
        return SimSource(seed=args.seed, speed=args.speed, laps=args.laps, session=args.session, time_scale=args.time_scale)
    if args.mode == "file":
        if not args.file:
            sys.exit("--mode file requires --file PATH")
        from .ingest.sources import FileSource
        return FileSource(args.file, speed=args.speed)
    from .ingest.sources import LiveSource
    return LiveSource(no_auth=args.no_auth, recordings_dir=args.recordings_dir)


def _port_in_use(host: str, port: int) -> bool:
    """True if something already listens on (host, port) - checked the way uvicorn will bind (SO_REUSEADDR)."""
    fam = socket.AF_INET6 if ":" in host else socket.AF_INET
    s = socket.socket(fam, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        return False
    except OSError:
        return True
    finally:
        s.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "doctor":
        from .doctor import main as doctor_main
        return doctor_main(argv[1:])

    ap = argparse.ArgumentParser(prog="python -m backend", description="F1 live dashboard server "
                                 "(run `python -m backend doctor` first on a race weekend)")
    env = os.environ.get
    ap.add_argument("--mode", choices=["sim", "file", "live"], default=env("F1_MODE", "sim"), help="env F1_MODE")
    ap.add_argument("--file", default=env("F1_FILE"), help="recording to replay (--mode file); env F1_FILE")
    ap.add_argument("--speed", type=float, default=float(env("F1_SPEED", "1")), help="replay/sim speed multiplier; env F1_SPEED")
    ap.add_argument("--laps", type=int, default=int(env("F1_LAPS", "20")), help="sim race length in laps")
    ap.add_argument("--session", choices=["race", "sprint", "fp", "quali", "sprint_quali"], default=env("F1_SESSION", "race"),
                    help="sim: which kind of session to simulate; env F1_SESSION")
    ap.add_argument("--time-scale", type=float, default=float(env("F1_TIME_SCALE", "1")),
                    help="sim fp/quali: scale part durations (0.3 = short test session)")
    ap.add_argument("--seed", type=int, default=int(env("F1_SEED", "1")), help="sim random seed")
    ap.add_argument("--port", type=int, default=int(env("PORT", "8000")), help="env PORT")
    ap.add_argument("--host", default=env("HOST", "0.0.0.0"), help="env HOST")
    ap.add_argument("--no-auth", action="store_true", default=env("F1_NO_AUTH", "") in ("1", "true", "yes"),
                    help="live: connect to the F1 SignalR feed without an F1 TV token; env F1_NO_AUTH")
    ap.add_argument("--recordings-dir", default=env("F1_RECORDINGS_DIR", "recordings"),
                    help="live: where the raw feed is recorded (<timestamp>.txt); env F1_RECORDINGS_DIR")
    ap.add_argument("--max-clients", type=int, default=int(env("F1_MAX_CLIENTS", "100")),
                    help="max concurrent browser WebSocket viewers; env F1_MAX_CLIENTS")
    ap.add_argument("--token", default=env("DASHBOARD_TOKEN"),
                    help="require this shared token (?token=, x-token header or cookie) for all requests; env DASHBOARD_TOKEN")
    ap.add_argument("--log-level", default=env("LOG_LEVEL", "info"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if _port_in_use(args.host, args.port):
        sys.exit("Port %d is already in use on %s. Stop the other process (an old dashboard server?) or pick another "
                 "port with --port. `python -m backend doctor` checks this." % (args.port, args.host))
    import uvicorn
    from .server import create_app

    source = build_source(args)
    app = create_app(source, mode=args.mode, max_clients=args.max_clients)
    if args.token:
        from .auth import TokenMiddleware
        app.add_middleware(TokenMiddleware, token=args.token)
    if args.mode == "live":
        logging.getLogger("backend").info(
            "live mode: recordings go to %s/ ; open http://localhost:%d - 'waiting for session' is normal until the "
            "session starts; the server never gives up reconnecting", os.path.abspath(args.recordings_dir), args.port)
    # ws_max_size: browsers only send tiny control messages; ws_ping keeps NAT/proxies from idling sockets out
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower(), ws_max_size=1 << 16,
                ws_ping_interval=20.0, ws_ping_timeout=30.0, timeout_graceful_shutdown=5)


if __name__ == "__main__":
    sys.exit(main())
