"""python -m backend --mode sim|file|live [options]"""
from __future__ import annotations

import argparse
import logging
import os
import sys


def build_source(args):
    if args.mode == "sim":
        from .ingest.sources import SimSource
        return SimSource(seed=args.seed, speed=args.speed, laps=args.laps)
    if args.mode == "file":
        if not args.file:
            sys.exit("--mode file requires --file PATH")
        from .ingest.sources import FileSource
        return FileSource(args.file, speed=args.speed)
    from .ingest.sources import LiveSource
    return LiveSource(no_auth=args.no_auth)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m backend", description="F1 live dashboard server")
    env = os.environ.get
    ap.add_argument("--mode", choices=["sim", "file", "live"], default=env("F1_MODE", "sim"), help="env F1_MODE")
    ap.add_argument("--file", default=env("F1_FILE"), help="recording to replay (--mode file); env F1_FILE")
    ap.add_argument("--speed", type=float, default=float(env("F1_SPEED", "1")), help="replay/sim speed multiplier; env F1_SPEED")
    ap.add_argument("--laps", type=int, default=int(env("F1_LAPS", "20")), help="sim race length in laps")
    ap.add_argument("--seed", type=int, default=int(env("F1_SEED", "1")), help="sim random seed")
    ap.add_argument("--port", type=int, default=int(env("PORT", "8000")), help="env PORT")
    ap.add_argument("--host", default=env("HOST", "0.0.0.0"), help="env HOST")
    ap.add_argument("--no-auth", action="store_true", default=env("F1_NO_AUTH", "") in ("1", "true", "yes"),
                    help="live: connect to the F1 SignalR feed without an F1 TV token; env F1_NO_AUTH")
    ap.add_argument("--token", default=env("DASHBOARD_TOKEN"),
                    help="require this shared token (?token=, x-token header or cookie) for all requests; env DASHBOARD_TOKEN")
    ap.add_argument("--log-level", default=env("LOG_LEVEL", "info"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn
    from .server import create_app

    source = build_source(args)
    app = create_app(source, mode=args.mode)
    if args.token:
        from .auth import TokenMiddleware
        app.add_middleware(TokenMiddleware, token=args.token)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
