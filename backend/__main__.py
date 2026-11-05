"""python -m backend --mode sim|file|live [options]"""
from __future__ import annotations

import argparse
import logging
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
    ap.add_argument("--mode", choices=["sim", "file", "live"], default="sim")
    ap.add_argument("--file", help="recording to replay (--mode file)")
    ap.add_argument("--speed", type=float, default=1.0, help="replay/sim speed multiplier")
    ap.add_argument("--laps", type=int, default=20, help="sim race length in laps")
    ap.add_argument("--seed", type=int, default=1, help="sim random seed")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--no-auth", action="store_true", help="live: connect to the F1 SignalR feed without an F1 TV token")
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn
    from .server import create_app

    source = build_source(args)
    app = create_app(source, mode=args.mode)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
