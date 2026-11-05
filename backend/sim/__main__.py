"""``python -m backend.sim --laps 20 --seed 1 --out recordings/sim.txt`` -- dump a recording."""
from __future__ import annotations

import argparse
import time

from .race import SimConfig, generate
from .raw import record_to_file
from .tracks import list_tracks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backend.sim", description="Write a synthetic race recording")
    ap.add_argument("--laps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--track", default="bahrain-like", help="one of: " + ", ".join(list_tracks()))
    ap.add_argument("--out", default="recordings/sim.txt")
    ap.add_argument("--decoded", action="store_true", help="write CarData/Position as plain dicts instead of .z blobs")
    ap.add_argument("--no-incidents", action="store_true", help="no retirements / safety car")
    args = ap.parse_args(argv)
    t0 = time.time()
    race = generate(SimConfig(seed=args.seed, laps=args.laps, track=args.track, incidents=not args.no_incidents))
    n = record_to_file(race, args.out, encode=not args.decoded)
    dur = race.end_utc - race.times[0]
    print("wrote %d lines to %s (race time %.0f s, generated in %.1f s)" % (n, args.out, dur, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
