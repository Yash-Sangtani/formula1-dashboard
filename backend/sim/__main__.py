"""``python -m backend.sim --session quali --seed 1 --out recordings/sim.txt`` -- dump a recording."""
from __future__ import annotations

import argparse
import time

from .race import SimConfig, generate
from .raw import record_to_file
from .tracks import list_tracks

SESSIONS = ("race", "sprint", "fp", "quali", "sprint_quali")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backend.sim", description="Write a synthetic session recording")
    ap.add_argument("--session", choices=SESSIONS, default="race",
                    help="race (default) | sprint | fp (practice) | quali | sprint_quali")
    ap.add_argument("--laps", type=int, default=None, help="race/sprint laps (default 20 / 12); ignored for fp/quali")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--track", default="bahrain-like", help="one of: " + ", ".join(list_tracks()))
    ap.add_argument("--out", default="recordings/sim.txt")
    ap.add_argument("--decoded", action="store_true", help="write CarData/Position as plain dicts instead of .z blobs")
    ap.add_argument("--no-incidents", action="store_true", help="no retirements / safety car (race, sprint)")
    ap.add_argument("--snapshot", action="store_true",
                    help="start the recording with the initial full state as CompletionMessage-style lines")
    ap.add_argument("--time-scale", type=float, default=1.0, help="fp/quali: scale session & part durations")
    ap.add_argument("--practice", type=int, default=1, choices=(1, 2, 3), help="fp: Practice number")
    args = ap.parse_args(argv)
    laps = args.laps if args.laps is not None else (12 if args.session == "sprint" else 20)
    t0 = time.time()
    race = generate(SimConfig(seed=args.seed, laps=laps, track=args.track, incidents=not args.no_incidents,
                              session=args.session, snapshot_first=args.snapshot, time_scale=args.time_scale,
                              practice=args.practice))
    n = record_to_file(race, args.out, encode=not args.decoded)
    dur = race.end_utc - race.times[0]
    print("wrote %d lines to %s (%s time %.0f s, generated in %.1f s)" % (n, args.out, args.session, dur, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
