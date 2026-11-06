"""Synthetic *non-race* sessions: practice, qualifying and sprint qualifying (real raw formats, CONTRACT §6).

Unlike the race generator (a physical simulation) this one is *script driven*: every car gets a list of
laps (pit-out, out lap, push / long-run laps, cool-down, in lap, pit-in) with explicit start/end times and
a position-along-the-lap profile derived from the circuit's reference speed.  Timing messages (sectors,
lap times, best laps, ``TimeDiffToFastest`` ...) are then produced by replaying the lap ends in time order.

Shapes (all patches in the same topics as the race):

* ``TimingData.Lines[n]``: ``Position`` (by best lap), ``TimeDiffToFastest`` / ``TimeDiffToPositionAhead``
  (``""`` for P1), ``BestLapTime {Value, Lap}``, quali: ``BestLapTimes`` (3 x ``{Value, Lap}``),
  ``KnockedOut``, ``Cutoff``, ``NumberOfLaps`` (timed laps only), ``InPit`` / ``PitOut``, ``LastLapTime``,
  ``Sectors``, ``Speeds``.  No ``GapToLeader`` / ``IntervalToPositionAhead``.  Top level: ``SessionPart``,
  ``Withheld``.
* No ``LapCount``.  ``ExtrapolatedClock`` counts down (stopped during a red flag / between parts).
* ``SessionStatus``: Inactive, Started, [Aborted, Started]..., Finished, (next part: Started ...), Finalised, Ends.
* Cars in the garage keep sending ``CarData`` / ``Position`` with a fixed XY and ``Status: OffTrack``.
"""
from __future__ import annotations

import bisect
import math
import random
from array import array
from typing import Any, Dict, List, Optional, Tuple

from ..ingest.messages import format_ts
from .drivers import COMPOUNDS, DRIVERS, TEAMS
from .race import SimConfig, SimRace, _GEAR_HI, _GEAR_LO, _hms, _short_ts, fmt_gap, fmt_lap
from .tracks import Track, get_track

G_TIME = 8.0  # seconds to roll between garage and the pit box
PIT_V = 80.0 / 3.6
PIT_ENTRY, PIT_BOX, PIT_EXIT = 0.965, 0.985, 0.03

#: session -> (SessionInfo Type, Name, part durations [s], cars kept after each part, gaps between parts [s])
SPECS: Dict[str, Dict[str, Any]] = {
    "fp": dict(type="Practice", name="Practice %d", parts=[3600.0], keep=[None], gaps=[], dir="Practice_%d"),
    "quali": dict(type="Qualifying", name="Qualifying", parts=[1080.0, 900.0, 720.0], keep=[15, 10, None],
                  gaps=[420.0, 480.0], dir="Qualifying"),
    "sprint_quali": dict(type="Qualifying", name="Sprint Qualifying", parts=[720.0, 600.0, 480.0],
                         keep=[15, 10, None], gaps=[300.0, 360.0], dir="Sprint_Qualifying"),
}
NON_RACE = tuple(SPECS)

_PROFILES: Dict[str, Tuple[List[float], float]] = {}


def _profile(track: Track) -> Tuple[List[float], float]:
    """Normalised cumulative driving time over the lap at the reference speed (len n+1) and its total."""
    p = _PROFILES.get(track.name)
    if p is None:
        cum = [0.0]
        for i in range(track.n):
            cum.append(cum[-1] + track.step / max(track.v_ref[i], 5.0))
        tot = cum[-1]
        p = ([c / tot for c in cum], tot)
        _PROFILES[track.name] = p
    return p


class _Lap:
    __slots__ = ("kind", "t0", "t1", "f0", "f1", "comp", "stint", "new", "start_age", "valid", "aborted",
                 "wait", "sec", "mode")

    def __init__(self, kind: str, t0: float, t1: float, f0: float, f1: float) -> None:
        self.kind, self.t0, self.t1, self.f0, self.f1 = kind, t0, t1, f0, f1
        self.comp = "SOFT"
        self.stint = 0
        self.new = True
        self.start_age = 0
        self.valid = True
        self.aborted = False
        self.wait = 0.0
        self.sec: Optional[Tuple[float, float, float]] = None
        self.mode = ""


class _Win:
    """One timed window (practice session or one qualifying part)."""

    def __init__(self, k: int, ts: float, dur: float, keep: Optional[int], part: Optional[int]) -> None:
        self.k, self.ts, self.dur, self.keep, self.part = k, ts, dur, keep, part
        self.r0: Optional[float] = None
        self.r1: Optional[float] = None
        self.red_dur = 0.0
        self.we = ts + dur  # wall end (after any red flag extension)

    def set_red(self, off: float, red_dur: float) -> None:
        self.r0 = self.ts + off
        self.r1 = self.r0 + red_dur
        self.red_dur = red_dur
        self.we = self.ts + self.dur + red_dur

    def wall(self, off: float) -> float:
        """Clock offset (s since the window's clock start) -> wall time."""
        if self.r0 is not None and self.ts + off >= self.r0:
            return self.ts + off + self.red_dur
        return self.ts + off

    def remaining(self, t: float) -> float:
        if self.r0 is None:
            return max(0.0, self.dur - (t - self.ts))
        return max(0.0, self.dur - (min(t, self.r0) - self.ts) - max(0.0, t - self.r1))


class _SCar:
    def __init__(self, sim: "_Sess", idx: int, d: Any) -> None:
        self.sim = sim
        self.idx = idx
        self.num, self.tla, self.first, self.last, self.team = d
        self.color, self.team_delta = TEAMS[d.team]
        self.drv_delta = sim.rng.gauss(0.0, 0.22)
        self.laps: List[_Lap] = []
        self.starts: List[float] = []
        self.n_stints = 0
        # session-state while replaying
        self.pos = idx + 1
        self.timed = 0
        self.lapno = 0
        self.best: Optional[float] = None  # best valid lap of the current window
        self.best_t = 0.0
        self.best_lap_no = 0
        self.pb_sec = [1e9, 1e9, 1e9]
        self.stint_laps = 0
        self.stint_no = -1
        self.last_diff: Tuple[str, str] = ("", "")
        self.cutoff = False
        self.pos0 = idx + 1
        self.out = False  # knocked out
        # garage slot: beside the pit box, spread along the lane
        tr = sim.track
        bx, by, bz = tr.point_at(PIT_BOX * tr.length)
        x2, y2, _ = tr.point_at(PIT_BOX * tr.length + 5.0)
        tx, ty = x2 - bx, y2 - by
        n = math.hypot(tx, ty) or 1.0
        tx, ty = tx / n, ty / n
        off = (idx - len(DRIVERS) / 2.0) * 4.0
        self.box = (bx, by, bz)
        self.garage = (bx - ty * (14.0 + 3.0 * (idx % 2)) + tx * off, by + tx * (14.0 + 3.0 * (idx % 2)) + ty * off, bz)

    # ------------------------------------------------------------------ geometry at time t
    def state(self, t: float) -> Tuple[float, float, float, int, float, str]:
        """(x, y, z, on_track, f, kind) -- ``kind`` is ``idle`` when parked in the garage."""
        sim = self.sim
        i = bisect.bisect_right(self.starts, t) - 1
        if i < 0 or t >= self.laps[i].t1:
            g = self.garage
            return g[0], g[1], g[2], 0, 0.0, "idle"
        lap = self.laps[i]
        tr, L = sim.track, sim.L
        u = t - lap.t0
        if lap.kind == "pitout":
            if u < G_TIME:
                return (*_lerp(self.garage, self.box, u / G_TIME), 0, 0.0, "pitout")
            u2 = u - G_TIME - lap.wait
            if u2 <= 0:
                b = self.box
                return b[0], b[1], b[2], 1, PIT_BOX, "pitout"
            span = max(lap.t1 - lap.t0 - G_TIME - lap.wait, 1e-6)
            f = (PIT_BOX - 1.0) + (1.0 - PIT_BOX + PIT_EXIT) * min(1.0, u2 / span)
            x, y, z = tr.point_at((f % 1.0) * L)
            return x, y, z, 1, f % 1.0, "pitout"
        if lap.kind == "pitin":
            span = (lap.t1 - lap.t0) - G_TIME
            if u >= span:
                return (*_lerp(self.box, self.garage, (u - span) / G_TIME), 0, 0.0, "pitin")
            f = PIT_ENTRY + (PIT_BOX - PIT_ENTRY) * (u / max(span, 1e-6))
            x, y, z = tr.point_at(f * L)
            return x, y, z, 1, f, "pitin"
        cum, n = sim.cum, sim.track.n
        a, b = _T(cum, n, lap.f0), _T(cum, n, lap.f1)
        f = _Tinv(cum, n, a + (b - a) * min(1.0, u / max(lap.t1 - lap.t0, 1e-6)))
        x, y, z = tr.point_at((f % 1.0) * L)
        return x, y, z, 1, f % 1.0, lap.kind


def _lerp(a: Tuple[float, float, float], b: Tuple[float, float, float], u: float) -> Tuple[float, float, float]:
    u = min(1.0, max(0.0, u))
    return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u)


def _T(cum: List[float], n: int, f: float) -> float:
    f = min(1.0, max(0.0, f))
    u = f * n
    i = min(int(u), n - 1)
    return cum[i] + (cum[i + 1] - cum[i]) * (u - i)


def _Tinv(cum: List[float], n: int, T: float) -> float:
    i = bisect.bisect_right(cum, T) - 1
    i = min(max(i, 0), n - 1)
    d = cum[i + 1] - cum[i]
    return (i + ((T - cum[i]) / d if d > 0 else 0.0)) / n


class _Sess:
    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed * 7919 + 17)
        self.spec = SPECS[cfg.session]
        self.track = get_track(cfg.track)
        self.L = self.track.length
        self.tref = self.track.lap_time_ref
        self.cum, _ = _profile(self.track)
        self.base = cfg.base_utc
        self.T0 = cfg.pre_race_s
        self.scale = max(0.05, float(cfg.time_scale))
        self.dt = cfg.tick
        self.cars = [_SCar(self, i, d) for i, d in enumerate(DRIVERS[: cfg.cars])]
        self.by_num = {c.num: c for c in self.cars}
        self.race = SimRace(cfg, self.track, [c.num for c in self.cars])
        self.rcm_count = 0
        self.series_n = 0
        self.part_series_n = 0
        self.radio_count = 0
        self.track_code = "1"
        self.parts_info: List[Dict[str, Any]] = []
        self.quali = cfg.session != "fp"
        self.durs = [d * self.scale for d in self.spec["parts"]]
        self.total = sum(self.durs)
        self.red_part = self.rng.randrange(len(self.durs)) if self.quali else 0
        self.weather = {"air": 27.8, "track": 36.5, "hum": 41.0, "pres": 1012.4, "wind": 2.1, "dir": 128.0}
        # per-window replay state
        self.ov_best: Optional[float] = None
        self.ov_sec = [1e9, 1e9, 1e9]

    # ------------------------------------------------------------------ emission helpers
    def utc(self, t: float) -> float:
        return self.base + t

    def emit(self, t: float, topic: str, data: Any) -> None:
        r = self.race
        r.events.append((topic, data))
        r._items.append((self.base + t, 0, len(r.events) - 1))

    def _rcm(self, t: float, entry: Dict[str, Any], lap: Optional[int] = None) -> None:
        entry.setdefault("Utc", _short_ts(self.utc(t)))
        entry.setdefault("Lap", lap if lap is not None else 1)
        payload: Any = {"Messages": [entry]} if self.rcm_count == 0 else {"Messages": {str(self.rcm_count): entry}}
        self.rcm_count += 1
        self.emit(t, "RaceControlMessages", payload)

    def _series(self, t: float, key: str, value: Any) -> None:
        self.emit(t, "SessionData", {"StatusSeries": {str(self.series_n): {"Utc": format_ts(self.utc(t)), key: value}}})
        self.series_n += 1

    def _session_status(self, t: float, status: str) -> None:
        self.emit(t, "SessionStatus", {"Status": status})
        self._series(t, "SessionStatus", status)

    def _track_status(self, t: float, code: str, msg: str) -> None:
        self.track_code = code
        self.emit(t, "TrackStatus", {"Status": code, "Message": msg})
        self._series(t, "TrackStatus", code)

    def _timing(self, t: float, c: _SCar, patch: Dict[str, Any]) -> None:
        self.emit(t, "TimingData", {"Lines": {c.num: patch}})

    def _clock(self, t: float, win: _Win, running: bool, remaining: Optional[float] = None) -> None:
        rem = win.remaining(t) if remaining is None else remaining
        self.emit(t, "ExtrapolatedClock", {"Utc": format_ts(self.utc(t)), "Remaining": _hms(rem), "Extrapolating": running})

    # ------------------------------------------------------------------ prelude
    def _prelude(self) -> None:
        cfg, spec = self.cfg, self.spec
        t = 0.0
        bt = self.base
        stamp = format_ts(bt)
        name = spec["name"] % cfg.practice if "%d" in spec["name"] else spec["name"]
        sdir = spec["dir"] % cfg.practice if "%d" in spec["dir"] else spec["dir"]
        self.emit(t, "SessionInfo", {
            "Meeting": {"Key": 9999, "Name": "%s Grand Prix (SIM)" % self.track.name.split("-")[0].title(),
                        "OfficialName": "SIMULATED FORMULA 1 GRAND PRIX", "Location": "Synthia",
                        "Country": {"Key": 0, "Code": "SIM", "Name": "Simulation"},
                        "Circuit": {"Key": 0, "ShortName": self.track.name}},
            "ArchiveStatus": {"Status": "Generating"}, "Key": 99999, "Type": spec["type"], "Name": name,
            "StartDate": stamp[:19], "EndDate": format_ts(bt + self.total + 1800)[:19], "GmtOffset": "00:00:00",
            "Path": "2025/2025-03-16_Simulated_Grand_Prix/2025-03-16_%s/" % sdir,
        })
        if cfg.session == "fp":
            self.race.events[-1][1]["Number"] = cfg.practice
        self.emit(t, "DriverList", {c.num: {
            "RacingNumber": c.num, "BroadcastName": "%s %s" % (c.first[0], c.last.upper()),
            "FullName": "%s %s" % (c.first, c.last.upper()), "Tla": c.tla, "Line": c.idx + 1, "TeamName": c.team,
            "TeamColour": c.color, "FirstName": c.first, "LastName": c.last.upper(), "Reference": c.tla[:3] + "01",
            "CountryCode": "SIM"} for c in self.cars})
        self.emit(t, "SessionStatus", {"Status": "Inactive"})
        self.emit(t, "TrackStatus", {"Status": "1", "Message": "AllClear"})
        self.emit(t, "WeatherData", self._weather_data())
        self.emit(t, "ExtrapolatedClock", {"Utc": stamp, "Remaining": _hms(self.durs[0]), "Extrapolating": False})
        self.emit(t, "TimingAppData", {"Lines": {c.num: {"RacingNumber": c.num, "Line": c.idx + 1, "Stints": {}}
                                                  for c in self.cars}})
        lines = {}
        for c in self.cars:
            ln: Dict[str, Any] = {
                "Position": str(c.idx + 1), "Line": c.idx + 1, "TimeDiffToFastest": "", "TimeDiffToPositionAhead": "",
                "BestLapTime": {"Value": ""}, "NumberOfLaps": 0, "InPit": True, "PitOut": False, "Retired": False,
                "Stopped": False, "KnockedOut": False, "Cutoff": False}
            if self.quali:
                ln["BestLapTimes"] = [{"Value": ""} for _ in range(3)]
            lines[c.num] = ln
        top: Dict[str, Any] = {"Lines": lines, "Withheld": False}
        if self.quali:
            top["SessionPart"] = 1
        self.emit(t, "TimingData", top)
        self.emit(t, "Heartbeat", {"Utc": stamp})
        self.race.prelude_events = len(self.race.events)

    def _weather_data(self) -> Dict[str, str]:
        w = self.weather
        return {"AirTemp": "%.1f" % w["air"], "Humidity": "%.1f" % w["hum"], "Pressure": "%.1f" % w["pres"],
                "Rainfall": "0", "TrackTemp": "%.1f" % w["track"], "WindDirection": "%d" % round(w["dir"]),
                "WindSpeed": "%.1f" % w["wind"]}

    # ------------------------------------------------------------------ planning
    def _make_window(self, k: int, ts: float) -> _Win:
        keep = self.spec["keep"][k]
        win = _Win(k, ts, self.durs[k], keep, (k + 1) if self.quali else None)
        if k == self.red_part:
            r = random.Random("%s-red" % self.cfg.seed)
            if self.quali:
                off = win.dur * r.uniform(0.30, 0.55)
                rd = max(90.0, 360.0 * self.scale)
            else:
                off = win.dur * r.uniform(0.33, 0.45)
                rd = max(120.0, 480.0 * self.scale)
            win.set_red(off, rd)
        return win

    def _run_list(self, c: _SCar, win: _Win, stag: int) -> List[Tuple[float, str, str]]:
        rr = random.Random("%s-%s-%s-runs" % (self.cfg.seed, c.idx, win.k))
        D = win.dur
        runs: List[Tuple[float, str, str]] = []
        if self.quali:
            o1 = 2.0 + stag * 2.5 + rr.uniform(0, 6)
            o2 = D * rr.uniform(0.50, 0.64)
            c1 = rr.choices(["SOFT", "MEDIUM"], [3, 2] if win.k == 0 else [5, 1])[0]
            runs = [(win.wall(o1), "q", c1), (win.wall(o2), "q", "SOFT" if win.k else rr.choice(["SOFT", "SOFT", "MEDIUM"]))]
        else:
            offs = [D * rr.uniform(0.01, 0.06) + stag * 3.0, D * rr.uniform(0.25, 0.40), D * rr.uniform(0.52, 0.68)]
            mid = ["qualisim", "longrun"]
            rr.shuffle(mid)
            kinds = ["install"] + mid
            comps = {"install": rr.choice(["HARD", "MEDIUM"]), "qualisim": rr.choices(["SOFT", "MEDIUM"], [7, 3])[0],
                     "longrun": rr.choice(["MEDIUM", "HARD"])}
            for o, kd in zip(offs, kinds):
                runs.append((win.wall(o), kd, comps[kd]))
            if rr.random() < 0.35:
                runs.append((win.wall(D * rr.uniform(0.82, 0.88)), "qualisim", "SOFT"))
        return runs

    def _plan_car(self, c: _SCar, win: _Win, stag: int, occ: Any) -> List[_Lap]:
        laps: List[_Lap] = []
        t_prev = win.ts
        stint = c.n_stints
        for ri, (st, rk, comp) in enumerate(self._run_list(c, win, stag)):
            rr = random.Random("%s-%s-%s-%s" % (self.cfg.seed, c.idx, win.k, ri))
            st = max(st, t_prev + 25.0)
            run = self._build_run(c, win, rk, st, comp, rr, occ, stint)
            if run:
                laps.extend(run)
                t_prev = run[-1].t1
                stint += 1
        return laps

    def _build_run(self, c: _SCar, win: _Win, rk: str, st: float, comp: str, rr: random.Random,
                   occ: Any, stint: int) -> List[_Lap]:
        L, tref = self.L, self.tref
        wait = rr.uniform(0.0, 18.0) if self.quali else rr.uniform(0.0, 5.0)
        stagger = rr.uniform(2.0, 40.0)
        pit_dur = (1.0 - PIT_BOX + PIT_EXIT) * L / PIT_V
        p_len = G_TIME + wait + pit_dur
        if win.r0 is not None and st < win.r1 and st + p_len > win.r0 - 1.0:
            st = max(st, win.r1) + stagger
        if st >= win.we - 60.0:
            return []
        new = rr.random() < (0.9 if self.quali else 0.8)
        age0 = 0 if new else rr.randint(2, 6)
        if rk == "q":
            plan = ["out", "push"] + (["cool", "push"] if rr.random() < 0.3 else [])
        elif rk == "install":
            plan = ["out"] + ["push"] * rr.randint(2, 3)
        elif rk == "qualisim":
            plan = ["out", "push", "cool", "push"]
        else:
            plan = ["out"] + ["long"] * rr.randint(5, 9)
        mode = {"q": "quali", "install": "install", "qualisim": "fast", "longrun": "long"}[rk]
        draws = [(rr.gauss(0.0, 1.0), rr.random(), rr.random()) for _ in range(len(plan) + 2)]
        pl = _Lap("pitout", st, st + p_len, PIT_BOX - 1.0, PIT_EXIT)
        pl.wait = wait
        pl.comp, pl.stint, pl.new, pl.start_age = comp, stint, new, age0
        laps = [pl]
        t = pl.t1
        age = age0
        n_push = 0
        n_timed = 0
        sf = self.track.sector_fracs
        for j, kind in enumerate(plan):
            nz, u1, u2 = draws[j]
            if kind in ("push", "long") and t >= win.we:
                break
            if kind == "cool":
                dur = tref * (1.20 + 0.12 * u1)
                if t + dur >= win.we:
                    break
                lp = _Lap("cool", t, t + round(dur, 3), 0.0, 1.0)
            elif kind == "out":
                dur = (1.0 - PIT_EXIT) * tref * (1.22 + 0.14 * u1)
                lp = _Lap("out", t, t + round(dur, 3), PIT_EXIT, 1.0)
            else:
                tl = self._lap_time(c, win, mode, comp, age, n_timed, t, nz, occ)
                lp = _Lap(kind, t, t + round(tl, 3), 0.0, 1.0)
                lp.mode = mode
                v1 = round(tl * _T(self.cum, self.track.n, sf[0]), 3)
                v2 = round(tl * (_T(self.cum, self.track.n, sf[1]) - _T(self.cum, self.track.n, sf[0])), 3)
                lp.sec = (v1, v2, round(tl - v1 - v2, 3))
                lp.valid = not (self.quali and u2 < 0.08)
                n_timed += 1
                n_push += 1
            lp.comp, lp.stint = comp, stint
            laps.append(lp)
            t = lp.t1
            age += 1
        if n_push == 0:
            return []
        dur = PIT_ENTRY * tref * (1.28 + 0.14 * draws[-1][1])
        il = _Lap("in", t, t + round(dur, 3), 0.0, PIT_ENTRY)
        il.comp, il.stint = comp, stint
        laps.append(il)
        if win.r0 is not None:
            laps = self._cut_red(laps, win, comp, stint)
        t = laps[-1].t1
        pi = _Lap("pitin", t, t + 5.0 + G_TIME, PIT_ENTRY, PIT_BOX)
        pi.comp, pi.stint = comp, stint
        laps.append(pi)
        return laps

    def _cut_red(self, laps: List[_Lap], win: _Win, comp: str, stint: int) -> List[_Lap]:
        r0 = win.r0
        for i, lp in enumerate(laps):
            if lp.kind == "pitout" or not (lp.t0 < r0 < lp.t1):
                continue
            a, b = _T(self.cum, self.track.n, lp.f0), _T(self.cum, self.track.n, lp.f1)
            f_at = _Tinv(self.cum, self.track.n, a + (b - a) * (r0 - lp.t0) / (lp.t1 - lp.t0))
            head = laps[:i]
            if lp.t0 < r0 - 0.5:
                part = _Lap(lp.kind, lp.t0, r0, lp.f0, f_at)
                part.comp, part.stint, part.aborted = comp, stint, True
                head.append(part)
            f_in = f_at
            t = r0
            if lp.kind != "in" and f_at >= PIT_ENTRY:  # past the pit entry already: finish the lap first
                extra = _Lap("out", t, t + max(4.0, (1.0 - f_at) * self.tref * 1.3), f_at, 1.0)
                extra.comp, extra.stint = comp, stint
                head.append(extra)
                t, f_in = extra.t1, 0.0
            il = _Lap("in", t, t + max(6.0, (PIT_ENTRY - f_in) * self.tref * 1.4), f_in, PIT_ENTRY)
            il.comp, il.stint = comp, stint
            head.append(il)
            return head
        return laps

    def _lap_time(self, c: _SCar, win: _Win, mode: str, comp: str, age: int, k: int, t: float, nz: float,
                  occ: Any) -> float:
        comp_d, rate, _life = COMPOUNDS[comp]
        evo = -0.7 * min(1.0, max(0.0, (t - self.T0) / (self.total * 0.9)))
        base = self.tref + c.team_delta + c.drv_delta
        if mode == "quali":
            off = -1.3 + comp_d + rate * max(0, age - 1) + evo
        elif mode == "fast":
            off = -0.45 + comp_d + rate * max(0, age - 1) + evo
        elif mode == "install":
            off = 1.6 + comp_d + 0.4 * abs(nz)
        else:  # long run: fuel-heavy, consistent
            off = 0.9 + comp_d + rate * age + max(0.0, 0.6 - 0.03 * k) + evo
        tl = base + off + 0.12 * nz
        if occ is not None and mode in ("quali", "fast"):
            n = occ(t + 0.5 * tl)
            if n > 2:
                tl += min(0.9, 0.06 * (n - 2) * (0.6 + 0.4 * abs(nz)))
        return max(tl, 0.55 * self.tref)

    @staticmethod
    def _occupancy(plans: Dict[str, List[_Lap]]) -> Any:
        ev: List[Tuple[float, int]] = []
        for laps in plans.values():
            for lp in laps:
                if lp.kind in ("out", "cool", "in"):
                    ev.append((lp.t0, 1))
                    ev.append((lp.t1, -1))
        ev.sort()
        times = [e[0] for e in ev]
        cnt = []
        n = 0
        for _, d in ev:
            n += d
            cnt.append(n)

        def occ(t: float) -> int:
            i = bisect.bisect_right(times, t) - 1
            return cnt[i] if i >= 0 else 0

        return occ

    def _plan_window(self, win: _Win, parts: List[_SCar]) -> None:
        rng = random.Random("%s-win-%s" % (self.cfg.seed, win.k))
        order = parts[:]
        rng.shuffle(order)
        stag = {c.num: i for i, c in enumerate(order)}
        occ = None
        plans: Dict[str, List[_Lap]] = {}
        for _pass in (1, 2):
            plans = {c.num: self._plan_car(c, win, stag[c.num], occ) for c in parts}
            occ = self._occupancy(plans)
        for c in parts:
            c.laps.extend(plans[c.num])
            c.n_stints += len({lp.stint for lp in plans[c.num]})
        self._win_laps = plans

    # ------------------------------------------------------------------ replay of one window
    def _flags(self, value: float, idx: int, c: _SCar, valid: bool) -> Dict[str, Any]:
        ov = valid and value < self.ov_sec[idx] - 1e-9
        pe = valid and value < c.pb_sec[idx] - 1e-9
        if ov:
            self.ov_sec[idx] = value
        if pe:
            c.pb_sec[idx] = value
        return {"OverallFastest": ov, "PersonalFastest": pe, "Status": 2051 if ov else (2049 if pe else 2048)}

    def _rank(self, t: float, win: _Win, parts: List[_SCar], fixed_after: int) -> List[_SCar]:
        """Re-rank the participants by best lap and emit the Position / TimeDiff / Cutoff changes."""
        timed = sorted([c for c in parts if c.best is not None], key=lambda c: (c.best, c.best_t))
        untimed = sorted([c for c in parts if c.best is None], key=lambda c: c.pos0)
        order = timed + untimed
        fastest = timed[0].best if timed else None
        for i, c in enumerate(order):
            rank = i + 1
            patch: Dict[str, Any] = {}
            if c.pos != rank:
                c.pos = rank
                patch["Position"] = str(rank)
                patch["Line"] = rank
            if c.best is not None:
                d = ("" if i == 0 else fmt_gap(c.best - fastest), "" if i == 0 else fmt_gap(c.best - timed[i - 1].best))
                if d != c.last_diff:
                    c.last_diff = d
                    patch["TimeDiffToFastest"] = d[0]
                    patch["TimeDiffToPositionAhead"] = d[1]
            if win.keep:
                cut = c.best is not None and rank > win.keep
                if cut != c.cutoff:
                    c.cutoff = cut
                    patch["Cutoff"] = cut
            if patch:
                self._timing(t, c, patch)
        return order

    def _window_events(self, win: _Win, parts: List[_SCar]) -> List[Tuple[float, int, _SCar, str, Any]]:
        ev: List[Tuple[float, int, _SCar, str, Any]] = []
        seq = 0
        sf = self.track.sector_fracs
        n = self.track.n
        tot = self.tref
        for c in parts:
            for lp in self._win_laps[c.num]:
                if lp.kind == "pitout":
                    ev.append((lp.t1, seq, c, "pit_exit", lp))
                elif lp.kind == "in":
                    if not lp.aborted:
                        ev.append((lp.t1, seq, c, "pit_in", lp))
                elif lp.kind in ("out", "cool", "push", "long") and lp.f1 >= 1.0 and not lp.aborted:
                    if lp.kind in ("push", "long"):
                        tl = lp.t1 - lp.t0
                        ev.append((lp.t0 + lp.sec[0], seq, c, "sector0", lp))
                        ev.append((lp.t0 + lp.sec[0] + lp.sec[1], seq + 1, c, "sector1", lp))
                        seq += 2
                    ev.append((lp.t1, seq, c, "cross", lp))
                seq += 1
        ev.sort(key=lambda e: (e[0], e[1]))
        return ev

    def _speed_at(self, f: float, tl: float) -> int:
        vr = self.track.v_ref
        return int(round(vr[int(f * len(vr)) % len(vr)] * (self.tref / tl) * 3.6))

    def _process(self, win: _Win, parts: List[_SCar]) -> Tuple[float, List[_SCar]]:
        """Replay the window's laps in time order; return (time of the last lap end, final order)."""
        for c in parts:
            c.best = None
            c.best_t = 0.0
            c.pb_sec = [1e9, 1e9, 1e9]
            c.last_diff = ("", "")
            c.cutoff = False
            c.pos0 = c.pos
        self.ov_best = None
        self.ov_sec = [1e9, 1e9, 1e9]
        sf = self.track.sector_fracs
        t_last = win.we
        order: List[_SCar] = sorted(parts, key=lambda c: c.pos)
        for t, _s, c, kind, lp in self._window_events(win, parts):
            if kind == "pit_exit":
                c.stint_no = lp.stint
                c.stint_laps = lp.start_age
                self._timing(t, c, {"InPit": False, "PitOut": True})
                self.emit(t, "TimingAppData", {"Lines": {c.num: {"Stints": {str(lp.stint): {
                    "LapFlags": 0, "Compound": lp.comp, "New": "true" if lp.new else "false", "TyresNotChanged": "0",
                    "TotalLaps": lp.start_age, "StartLaps": lp.start_age}}}}})
            elif kind == "pit_in":
                c.stint_laps += 1
                self._timing(t, c, {"InPit": True})
                self.emit(t, "TimingAppData", {"Lines": {c.num: {"Stints": {str(lp.stint): {"TotalLaps": c.stint_laps}}}}})
            elif kind.startswith("sector"):
                si = int(kind[-1])
                tl = lp.t1 - lp.t0
                fl = self._flags(lp.sec[si], si, c, lp.valid)
                spd = self._speed_at(sf[si], tl)
                self._timing(t, c, {"Sectors": {str(si): {"Value": "%.3f" % lp.sec[si], "PreviousValue": "", **fl}},
                                    "Speeds": {("I1", "I2")[si]: {"Value": str(spd), "Status": 0, "OverallFastest": False,
                                                                  "PersonalFastest": False}}})
            else:  # cross
                c.lapno += 1
                c.stint_laps += 1
                patch: Dict[str, Any] = {}
                self.emit(t, "TimingAppData", {"Lines": {c.num: {"Stints": {str(lp.stint): {"TotalLaps": c.stint_laps}}}}})
                rerank = False
                if lp.kind == "out":
                    patch["PitOut"] = False
                elif lp.kind in ("push", "long"):
                    tl = round(lp.t1 - lp.t0, 3)
                    fl = self._flags(lp.sec[2], 2, c, lp.valid)
                    ov = lp.valid and (self.ov_best is None or tl < self.ov_best - 1e-9)
                    pe = lp.valid and (c.best is None or tl < c.best - 1e-9)
                    patch["Sectors"] = {"2": {"Value": "%.3f" % lp.sec[2], "PreviousValue": "", **fl}}
                    patch["Speeds"] = {
                        "FL": {"Value": str(self._speed_at(0.0, tl)), "Status": 0, "OverallFastest": False, "PersonalFastest": False},
                        "ST": {"Value": str(int(round(max(self.track.v_ref) * (self.tref / tl) * 3.6 * 0.985))), "Status": 0,
                               "OverallFastest": False, "PersonalFastest": False}}
                    patch["LastLapTime"] = {"Value": fmt_lap(tl), "Status": 2051 if ov else (2049 if pe else 0),
                                            "OverallFastest": ov, "PersonalFastest": pe}
                    c.timed += 1
                    patch["NumberOfLaps"] = c.timed
                    if not lp.valid:
                        turn = 1 + (c.idx * 3 + c.lapno) % 14
                        self._rcm(t + 12.0, {"Category": "Other", "RacingNumber": c.num,
                                             "Message": "CAR %s (%s) TIME %s DELETED - TRACK LIMITS AT TURN %d LAP %d" % (
                                                 c.num, c.tla, fmt_lap(tl), turn, c.lapno)}, lap=c.lapno)
                    if ov:
                        self.ov_best = tl
                    if pe:
                        c.best, c.best_t, c.best_lap_no = tl, t, c.lapno
                        patch["BestLapTime"] = {"Value": fmt_lap(tl), "Lap": c.lapno}
                        if self.quali:
                            patch["BestLapTimes"] = {str(win.part - 1): {"Value": fmt_lap(tl), "Lap": c.lapno}}
                        rerank = True
                if patch:
                    self._timing(t, c, patch)
                if rerank:
                    order = self._rank(t, win, parts, 0)
            t_last = max(t_last, t) if kind == "cross" else t_last
        return t_last, order

    # ------------------------------------------------------------------ driver
    def run(self) -> SimRace:
        cfg = self.cfg
        self._prelude()
        r = self.race
        alive = list(self.cars)
        t = self.T0
        nparts = len(self.durs)
        t_prev_final = 0.0
        ranking: List[_SCar] = list(self.cars)
        fixed: List[_SCar] = []  # knocked-out cars, best (earliest-knocked-out last)
        end_all = t
        for k in range(nparts):
            win = self._make_window(k, t)
            self._plan_window(win, alive)
            # window start
            if k > 0:
                self._clock(t_prev_final + 1.0, win, False, win.dur)
            ts = win.ts
            self._session_status(ts, "Started")
            self._clock(ts, win, True)
            if self.quali:
                self.emit(ts, "TimingData", {"SessionPart": win.part})
                self.emit(ts, "SessionData", {"Series": {str(self.part_series_n): {
                    "Utc": format_ts(self.utc(ts)), "QualifyingPart": win.part}}})
                self.part_series_n += 1
                if k > 0:
                    for c in alive:
                        self._timing(ts, c, {"BestLapTime": {"Value": ""}, "TimeDiffToFastest": "",
                                             "TimeDiffToPositionAhead": "", "Cutoff": False})
            self._rcm(ts, {"Category": "Flag", "Flag": "GREEN", "Scope": "Track", "Message": "GREEN LIGHT - PIT EXIT OPEN"})
            tt = ts + 300.0
            while tt < win.we - 1.0:
                if not (win.r0 is not None and win.r0 <= tt <= win.r1):
                    self._clock(tt, win, True)
                tt += 300.0
            if win.r0 is not None:
                self._session_status(win.r0, "Aborted")
                self._track_status(win.r0, "5", "Red")
                self._rcm(win.r0, {"Category": "Flag", "Flag": "RED", "Scope": "Track", "Message": "RED FLAG"})
                self._clock(win.r0, win, False)
                self._track_status(win.r1, "1", "AllClear")
                self._session_status(win.r1, "Started")
                self._rcm(win.r1, {"Category": "Flag", "Flag": "GREEN", "Scope": "Track",
                                   "Message": "GREEN LIGHT - PIT EXIT OPEN"})
                self._clock(win.r1, win, True)
            self._rcm(win.we, {"Category": "Flag", "Flag": "CHEQUERED", "Scope": "Track", "Message": "CHEQUERED FLAG"})
            self._session_status(win.we, "Finished")
            self._clock(win.we, win, False, 0.0)
            # replay laps
            t_last, order = self._process(win, alive)
            t_final = t_last + 0.5
            laps_end = max([lp.t1 for laps in self._win_laps.values() for lp in laps] + [win.we])
            end_all = max(end_all, laps_end)
            knocked: List[str] = []
            if win.keep and k < nparts - 1:
                for c in order[win.keep:]:
                    c.out = True
                    knocked.append(c.num)
                    self._timing(t_final, c, {"KnockedOut": True, "Cutoff": True})
            self.parts_info.append({
                "part": win.part, "start_utc": self.utc(ts), "end_utc": self.utc(win.we), "final_utc": self.utc(t_final),
                "red": (self.utc(win.r0), self.utc(win.r1)) if win.r0 is not None else None,
                "participants": [c.num for c in alive], "knocked_out": knocked, "order": [c.num for c in order],
                "laps_end_utc": self.utc(laps_end)})
            if knocked:
                fixed = [c for c in order[win.keep:]] + fixed
                alive = order[: win.keep]
            else:
                alive = order
            ranking = alive + fixed
            t_prev_final = t_final
            if k < nparts - 1:
                t = max(win.we + self.spec["gaps"][k] * max(self.scale, 0.6), laps_end + 30.0)
        # end of session
        t_fin = end_all + 20.0
        self._session_status(t_fin, "Finalised")
        self._session_status(t_fin + 0.05, "Ends")
        # background chatter
        self._chatter(t_fin)
        # telemetry
        self._telemetry(t_fin + 0.1)
        r.final_order = [c.num for c in ranking]
        r.race_start_utc = self.base + self.T0
        r.info = {"session": cfg.session, "parts": self.parts_info}
        r._finalise()
        return r

    # ------------------------------------------------------------------ chatter / telemetry
    def _chatter(self, t_end: float) -> None:
        rng = random.Random("%s-chatter" % self.cfg.seed)
        for c in self.cars:
            c.starts = [lp.t0 for lp in c.laps]
        t = 5.0
        while t < t_end:
            self.emit(t, "Heartbeat", {"Utc": format_ts(self.utc(t))})
            t += 5.0
        w = self.weather
        t = 60.0
        while t < t_end:
            w["air"] += rng.gauss(-0.01, 0.12)
            w["track"] += rng.gauss(-0.02, 0.25)
            w["hum"] = min(95.0, max(10.0, w["hum"] + rng.gauss(0.0, 0.4)))
            w["pres"] += rng.gauss(0.0, 0.08)
            w["wind"] = min(12.0, max(0.1, w["wind"] + rng.gauss(0.0, 0.3)))
            w["dir"] = (w["dir"] + rng.gauss(0.0, 6.0)) % 360
            self.emit(t, "WeatherData", self._weather_data())
            t += 60.0
        t = self.T0 + rng.uniform(30, 90)
        while t < t_end - 30:
            on = [c for c in self.cars if c.state(t)[5] not in ("idle",)]
            if on:
                c = rng.choice(on)
                stamp = format_ts(self.utc(t))
                path = "TeamRadio/%s%s01_%s_%s_%s.mp3" % (c.first[:3].upper(), c.last[:3].upper(), c.num,
                                                          stamp[:10].replace("-", ""), stamp[11:19].replace(":", ""))
                cap = {"Utc": stamp, "RacingNumber": c.num, "Path": path}
                payload: Any = {"Captures": [cap]} if self.radio_count == 0 else {"Captures": {str(self.radio_count): cap}}
                self.radio_count += 1
                self.emit(t, "TeamRadio", payload)
            t += rng.uniform(45, 130)

    def _telemetry(self, t_end: float) -> None:
        r = self.race
        tr = self.track
        n = tr.n
        dt = self.dt
        curv = tr.curv
        a_lat = 27.0
        zones = tr.drs_zones
        prev = {c.num: c.state(0.0)[:2] for c in self.cars}
        pv = {c.num: 0.0 for c in self.cars}
        k = 0
        while True:
            k += 1
            t1 = k * dt
            if t1 > t_end:
                break
            cd = array("i")
            po = array("i")
            for c in self.cars:
                x, y, z, on, f, kind = c.state(t1)
                px, py = prev[c.num]
                v = math.hypot(x - px, y - py) / dt if kind != "idle" else 0.0
                if v > 100.0:  # teleport guard
                    v = pv[c.num]
                prev[c.num] = (x, y)
                vprev = pv[c.num]
                pv[c.num] = v
                kmh = int(v * 3.6 + 0.5)
                acc = (v - vprev) / dt
                if v < 1.0:
                    gear, rpm, thr, brk = (0, 0, 0, 0) if kind == "idle" else (1, 4000 + (c.idx * 7 % 5) * 20, 0, 0)
                else:
                    gear = 1
                    for th in _GEAR_HI[:-1]:
                        if kmh >= th:
                            gear += 1
                        else:
                            break
                    lo, hi = _GEAR_LO[gear - 1], _GEAR_HI[gear - 1]
                    rpm = int(9000 + (kmh - lo) / (hi - lo) * 3300) + ((int(t1 * 4) * 7 + c.idx * 3) % 9) * 8
                    if acc < -7.0:
                        brk, thr = 100, 0
                    elif acc < -1.0:
                        brk, thr = 0, 0
                    elif acc > 1.0:
                        brk, thr = 0, 100
                    else:
                        lat = min(1.0, v * v * curv[int(f * n) % n] / a_lat)
                        brk, thr = 0, int(100 - 55 * lat ** 1.5)
                drs = 0
                if kind in ("push", "long") and v > 20.0:
                    for za, zb in zones:
                        if (za <= f < zb) if za < zb else (f >= za or f < zb):
                            drs = 12
                            break
                cd.extend((rpm, kmh, gear, thr, brk, drs))
                po.extend((round(x * 10), round(y * 10), round(z * 10), on))
            r.cd_rows.append(cd)
            r.pos_rows.append(po)
            idx = len(r.cd_rows) - 1
            r._items.append((self.base + t1, 1, idx))
            r._items.append((self.base + t1, 2, idx))


def generate_session(cfg: SimConfig) -> SimRace:
    if cfg.session not in SPECS:
        raise ValueError("unknown session %r" % cfg.session)
    return _Sess(cfg).run()
