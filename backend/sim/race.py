"""Deterministic synthetic race generator.

``generate(SimConfig(seed=1, laps=20))`` runs a small physical race simulation (0.25 s fixed step) and
returns a :class:`SimRace` -- a time-ordered, *lazily materialised* list of :class:`Msg` in the exact raw
F1 topic shapes (TimingData, TimingAppData, CarData, Position, ...).  High-rate telemetry is stored as
compact ``array('i')`` rows and only turned into dicts on :meth:`SimRace.get`, so a 57-lap race costs a
few tens of MB, not hundreds.

Model overview
--------------
* every car integrates its own distance ``s`` (metres since the start line, negative on the grid);
* desired speed = circuit reference speed profile (from curvature) x a per-lap pace factor derived from
  team/driver pace, tyre compound + degradation (with a cliff), fuel burn and lap noise;
* a car following another is limited to ``v_ahead + (gap - gmin)/tau`` and may never overlap (4 m min
  separation); overtakes are explicit "attempts" (probability rises with pace advantage / DRS / lapped
  traffic) that either complete (position swap) or fail;
* pit stops go through a pit-lane state machine (80 km/h limit, stationary time, tyre change at exit);
* a retirement triggers yellow flags and then a Safety Car (3 leader laps, field bunches up, cars pit
  under it) or a VSC; plus minor yellow-flag incidents, penalties, team radio and weather drift;
* timing gaps/intervals are computed from loop-crossing times (20 loops/lap), as a real timing system
  does, so they are always consistent with the order on track.
"""
from __future__ import annotations

import calendar
import heapq
import math
import random
from array import array
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ..ingest.messages import Msg, format_ts
from .drivers import COMPOUNDS, DRIVERS, TEAMS
from .tracks import Track, get_track

LOOPS = 20  # timing loops per lap
SECTOR_LOOPS = (6, 13, 20)  # loop index (within the lap) at which sectors 1, 2, 3 end
PIT_ENTRY, PIT_BOX, PIT_EXIT = 0.965, 0.985, 0.03  # fractions of the lap
PIT_V = 80.0 / 3.6
A_UP, A_DN = 8.0, 40.0
FUEL_S_PER_LAP = 0.030
MIN_SEP = 4.0
GRID_SPACING = 8.0

Z_TOPICS = ("CarData", "Position")
_GEAR_LO = (0, 62, 100, 138, 176, 214, 252, 290)
_GEAR_HI = (62, 100, 138, 176, 214, 252, 290, 330)


def _default_base() -> float:
    return float(calendar.timegm((2025, 3, 16, 4, 0, 0)))


@dataclass
class SimConfig:
    seed: int = 1
    laps: int = 20
    track: str = "bahrain-like"
    base_utc: float = field(default_factory=_default_base)
    pre_race_s: float = 20.0  # seconds of grid time before lights out
    tick: float = 0.25  # simulation + telemetry step (4 Hz)
    incidents: bool = True  # retirements -> safety car/VSC, minor yellows
    cars: int = 20
    session: str = "race"  # race | sprint | fp | quali | sprint_quali
    snapshot_first: bool = False  # raw-line output starts with CompletionMessage-style full-state lines
    time_scale: float = 1.0  # fp / quali / sprint_quali: scales the session & part durations
    practice: int = 1  # fp: Practice 1/2/3


# --------------------------------------------------------------------------- formatting helpers
def fmt_lap(sec: float) -> str:
    m = int(sec // 60)
    s = sec - 60 * m
    return "%d:%06.3f" % (m, s) if m else "%.3f" % s


def fmt_gap(sec: float) -> str:
    return "+%.3f" % sec


def _hms(sec: float) -> str:
    sec = max(0, int(sec))
    return "%02d:%02d:%02d" % (sec // 3600, sec % 3600 // 60, sec % 60)


def _short_ts(utc: float) -> str:
    return format_ts(utc)[:19]  # RaceControlMessages use second precision, no 'Z'


# --------------------------------------------------------------------------- result container
class SimRace:
    """Time-ordered, lazily materialised message list (see module docstring)."""

    def __init__(self, cfg: SimConfig, track: Track, nums: List[str]) -> None:
        self.cfg = cfg
        self.track = track
        self.nums = nums
        self.events: List[Tuple[str, Any]] = []
        self.cd_rows: List[array] = []
        self.pos_rows: List[array] = []
        self._items: List[Tuple[float, int, int]] = []  # (utc, kind, index)
        self.times: List[float] = []
        self.topics: List[str] = []
        self._kinds: List[Tuple[int, int]] = []
        self.final_order: List[str] = []
        self.race_start_utc: float = 0.0
        self.end_utc: float = 0.0
        self.prelude_events: int = 0  # number of leading events that form the initial full state
        self.snapshot_count: int = 0  # leading messages (utc == base) making up that state
        self.info: Dict[str, Any] = {}  # non-race sessions: {"session":..., "parts":[...]}

    # building
    def _finalise(self) -> None:
        self._items.sort(key=lambda it: it[0])  # stable: ties keep generation order
        for utc, kind, idx in self._items:
            self.times.append(utc)
            self._kinds.append((kind, idx))
            self.topics.append(self.events[idx][0] if kind == 0 else Z_TOPICS[kind - 1])
        self._items = []
        self.end_utc = self.times[-1] if self.times else 0.0
        n = 0
        for kind, idx in self._kinds:
            if kind != 0 or idx >= self.prelude_events:
                break
            n += 1
        self.snapshot_count = n

    # access
    def __len__(self) -> int:
        return len(self.times)

    def utc(self, i: int) -> float:
        return self.times[i]

    def get(self, i: int) -> Msg:
        kind, idx = self._kinds[i]
        utc = self.times[i]
        if kind == 0:
            topic, data = self.events[idx]
            return Msg(topic, data, utc)
        ts = format_ts(utc)
        nums = self.nums
        if kind == 1:
            row = self.cd_rows[idx]
            cars = {}
            for j, num in enumerate(nums):
                o = j * 6
                cars[num] = {
                    "Channels": {
                        "0": row[o],
                        "2": row[o + 1],
                        "3": row[o + 2],
                        "4": row[o + 3],
                        "5": row[o + 4],
                        "45": row[o + 5],
                    }
                }
            return Msg("CarData", {"Entries": [{"Utc": ts, "Cars": cars}]}, utc)
        row = self.pos_rows[idx]
        ents = {}
        for j, num in enumerate(nums):
            o = j * 4
            ents[num] = {"Status": "OnTrack" if row[o + 3] else "OffTrack", "X": row[o], "Y": row[o + 1], "Z": row[o + 2]}
        return Msg("Position", {"Position": [{"Timestamp": ts, "Entries": ents}]}, utc)

    def __iter__(self) -> Iterator[Msg]:
        for i in range(len(self)):
            yield self.get(i)

    def messages(self) -> List[Msg]:
        return list(self)

    # snapshot (CompletionMessage-style) output
    def snapshot_messages(self) -> List[Msg]:
        """The initial full state: the leading ``snapshot_count`` messages merged to one per topic."""
        from ..ingest.messages import deep_merge

        merged: Dict[str, Any] = {}
        for i in range(self.snapshot_count):
            m = self.get(i)
            if m.topic in merged:
                deep_merge(merged[m.topic].data, m.data)
            else:
                merged[m.topic] = Msg(m.topic, deep_merge({}, m.data), m.utc)
        return list(merged.values())

    def snapshot_lines(self, encode: bool = True) -> List[str]:
        """``['Topic', '<json string>', '']`` lines, as ``fastf1``'s ``_on_message`` dumps a CompletionMessage
        (``.z`` topics carry the base64 string).  Parsed with ``parse_line`` they yield untimestamped ``Msg``."""
        import json

        from ..ingest.messages import Z_TOPICS as _Z, encode_z

        out = []
        for m in self.snapshot_messages():
            if encode and m.topic in _Z:
                out.append(str([m.topic + ".z", encode_z(m.data), ""]))
            else:
                out.append(str([m.topic, json.dumps(m.data, separators=(",", ":")), ""]))
        return out

    def raw_lines(self, encode: bool = True, snapshot_first: Optional[bool] = None) -> Iterator[str]:
        """Recording lines for the whole session.  With ``snapshot_first`` (default: ``cfg.snapshot_first``) the
        initial state is written as snapshot lines first, followed by every later message as a normal line."""
        from ..ingest.messages import to_raw_line

        sf = self.cfg.snapshot_first if snapshot_first is None else snapshot_first
        start = 0
        if sf:
            yield from self.snapshot_lines(encode)
            start = self.snapshot_count
        for i in range(start, len(self)):
            yield to_raw_line(self.get(i), encode)


# --------------------------------------------------------------------------- simulation internals
class _Car:
    __slots__ = (
        "num", "tla", "first", "last", "team", "color", "idx", "team_delta", "drv_delta", "grid",
        "s", "s_prev", "p", "v", "vprev", "ds", "nv", "lap_m", "laps_done", "xt", "pos",
        "compound", "start_age", "stint_laps", "stint_idx", "new", "plan", "plan_idx", "compounds", "stops_done",
        "phase", "pit_flag", "box_s", "exit_s", "timer", "pit_out", "sc_roll", "sc_decided",
        "retire_s", "retiring", "retired_cls", "stopped", "finished", "fin_t",
        "best_lap", "best_sec", "vmax_seg", "vmax_lap", "zone", "drs_open", "ahead", "ahead_d",
        "att_until", "att_ok", "cool_until", "spin_until", "tel_prev_v",
    )

    def __init__(self) -> None:
        self.s = 0.0
        self.s_prev = 0.0
        self.p = 0.0
        self.v = 0.0
        self.vprev = 0.0
        self.ds = 0.0
        self.nv = 0.0
        self.lap_m = 1.0
        self.laps_done = 0
        self.xt = []
        self.stint_laps = 0
        self.stint_idx = 0
        self.plan_idx = 0
        self.stops_done = 0
        self.phase = 0
        self.pit_flag = False
        self.box_s = 0.0
        self.exit_s = 0.0
        self.timer = 0.0
        self.pit_out = False
        self.sc_decided = -1
        self.retire_s = None
        self.retiring = False
        self.retired_cls = False
        self.stopped = False
        self.finished = False
        self.fin_t = 0.0
        self.best_lap = 1e9
        self.best_sec = [1e9, 1e9, 1e9]
        self.vmax_seg = 0.0
        self.vmax_lap = 0.0
        self.zone = -1
        self.drs_open = False
        self.ahead = None
        self.ahead_d = 0.0
        self.att_until = 0.0
        self.att_ok = False
        self.cool_until = 0.0
        self.spin_until = 0.0


class _Sim:
    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.track = get_track(cfg.track)
        self.L = self.track.length
        self.Ll = self.L / LOOPS
        self.sprint = cfg.session == "sprint"
        self.N = max(1, min(cfg.laps, 15) if self.sprint else cfg.laps)
        self.clock_total = 3600 if self.sprint else 7200
        self.T0 = cfg.pre_race_s
        self.dt = cfg.tick
        self.base = cfg.base_utc
        self.t_ref = self.track.lap_time_ref
        self.vref = self.track.v_ref
        self.step_m = self.track.step
        self.n_pts = self.track.n
        self.curv = self.track.curv
        self.cars: List[_Car] = []
        self._make_cars()
        self.race = SimRace(cfg, self.track, [c.num for c in self.cars])
        self.order: List[_Car] = list(self.cars)
        self._seq = 0
        self.pending: List[Tuple[float, int, Callable[[float], None]]] = []
        # race state
        self.state = "green"  # physical: green | sc | vsc
        self.track_code = "1"
        self.drs_enabled = False
        self.drs_enable_lap = 2
        self.lead_laps = 0
        self.sc_laps_seen = 0
        self.sc_in = False
        self.sc_id = 0
        self.flag_out = False
        self.flag_t = 0.0
        self.fin_count = 0
        self.end_t: Optional[float] = None
        self.overall_best_lap = 1e9
        self.overall_best_sec = [1e9, 1e9, 1e9]
        self.fastest_owner: Optional[_Car] = None
        self.incident_done = False
        self.incident_kind = "sc" if self.rng.random() < 0.55 else "vsc"
        self.rcm_count = 0
        self.radio_count = 0
        self.weather = {"air": 27.8, "track": 36.5, "hum": 41.0, "pres": 1012.4, "wind": 2.1, "dir": 128.0}
        self.cross: List[Tuple[float, _Car, int]] = []
        self.zones = self.track.drs_zones
        self._schedule_static()

    # ------------------------------------------------------------------ setup
    def _make_cars(self) -> None:
        rng = self.rng
        drivers = DRIVERS[: self.cfg.cars]
        raw = []
        for i, d in enumerate(drivers):
            c = _Car()
            c.idx = i
            c.num, c.tla, c.first, c.last, c.team = d
            c.color, c.team_delta = TEAMS[d.team]
            c.drv_delta = rng.gauss(0.0, 0.22)
            raw.append((c.team_delta + c.drv_delta + rng.gauss(0.0, 0.25), i, c))
        raw.sort(key=lambda r: (r[0], r[1]))
        for g, (_, _, c) in enumerate(raw, 1):
            c.grid = g
            c.pos = g
            c.s = -g * GRID_SPACING
            c.p = c.s % self.L
        self.cars = [r[2] for r in sorted(raw, key=lambda r: r[1])]  # dict/array order = DRIVERS order
        for c in self.cars:
            self._strategy(c)
            c.sc_roll = rng.random()
            self._lap_pace(c, 1)
        # retirements
        n = self.N
        if self.cfg.incidents and n >= 6:
            k = 1 if n < 12 else rng.choice([1, 1, 2, 2, 3])
            victims = rng.sample(self.cars, k)
            fr = sorted(rng.uniform(0.2, 0.7) for _ in range(k))
            fr[0] = rng.uniform(0.2, 0.45)
            for v, f in zip(victims, fr):
                v.retire_s = f * n * self.L + rng.uniform(0, 0.6) * self.L

    def _strategy(self, c: _Car) -> None:
        rng, n = self.rng, self.N
        if self.sprint:
            stops = 0
        elif n < 8:
            stops = 0
        elif n <= 30:
            stops = 1
        elif n <= 48:
            stops = rng.choice([1, 2, 2])
        else:
            stops = rng.choice([1, 2, 2, 2, 3])
        laps: List[int] = []
        for i in range(1, stops + 1):
            lap = round(n * i / (stops + 1) + rng.uniform(-0.08, 0.08) * n)
            lo = (laps[-1] + 4) if laps else 4
            lap = max(lap, lo)
            if lap > n - 3:
                break
            laps.append(lap)
        comps = ["SOFT", "MEDIUM", "HARD"]
        seq = [rng.choices(comps, [3, 4, 2])[0]]
        for _ in laps:
            seq.append(rng.choice([x for x in comps if x != seq[-1]]))
        if len(laps) and len(set(seq)) == 1:
            seq[-1] = rng.choice([x for x in comps if x != seq[0]])
        c.plan = laps
        c.compounds = seq
        c.compound = seq[0]
        c.start_age = 0
        c.new = True

    def _lap_pace(self, c: _Car, lap_no: int) -> None:
        comp_d, rate, life = COMPOUNDS[c.compound]
        age = c.start_age + c.stint_laps
        deg = rate * age + max(0, age - life) * 0.22
        fuel = FUEL_S_PER_LAP * (self.N - (lap_no - 1))
        delta = c.team_delta + c.drv_delta + comp_d + deg + fuel + self.rng.gauss(0.0, 0.10)
        c.lap_m = self.t_ref / (self.t_ref + delta)

    # ------------------------------------------------------------------ emission
    def utc(self, t: float) -> float:
        return self.base + t

    def emit(self, t: float, topic: str, data: Any) -> None:
        r = self.race
        r.events.append((topic, data))
        r._items.append((self.base + t, 0, len(r.events) - 1))

    def schedule(self, t: float, fn: Callable[[float], None]) -> None:
        self._seq += 1
        heapq.heappush(self.pending, (t, self._seq, fn))

    def _rcm(self, t: float, entry: Dict[str, Any]) -> None:
        entry.setdefault("Utc", _short_ts(self.utc(t)))
        entry.setdefault("Lap", min(self.N, self.lead_laps + 1))
        if self.rcm_count == 0:
            payload: Any = {"Messages": [entry]}
        else:
            payload = {"Messages": {str(self.rcm_count): entry}}
        self.rcm_count += 1
        self.emit(t, "RaceControlMessages", payload)

    def _track_status(self, t: float, code: str, msg: str) -> None:
        self.track_code = code
        self.emit(t, "TrackStatus", {"Status": code, "Message": msg})

    def _timing(self, t: float, c: _Car, patch: Dict[str, Any]) -> None:
        self.emit(t, "TimingData", {"Lines": {c.num: patch}})

    # ------------------------------------------------------------------ static schedule
    def _schedule_static(self) -> None:
        rng = self.rng
        dur = self.N * (self.t_ref + 1.5)
        T0 = self.T0
        # heartbeat / weather / clock
        self.schedule(5.0, self._heartbeat)
        self.schedule(60.0, self._weather)
        self.schedule(T0 + 300.0, self._clock)
        # team radio
        t = T0 + rng.uniform(15, 40)
        while t < T0 + dur:
            self.schedule(t, self._radio)
            t += rng.uniform(35, 85)
        # minor yellow incidents
        if self.cfg.incidents and self.N >= 5:
            for _ in range(1 + (self.N >= 25) + (self.N >= 45)):
                self.schedule(T0 + dur * rng.uniform(0.08, 0.9), self._minor_yellow)
        # penalties / track limits
        for _ in range(1 + (self.N >= 30)):
            self.schedule(T0 + dur * rng.uniform(0.2, 0.85), self._penalty)
        if self.N >= 4:
            self.schedule(T0 + dur * rng.uniform(0.1, 0.5), self._deleted_lap)

    def _heartbeat(self, t: float) -> None:
        self.emit(t, "Heartbeat", {"Utc": format_ts(self.utc(t))})
        self.schedule(t + 5.0, self._heartbeat)

    def _weather(self, t: float) -> None:
        w, rng = self.weather, self.rng
        w["air"] += rng.gauss(-0.01, 0.12)
        w["track"] += rng.gauss(-0.02, 0.25)
        w["hum"] = min(95.0, max(10.0, w["hum"] + rng.gauss(0.0, 0.4)))
        w["pres"] += rng.gauss(0.0, 0.08)
        w["wind"] = min(12.0, max(0.1, w["wind"] + rng.gauss(0.0, 0.3)))
        w["dir"] = (w["dir"] + rng.gauss(0.0, 6.0)) % 360
        self.emit(t, "WeatherData", self._weather_data())
        self.schedule(t + 60.0, self._weather)

    def _weather_data(self) -> Dict[str, str]:
        w = self.weather
        return {
            "AirTemp": "%.1f" % w["air"], "Humidity": "%.1f" % w["hum"], "Pressure": "%.1f" % w["pres"],
            "Rainfall": "0", "TrackTemp": "%.1f" % w["track"], "WindDirection": "%d" % round(w["dir"]),
            "WindSpeed": "%.1f" % w["wind"],
        }

    def _clock(self, t: float) -> None:
        if self.end_t is None and not self.flag_out:
            self.emit(t, "ExtrapolatedClock", {
                "Utc": format_ts(self.utc(t)), "Remaining": _hms(self.clock_total - (t - self.T0)), "Extrapolating": True})
            self.schedule(t + 300.0, self._clock)

    def _radio(self, t: float) -> None:
        live = [c for c in self.cars if not c.retired_cls]
        if not live:
            return
        c = self.rng.choice(live)
        ts = self.utc(t)
        stamp = format_ts(ts)
        path = "TeamRadio/%s%s01_%s_%s_%s.mp3" % (
            c.first[:3].upper(), c.last[:3].upper(), c.num, stamp[:10].replace("-", ""), stamp[11:19].replace(":", ""))
        cap = {"Utc": stamp, "RacingNumber": c.num, "Path": path}
        if self.radio_count == 0:
            payload: Any = {"Captures": [cap]}
        else:
            payload = {"Captures": {str(self.radio_count): cap}}
        self.radio_count += 1
        self.emit(t, "TeamRadio", payload)

    def _penalty(self, t: float) -> None:
        live = [c for c in self.cars if not c.retired_cls]
        if not live:
            return
        c = self.rng.choice(live)
        reason = self.rng.choice([
            "SPEEDING IN THE PIT LANE", "CAUSING A COLLISION", "LEAVING THE TRACK AND GAINING A LASTING ADVANTAGE",
            "FORCING ANOTHER DRIVER OFF THE TRACK", "UNSAFE RELEASE"])
        secs = self.rng.choice([5, 5, 10])
        self._rcm(t, {
            "Category": "Other", "Message": "FIA STEWARDS: %d SECOND TIME PENALTY FOR CAR %s (%s) - %s" % (
                secs, c.num, c.tla, reason), "RacingNumber": c.num})

    def _deleted_lap(self, t: float) -> None:
        live = [c for c in self.cars if not c.retired_cls and c.best_lap < 1e8]
        if not live:
            self.schedule(t + 30.0, self._deleted_lap)
            return
        c = self.rng.choice(live)
        turn = self.rng.randint(1, max(2, len(self.track.corners) or 12))
        self._rcm(t, {
            "Category": "Other", "RacingNumber": c.num,
            "Message": "CAR %s (%s) TIME %s DELETED - TRACK LIMITS AT TURN %d LAP %d" % (
                c.num, c.tla, fmt_lap(c.best_lap + 0.4), turn, max(1, c.laps_done))})

    def _sector_no(self, p: float) -> int:
        return 1 + int(p / self.L * LOOPS) % LOOPS

    def _minor_yellow(self, t: float) -> None:
        if self.state != "green" or self.track_code != "1" or self.end_t is not None or self.flag_out:
            return
        cands = [c for c in self.cars if c.phase == 0 and not c.retiring and not c.finished and c.s > 0]
        if not cands:
            return
        c = self.rng.choice(cands)
        c.spin_until = t + 4.0
        sec = self._sector_no(c.p)
        self._rcm(t, {"Category": "Flag", "Flag": "YELLOW", "Scope": "Sector", "Sector": sec,
                      "Message": "YELLOW IN TRACK SECTOR %d" % sec})
        self._rcm(t + 0.1, {"Category": "Other", "RacingNumber": c.num,
                            "Message": "CAR %s (%s) SPUN AT TURN %d" % (c.num, c.tla, 1 + sec // 2)})
        self._track_status(t, "2", "Yellow")

        def clear(t2: float, sec: int = sec) -> None:
            if self.track_code == "2" and self.state == "green":
                self._rcm(t2, {"Category": "Flag", "Flag": "CLEAR", "Scope": "Sector", "Sector": sec,
                               "Message": "CLEAR IN TRACK SECTOR %d" % sec})
                self._track_status(t2, "1", "AllClear")

        self.schedule(t + 22.0, clear)

    # ------------------------------------------------------------------ prelude
    def _prelude(self) -> None:
        cfg = self.cfg
        t = 0.0
        bt = self.base
        stamp = format_ts(bt)
        self.emit(t, "SessionInfo", {
            "Meeting": {"Key": 9999, "Name": "%s Grand Prix (SIM)" % self.track.name.split("-")[0].title(),
                        "OfficialName": "SIMULATED FORMULA 1 GRAND PRIX", "Location": "Synthia",
                        "Country": {"Key": 0, "Code": "SIM", "Name": "Simulation"},
                        "Circuit": {"Key": 0, "ShortName": self.track.name}},
            "ArchiveStatus": {"Status": "Generating"}, "Key": 99999, "Type": "Race", "Name": "Sprint" if self.sprint else "Race",
            "StartDate": stamp[:19], "EndDate": format_ts(bt + self.clock_total)[:19], "GmtOffset": "00:00:00",
            "Path": "2025/2025-03-16_Simulated_Grand_Prix/2025-03-16_%s/" % ("Sprint" if self.sprint else "Race"),
        })
        self.emit(t, "DriverList", {c.num: {
            "RacingNumber": c.num, "BroadcastName": "%s %s" % (c.first[0], c.last.upper()), "FullName": "%s %s" % (c.first, c.last.upper()),
            "Tla": c.tla, "Line": c.grid, "TeamName": c.team, "TeamColour": c.color,
            "FirstName": c.first, "LastName": c.last.upper(), "Reference": c.tla[:3] + "01", "CountryCode": "SIM"}
            for c in self.cars})
        self.emit(t, "SessionStatus", {"Status": "Inactive"})
        self.emit(t, "TrackStatus", {"Status": "1", "Message": "AllClear"})
        self.emit(t, "LapCount", {"CurrentLap": 1, "TotalLaps": self.N})
        self.emit(t, "WeatherData", self._weather_data())
        self.emit(t, "ExtrapolatedClock", {"Utc": stamp, "Remaining": _hms(self.clock_total), "Extrapolating": False})
        self.emit(t, "TimingAppData", {"Lines": {c.num: {
            "RacingNumber": c.num, "Line": c.grid, "GridPos": str(c.grid),
            "Stints": {"0": {"LapFlags": 0, "Compound": c.compound, "New": "true", "TyresNotChanged": "0",
                             "TotalLaps": 0, "StartLaps": 0}}} for c in self.cars}})
        self.emit(t, "TimingData", {"Lines": {c.num: {
            "Position": str(c.grid), "Line": c.grid, "GapToLeader": "", "IntervalToPositionAhead": {"Value": ""},
            "NumberOfLaps": 0, "NumberOfPitStops": 0, "InPit": False, "PitOut": False, "Retired": False,
            "Stopped": False, "KnockedOut": False, "Cutoff": False} for c in self.cars}})
        self.emit(t, "Heartbeat", {"Utc": stamp})
        self._rcm(0.0, {"Category": "Flag", "Flag": "GREEN", "Scope": "Track", "Message": "GREEN LIGHT - PIT EXIT OPEN", "Lap": 1})

    # ------------------------------------------------------------------ main loop
    def run(self) -> SimRace:
        self._prelude()
        self.race.prelude_events = len(self.race.events)
        dt = self.dt
        T0 = self.T0
        max_t = T0 + self.N * (self.t_ref + 25.0) + 600.0
        k = 0
        started = False
        while True:
            k += 1
            t0 = (k - 1) * dt
            t1 = k * dt
            self._run_pending(t1)
            if not started and t1 > T0:
                started = True
                self.emit(T0, "SessionStatus", {"Status": "Started"})
                self.emit(T0, "ExtrapolatedClock", {"Utc": format_ts(self.utc(T0)), "Remaining": _hms(self.clock_total), "Extrapolating": True})
                for c in self.cars:
                    c.vmax_seg = 0.0
            self._step(t0, t1, started)
            self._telemetry(t1)
            if self.end_t is None:
                if started and all(c.finished or c.retired_cls for c in self.cars):
                    self.end_t = t1 + 12.0
                elif t1 > max_t:  # safety net: should never trigger
                    for c in self.cars:
                        c.finished = True
                    self.end_t = t1 + 1.0
            elif t1 >= self.end_t:
                break
        end = self.end_t or 0.0
        self.emit(end, "SessionStatus", {"Status": "Finalised"})
        self.emit(end + 0.05, "SessionStatus", {"Status": "Ends"})
        r = self.race
        r.final_order = [c.num for c in self.order]
        r.race_start_utc = self.base + T0
        r._finalise()
        return r

    def _run_pending(self, t1: float) -> None:
        while self.pending and self.pending[0][0] <= t1:
            t, _, fn = heapq.heappop(self.pending)
            fn(t)

    # ------------------------------------------------------------------ physics step
    def _step(self, t0: float, t1: float, started: bool) -> None:
        dt = self.dt
        L, Ll = self.L, self.Ll
        vref, step_m, npts = self.vref, self.step_m, self.n_pts
        cars = self.cars
        state = self.state
        rng = self.rng
        self.cross = []
        if started:
            onroad = [c for c in cars if c.phase == 0 and not c.retiring and not c.finished]
            onroad.sort(key=lambda c: c.p)
            m = len(onroad)
            for c in cars:
                c.ahead = None
            for i in range(m):
                c = onroad[i]
                a = onroad[(i + 1) % m] if m > 1 else None
                if a is not None:
                    d = (a.p - c.p) % L
                    if d < 150.0:
                        c.ahead = a
                        c.ahead_d = d
            sc_leader = None
            if state == "sc":
                for c in self.order:
                    if c.phase == 0 and not c.retiring and not c.finished:
                        sc_leader = c
                        break
            self._sc_pit_decisions(t1)
            drs_ok = self.drs_enabled and state == "green" and self.track_code in ("1", "2")
            for c in cars:
                s_start = c.s
                a = c.ahead
                # --- overtake resolution / attempts
                if a is not None and state == "green" and not c.retiring:
                    if t1 < c.att_until and c.att_ok and c.ahead_d <= 7.5:
                        c.s += c.ahead_d + MIN_SEP + 0.5
                        c.v = max(c.v, a.v * 1.01)
                        c.att_until = 0.0
                        c.att_ok = False
                        c.cool_until = t1 + 10.0
                        a.cool_until = t1 + 6.0
                        a.att_until = 0.0
                        a.att_ok = False
                        c.ahead = None
                        a = None
                    elif t1 >= c.att_until and t1 >= c.cool_until and c.ahead_d < 1.2 * (12.0 + 0.45 * a.v):
                        adv = c.lap_m - a.lap_m
                        lapped = int(max(a.s, 0.0) // L) < int(max(c.s, 0.0) // L)
                        if adv > 0.002 or lapped or c.drs_open:
                            pr = 0.012 * (3.0 if c.drs_open else 1.0) * (8.0 if lapped else 1.0) * min(3.0, 1.0 + max(adv, 0.0) * 300.0)
                            if rng.random() < pr:
                                c.att_until = t1 + 4.0
                                c.att_ok = lapped or rng.random() < 0.6
                # --- desired speed
                v = c.v
                a_dn = A_DN
                if c.retiring:
                    v_des = 0.0
                    a_dn = 12.0
                else:
                    ref = vref[int((c.p + 0.5 * v * dt) / step_m) % npts]
                    if c.finished:
                        v_des = max(0.5 * ref, 19.4)
                    elif c.phase == 1:
                        v_des = min(PIT_V, math.sqrt(2.0 * 18.0 * max(c.box_s - c.s, 0.0)))
                    elif c.phase == 2:
                        v_des = 0.0
                    elif c.phase == 3:
                        v_des = PIT_V
                    else:
                        if state == "sc":
                            v_des = max(0.55 * ref, 19.4) if c is sc_leader else ref * c.lap_m
                        elif state == "vsc":
                            v_des = max(0.62 * ref, 19.4)
                        else:
                            v_des = ref * c.lap_m
                            if c.drs_open:
                                v_des *= 1.028
                        if t1 < c.spin_until:
                            v_des = max(v_des * 0.35, 18.0)
                        elif t1 < c.att_until and c.att_ok:
                            v_des *= 1.02
                        if a is not None and c.phase == 0 and not c.finished:
                            va = a.v
                            if state == "sc":
                                gmin = 9.0 + 0.12 * va
                            elif t1 < c.att_until:
                                gmin = 6.0
                            else:
                                gmin = 12.0 + 0.45 * va
                            vl = va + (c.ahead_d - gmin) / 1.2
                            if vl < 0.92 * va:
                                vl = 0.92 * va
                            if vl < 18.5:
                                vl = 18.5
                            if vl < v_des:
                                v_des = vl
                dv = v_des - v
                nv = v + min(dv, A_UP * dt) if dv > 0 else v - min(-dv, a_dn * dt)
                c.nv = nv
                c.ds = (c.s - s_start) + 0.5 * (v + nv) * dt  # includes a pass jump
                c.s_prev = s_start
            # --- separation constraint (front to back by classification)
            for c in self.order:
                a = c.ahead
                if a is not None and c.phase == 0 and not c.finished and not c.retiring:
                    gap_new = c.ahead_d + a.ds - c.ds
                    if gap_new < MIN_SEP and c.ahead_d > 0:
                        c.ds = max(0.0, c.ahead_d + a.ds - MIN_SEP)
                        c.nv = c.ds / dt
            # --- apply movement, detect crossings and state changes
            for c in cars:
                s0 = c.s_prev
                s1 = s0 + c.ds
                c.vprev = c.v
                c.v = c.nv
                c.s = s1
                c.p = s1 % L
                if c.v > c.vmax_seg:
                    c.vmax_seg = c.v
                if c.v > c.vmax_lap:
                    c.vmax_lap = c.v
                k0 = int(s0 // Ll)
                k1 = int(s1 // Ll)
                if k1 > k0:
                    span = s1 - s0
                    for kk in range(max(k0 + 1, 0), k1 + 1):
                        tc = t0 + (t1 - t0) * ((kk * Ll - s0) / span if span > 0 else 1.0)
                        c.xt.append(tc)
                        self.cross.append((tc, c, kk))
                self._pit_logic(c, s0, s1, t0, t1)
                if c.retiring:
                    if not c.stopped and c.v < 0.4:
                        c.v = 0.0
                        c.stopped = True
                        self._timing(t1, c, {"Stopped": True})
                        self._rcm(t1, {"Category": "Other", "RacingNumber": c.num,
                                       "Message": "CAR %s (%s) STOPPED ON TRACK" % (c.num, c.tla)})
                elif (c.retire_s is not None and c.phase == 0 and not c.finished and s1 >= c.retire_s):
                    c.retiring = True
                    c.retired_cls = True
                    c.pit_flag = False
                    self._timing(t1, c, {"Retired": True})
                    self._on_retire(c, t1)
                # DRS zone bookkeeping
                zone = -1
                fr = c.p / L
                for zi, (za, zb) in enumerate(self.zones):
                    if (za <= fr < zb) if za < zb else (fr >= za or fr < zb):
                        zone = zi
                        break
                if zone != c.zone:
                    if zone >= 0:
                        a = c.ahead
                        c.drs_open = bool(drs_ok and a is not None and c.phase == 0 and not c.finished
                                          and c.ahead_d / max(a.v, 10.0) < 1.0)
                    else:
                        c.drs_open = False
                    c.zone = zone
                if c.drs_open and not drs_ok:
                    c.drs_open = False
            self._rank(t1)
            for tc, c, kk in sorted(self.cross, key=lambda x: x[0]):
                self._on_cross(tc, c, kk)
            self._rank(t1)

    def _rank(self, t1: float) -> None:
        def key(c: _Car):
            if c.retired_cls:
                return (2, -c.s, 0.0)
            if c.finished:
                return (0, -c.laps_done, c.fin_t)
            return (1, -c.s, 0.0)

        self.order = sorted(self.cars, key=key)
        changed = {}
        for pos, c in enumerate(self.order, 1):
            if c.pos != pos:
                c.pos = pos
                changed[c.num] = {"Position": str(pos), "Line": pos}
        if changed:
            self.emit(t1, "TimingData", {"Lines": changed})

    def _on_retire(self, c: _Car, t: float) -> None:
        sec = self._sector_no(c.p)
        if self.state == "green":
            self._rcm(t, {"Category": "Flag", "Flag": "YELLOW", "Scope": "Sector", "Sector": sec,
                          "Message": "YELLOW IN TRACK SECTOR %d" % sec})
            if self.track_code == "1":
                self._track_status(t, "2", "Yellow")
        if not self.incident_done and self.cfg.incidents:
            self.incident_done = True
            if self.incident_kind == "sc":
                self.schedule(t + self.rng.uniform(8, 16), self._deploy_sc)
            else:
                self.schedule(t + self.rng.uniform(4, 9), self._deploy_vsc)
        else:
            def clear(t2: float, sec: int = sec) -> None:
                if self.state == "green":
                    self._rcm(t2, {"Category": "Flag", "Flag": "CLEAR", "Scope": "Sector", "Sector": sec,
                                   "Message": "CLEAR IN TRACK SECTOR %d" % sec})
                    if self.track_code == "2":
                        self._track_status(t2, "1", "AllClear")

            self.schedule(t + 25.0, clear)

    # ------------------------------------------------------------------ safety car / vsc
    def _deploy_sc(self, t: float) -> None:
        if self.flag_out or self.state != "green":
            return
        self.state = "sc"
        self.sc_id += 1
        self.sc_laps_seen = 0
        self.sc_in = False
        self._track_status(t, "4", "SCDeployed")
        self._rcm(t, {"Category": "SafetyCar", "Status": "DEPLOYED", "Mode": "SAFETY CAR", "Message": "SAFETY CAR DEPLOYED"})
        self._drs_disable(t)

    def _deploy_vsc(self, t: float) -> None:
        if self.flag_out or self.state != "green":
            return
        self.state = "vsc"
        self.sc_id += 1
        self._track_status(t, "6", "VSCDeployed")
        self._rcm(t, {"Category": "SafetyCar", "Status": "DEPLOYED", "Mode": "VIRTUAL SAFETY CAR",
                      "Message": "VIRTUAL SAFETY CAR DEPLOYED"})
        self._drs_disable(t)

        def ending(t2: float) -> None:
            self._track_status(t2, "7", "VSCEnding")
            self._rcm(t2, {"Category": "SafetyCar", "Status": "ENDING", "Mode": "VIRTUAL SAFETY CAR",
                           "Message": "VIRTUAL SAFETY CAR ENDING"})
            self.schedule(t2 + 9.0, self._go_green)

        self.schedule(t + self.rng.uniform(55, 80), ending)

    def _go_green(self, t: float) -> None:
        self.state = "green"
        self._track_status(t, "1", "AllClear")
        self._rcm(t, {"Category": "Flag", "Flag": "GREEN", "Scope": "Track", "Message": "TRACK CLEAR"})
        self.drs_enable_lap = self.lead_laps + 2

    def _drs_disable(self, t: float) -> None:
        if self.drs_enabled:
            self.drs_enabled = False
            self._rcm(t, {"Category": "Drs", "Status": "DISABLED", "Message": "DRS DISABLED"})
        for c in self.cars:
            c.drs_open = False

    def _sc_pit_decisions(self, t1: float) -> None:
        if self.state == "green" or self.flag_out:
            return
        L = self.L
        for c in self.cars:
            if c.sc_decided == self.sc_id or c.phase != 0 or c.retiring or c.finished or c.pit_flag:
                continue
            if c.p / L > 0.94 or c.s < 0:
                continue
            c.sc_decided = self.sc_id
            remaining = self.N - c.laps_done
            if (c.sc_roll < 0.65 and c.stint_laps >= 5 and remaining >= 5
                    and (c.plan_idx < len(c.plan) or c.stint_laps >= 10)):
                c.pit_flag = True
                if c.plan_idx < len(c.plan):
                    c.plan_idx += 1
                else:
                    c.compounds = c.compounds + [self.rng.choice([x for x in COMPOUNDS if x != c.compound])]

    # ------------------------------------------------------------------ pit lane
    def _pit_logic(self, c: _Car, s0: float, s1: float, t0: float, t1: float) -> None:
        L = self.L
        if c.phase == 0:
            if c.pit_flag and not c.retiring and not c.finished and s0 >= 0:
                lap_i = int(s1 // L)
                e = lap_i * L + PIT_ENTRY * L
                if s0 < e <= s1:
                    c.phase = 1
                    c.pit_flag = False
                    c.box_s = lap_i * L + PIT_BOX * L
                    c.exit_s = (lap_i + 1) * L + PIT_EXIT * L
                    tc = t0 + (t1 - t0) * (e - s0) / max(s1 - s0, 1e-9)
                    self._timing(tc, c, {"InPit": True})
        elif c.phase == 1:
            if c.s >= c.box_s - 0.5 or (c.v < 0.3 and c.box_s - c.s < 3.0):
                c.s = c.box_s
                c.p = c.s % L
                c.v = 0.0
                c.phase = 2
                c.timer = 2.0 + abs(self.rng.gauss(0.4, 0.5)) + (3.0 if self.rng.random() < 0.04 else 0.0)
        elif c.phase == 2:
            c.timer -= self.dt
            if c.timer <= 0:
                c.phase = 3
        elif c.phase == 3:
            if c.s >= c.exit_s:
                if self._exit_clear(c):
                    self._pit_exit(c, t1)
                else:  # wait at the pit exit (red light) until the track is free
                    c.s = c.exit_s
                    c.p = c.s % L
                    c.v = 0.0

    def _exit_clear(self, c: _Car) -> bool:
        L = self.L
        px = c.exit_s % L
        for o in self.cars:
            if o is c or o.phase != 0 or o.retiring or o.finished or o.s < 0:
                continue
            ahead = (o.p - px) % L
            behind = (px - o.p) % L
            if ahead < 15.0 or behind < 25.0 + 0.9 * o.v:
                return False
        return True

    def _pit_exit(self, c: _Car, t: float) -> None:
        c.phase = 0
        c.pit_out = True
        if c.v < PIT_V:  # released from the exit light: leave at pit-lane speed
            c.v = c.vprev = PIT_V
        c.stops_done += 1
        c.stint_idx += 1
        nxt = c.compounds[c.stops_done] if c.stops_done < len(c.compounds) else self.rng.choice(list(COMPOUNDS))
        c.compound = nxt
        c.new = self.rng.random() < 0.85
        c.start_age = 0 if c.new else self.rng.randint(2, 6)
        c.stint_laps = 0
        self._lap_pace(c, max(1, c.laps_done + 1))
        self._timing(t, c, {"InPit": False, "PitOut": True, "NumberOfPitStops": c.stops_done})
        self.emit(t, "TimingAppData", {"Lines": {c.num: {"Stints": {str(c.stint_idx): {
            "LapFlags": 0, "Compound": c.compound, "New": "true" if c.new else "false", "TyresNotChanged": "0",
            "TotalLaps": c.start_age, "StartLaps": c.start_age}}}}})

    # ------------------------------------------------------------------ timing events
    def _flags(self, c: _Car, idx: int, value: float, personal: List[float], overall: List[float]) -> Dict[str, Any]:
        ov = value < overall[idx] - 1e-9
        pe = value < personal[idx] - 1e-9
        if ov:
            overall[idx] = value
        if pe:
            personal[idx] = value
        return {"OverallFastest": ov, "PersonalFastest": pe, "Status": 2051 if ov else (2049 if pe else 2048)}

    def _gap_text(self, c: _Car, z: _Car, k: int) -> str:
        d = z.s - c.s
        if d >= self.L:
            return "%dL" % int(d // self.L)
        if c.finished and z.finished and z.laps_done > c.laps_done:
            return "%dL" % (z.laps_done - c.laps_done)
        if k < len(z.xt) and k < len(c.xt):
            return fmt_gap(max(0.0, c.xt[k] - z.xt[k]))
        return fmt_gap(max(0.0, d / max(c.v, 20.0)))

    def _on_cross(self, tc: float, c: _Car, k: int) -> None:
        if c.finished:  # the timing screen freezes once a car has taken the flag
            return
        loop = k % LOOPS
        lapno = k // LOOPS
        patch: Dict[str, Any] = {}
        is_line = loop == 0 and k > 0
        # gaps (skip for retired cars)
        if not c.retired_cls:
            leader = self.order[0]
            if c is leader:
                patch["GapToLeader"] = ""
                patch["IntervalToPositionAhead"] = {"Value": "", "Catching": False}
            else:
                ahead = self.order[c.pos - 2] if c.pos >= 2 else leader
                patch["GapToLeader"] = self._gap_text(c, leader, k)
                patch["IntervalToPositionAhead"] = {"Value": self._gap_text(c, ahead, k), "Catching": False}
        if c.pit_out and loop >= 6:
            c.pit_out = False
            patch["PitOut"] = False
        base = k - LOOPS if is_line else k - loop  # global loop index at the start of this lap
        b1 = base + SECTOR_LOOPS[0]
        b2 = base + SECTOR_LOOPS[1]
        start_t = self.T0 if base == 0 else c.xt[base]
        if loop == SECTOR_LOOPS[0] or loop == SECTOR_LOOPS[1] or is_line:
            if loop == SECTOR_LOOPS[0]:
                si, val = 0, c.xt[k] - start_t
                spd_key = "I1"
            elif loop == SECTOR_LOOPS[1]:
                si, val = 1, c.xt[k] - c.xt[b1]
                spd_key = "I2"
            else:
                si, val = 2, c.xt[k] - c.xt[b2]
                spd_key = "FL"
            val = round(val, 3)
            fl = self._flags(c, si, val, c.best_sec, self.overall_best_sec)
            sec = {"Value": "%.3f" % val, "PreviousValue": "", **fl}
            patch["Sectors"] = {str(si): sec}
            kmh = int(round(c.vmax_seg * 3.6)) if spd_key != "FL" else int(round(c.v * 3.6))
            patch["Speeds"] = {spd_key: {"Value": str(kmh), "Status": 0, "OverallFastest": False, "PersonalFastest": False}}
            c.vmax_seg = 0.0
            if is_line:
                patch["Speeds"]["ST"] = {"Value": str(int(round(c.vmax_lap * 3.6))), "Status": 0,
                                         "OverallFastest": False, "PersonalFastest": False}
                c.vmax_lap = 0.0
        if is_line:
            lt = round(tc - start_t, 3)
            ov = lt < self.overall_best_lap - 1e-9
            pe = lt < c.best_lap - 1e-9
            if ov:
                self.overall_best_lap = lt
                self.fastest_owner = c
            if pe:
                c.best_lap = lt
                patch["BestLapTime"] = {"Value": fmt_lap(lt), "Lap": lapno}
            patch["LastLapTime"] = {"Value": fmt_lap(lt), "Status": 2051 if ov else (2049 if pe else 0),
                                    "OverallFastest": ov, "PersonalFastest": pe}
            c.laps_done = lapno
            patch["NumberOfLaps"] = lapno
            c.stint_laps += 1
            self.emit(tc, "TimingAppData", {"Lines": {c.num: {"Stints": {str(c.stint_idx): {
                "TotalLaps": c.start_age + c.stint_laps}}}}})
            if ov and lapno >= 2:
                self._rcm(tc, {"Category": "Other", "RacingNumber": c.num,
                               "Message": "FASTEST LAP %s (%s)" % (fmt_lap(lt), c.tla)})
            # next lap setup
            if lapno < self.N:
                self._lap_pace(c, lapno + 1)
                if c.plan_idx < len(c.plan) and lapno + 1 == c.plan[c.plan_idx] and not c.pit_flag:
                    c.pit_flag = True
                    c.plan_idx += 1
            # leader of the lap?
            if lapno > self.lead_laps:
                self.lead_laps = lapno
                self._on_leader_lap(tc, c, lapno)
            # chequered flag
            if not c.finished and not c.retired_cls and (lapno >= self.N or (self.flag_out and tc >= self.flag_t)):
                c.finished = True
                c.fin_t = tc
                self.fin_count += 1
                if not self.flag_out:
                    self.flag_out = True
                    self.flag_t = tc
                    self._rcm(tc, {"Category": "Flag", "Flag": "CHEQUERED", "Scope": "Track", "Message": "CHEQUERED FLAG"})
                    self.emit(tc, "SessionStatus", {"Status": "Finished"})
                    self.emit(tc, "ExtrapolatedClock", {"Utc": format_ts(self.utc(tc)),
                                                       "Remaining": _hms(self.clock_total - (tc - self.T0)), "Extrapolating": False})
        if patch:
            self._timing(tc, c, patch)

    def _on_leader_lap(self, tc: float, c: _Car, lapno: int) -> None:
        if lapno < self.N:
            self.emit(tc, "LapCount", {"CurrentLap": lapno + 1, "TotalLaps": self.N})
        if self.state == "sc":
            self.sc_laps_seen += 1
            if self.sc_in:
                self._go_green(tc)
            elif self.sc_laps_seen >= 3:
                self.sc_in = True
                self._rcm(tc, {"Category": "SafetyCar", "Status": "IN THIS LAP", "Mode": "SAFETY CAR",
                               "Message": "SAFETY CAR IN THIS LAP"})
        if (not self.drs_enabled and self.state == "green" and lapno >= self.drs_enable_lap
                and self.track_code in ("1", "2") and lapno < self.N):
            self.drs_enabled = True
            self._rcm(tc, {"Category": "Drs", "Status": "ENABLED", "Message": "DRS ENABLED"})

    # ------------------------------------------------------------------ telemetry
    def _telemetry(self, t1: float) -> None:
        r = self.race
        tr = self.track
        L = self.L
        dt = self.dt
        curv, step_m, npts = self.curv, self.step_m, self.n_pts
        cd = array("i")
        po = array("i")
        a_lat = 27.0
        for c in self.cars:
            v = c.v
            kmh = int(v * 3.6 + 0.5)
            acc = (v - c.vprev) / dt
            if v < 1.0:
                gear, rpm, thr, brk = (1 if c.phase != 2 and not c.retired_cls else 0), 4000 + (c.idx * 7 % 5) * 20, 0, 0
            else:
                gear = 1
                for th in _GEAR_HI[:-1]:
                    if kmh >= th:
                        gear += 1
                    else:
                        break
                lo = _GEAR_LO[gear - 1]
                hi = _GEAR_HI[gear - 1]
                rpm = int(9000 + (kmh - lo) / (hi - lo) * 3300) + ((int(t1 * 4) * 7 + c.idx * 3) % 9) * 8
                if acc < -7.0:
                    brk, thr = 100, 0
                elif acc < -1.0:
                    brk, thr = 0, 0
                elif acc > 1.0:
                    brk, thr = 0, 100
                else:
                    lat = min(1.0, v * v * curv[int(c.p / step_m) % npts] / a_lat)
                    brk, thr = 0, int(100 - 55 * lat ** 1.5)
            if c.drs_open:
                drs = 12
            elif (self.drs_enabled and c.ahead is not None and c.phase == 0 and c.ahead_d / max(c.ahead.v, 10.0) < 1.0):
                drs = 8
            else:
                drs = 0
            cd.extend((rpm, kmh, gear, thr, brk, drs))
            x, y, z = tr.point_at(c.p)
            po.extend((round(x * 10), round(y * 10), round(z * 10), 0 if (c.stopped and c.retired_cls) else 1))
        r.cd_rows.append(cd)
        r.pos_rows.append(po)
        n = len(r.cd_rows) - 1
        r._items.append((self.base + t1, 1, n))
        r._items.append((self.base + t1, 2, n))


def generate(cfg: Optional[SimConfig] = None, **kw: Any) -> SimRace:
    """Generate a full race. ``generate(seed=3, laps=10)`` is shorthand for ``generate(SimConfig(...))``."""
    if cfg is None:
        cfg = SimConfig(**kw)
    if cfg.session in ("fp", "quali", "sprint_quali"):
        from .sessions import generate_session

        return generate_session(cfg)
    if cfg.session not in ("race", "sprint"):
        raise ValueError("unknown session %r (race|sprint|fp|quali|sprint_quali)" % cfg.session)
    return _Sim(cfg).run()
