"""RaceEngine: consumes raw F1 live-timing `Msg` patches and derives dashboard state.

Public API (see docs/CONTRACT.md):
    engine.apply(msg)          -- never raises
    engine.reset()             -- forget everything (track geometry is kept)
    engine.state()  -> dict    -- {"type":"state", ...}
    engine.fast_delta() -> dict|None   -- {"type":"fast", ...} cars changed since last call
    engine.history() -> dict   -- {"type":"history", ...}
    engine.track_dict() -> dict|None
    engine.rev / engine.history_rev / engine.track_version  -- change counters
    engine.stats() -> dict
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from typing import Any, Dict, List, Optional

from ..track import TrackBuilder
from .parsing import (as_list, clean_text, deep_merge, indexed_items, is_drs_open, norm_brake,
                      parse_clock, parse_gap, parse_iso, parse_laptime, parse_time, to_bool,
                      to_float, to_int)

log = logging.getLogger(__name__)

STATIC_URL = "https://livetiming.formula1.com/static/"
TRACK_LABELS = {"1": "AllClear", "2": "Yellow", "3": "Yellow", "4": "SCDeployed", "5": "Red",
                "6": "VSCDeployed", "7": "VSCEnding"}
TRACK_FLAGS = {"1": "green", "2": "yellow", "3": "yellow", "4": "sc", "5": "red", "6": "vsc", "7": "vsc"}
TRACK_SEVERITY = {"1": 0, "2": 1, "3": 1, "7": 2, "6": 3, "4": 4, "5": 5}
MAX_EVENTS = 500
MAX_RC = 100
MAX_RADIO = 30
MAX_ERR_LOGS = 20
MAX_RECS = 400          # lap records kept per driver
MAX_STOPS = 400
MAX_SEEN = 3000         # dedupe-set size before it is pruned
MAX_TOPICS = 64
GARAGE_AFTER = 60.0     # seconds sitting in the pit lane before a car counts as "in the garage"
NONRACE_KINDS = ("practice", "qualifying", "sprint_qualifying")

_re_car = re.compile(r"CAR\s+(\d+)", re.I)
_re_secs = re.compile(r"(\d+)\s*(?:-\s*)?SECOND", re.I)
_re_num = re.compile(r"^\d{1,3}$")


def session_kind(typ: Any, name: Any) -> str:
    """Derive session.kind from SessionInfo Type + Name (case-insensitive, lenient)."""
    t = str(typ or "").strip().lower()
    n = str(name or "").strip().lower()
    both = t + " " + n
    if "shootout" in both or ("sprint" in both and ("quali" in both or "shoot" in both)):
        return "sprint_qualifying"
    if "sprint" in both:
        return "sprint"
    if "quali" in both:
        return "qualifying"
    if "practice" in both or re.match(r"^(fp|p)\s*[123]\b", n):
        return "practice"
    if "race" in both or "grand prix" in both:
        return "race"
    return "unknown"


class Stint:
    __slots__ = ("idx", "compound", "new", "start_laps", "total_laps", "start_count")

    def __init__(self, idx: int, start_count: int):
        self.idx = idx
        self.compound = None
        self.new = None
        self.start_laps = 0
        self.total_laps = None
        self.start_count = start_count


class Drv:
    def __init__(self, num: str):
        self.num = num
        self.tla = num
        self.name = ""
        self.team = ""
        self.color = "#888888"
        self.line = 999
        # timing
        self.pos: Optional[int] = None
        self.first_pos: Optional[int] = None
        self.grid: Optional[int] = None
        self.laps = 0
        self.laps_seen = False
        self.pits = 0
        self.in_pit = False
        self.pit_out = False
        self.retired = False
        self.stopped = False
        self.gap_text = ""
        self.gap_sec: Optional[float] = None
        self.int_text = ""
        self.int_sec: Optional[float] = None
        # non-race (TimeDiffToFastest / TimeDiffToPositionAhead)
        self.tdf_text = ""
        self.tdf_sec: Optional[float] = None
        self.tdf_seen = False
        self.tda_text = ""
        self.tda_sec: Optional[float] = None
        self.tda_seen = False
        self.best_lap_no: Optional[int] = None
        self.best_laps: List[Optional[Dict[str, Any]]] = [None, None, None]
        self.knocked_out = False
        self.cutoff = False
        self.in_pit_since: Optional[float] = None
        self.xy_ts = ""
        self.tel_ts = ""
        self.last_text = ""
        self.last_sec: Optional[float] = None
        self.last_overall = False
        self.last_personal = False
        self.best_text = ""
        self.best_sec: Optional[float] = None
        self.best_overall: Optional[bool] = None
        self.best_personal: Optional[bool] = None
        self.sectors: List[Dict[str, Any]] = [{}, {}, {}]
        self.traps: Dict[str, int] = {}
        self.sec_by_lap: Dict[int, Dict[int, float]] = {}
        self.last_updated = False
        self.pending: Optional[dict] = None
        # laps
        self.recs: Dict[int, dict] = {}
        self.pit_flag = False
        self.worst_track = "1"
        # pits
        self.pit_in_utc: Optional[float] = None
        self.pit_in_lap = 0
        self.stops: List[dict] = []
        # tyres
        self.stints: Dict[int, Stint] = {}
        # telemetry / position
        self.tel: Dict[str, Any] = {}
        self.xy: Optional[Dict[str, Any]] = None
        self.progress: Optional[float] = None
        self.hint: Optional[int] = None
        self.max_speed = 0

    def cur_stint(self) -> Optional[Stint]:
        if not self.stints:
            return None
        return self.stints[max(self.stints)]

    def stint_age(self, st: Stint, laps: Optional[int] = None) -> int:
        n = self.laps if laps is None else laps
        age = st.start_laps + max(0, n - st.start_count)
        if laps is None and st.total_laps is not None and st.total_laps > age:
            age = st.total_laps
        return age


def _clean_compound(c: Any) -> Optional[str]:
    if not c:
        return None
    return str(c).strip().upper() or None


class RaceEngine:
    def __init__(self, track_builder: Optional[TrackBuilder] = None):
        self.tb = track_builder or TrackBuilder()
        self._err_logs = 0
        self._handlers = {
            "DriverList": self._h_driverlist,
            "TimingData": self._h_timing,
            "TimingAppData": self._h_timingapp,
            "CarData": self._h_cardata,
            "Position": self._h_position,
            "RaceControlMessages": self._h_rc,
            "TrackStatus": self._h_trackstatus,
            "WeatherData": self._h_weather,
            "LapCount": self._h_lapcount,
            "SessionStatus": self._h_sessionstatus,
            "SessionInfo": self._h_sessioninfo,
            "ExtrapolatedClock": self._h_clock,
            "TeamRadio": self._h_radio,
        }
        self.reset(keep_track=False)

    # ------------------------------------------------------------------ reset
    def reset(self, keep_track: bool = True) -> None:
        self.drivers: Dict[str, Drv] = {}
        self.raw: Dict[str, Any] = {}
        self.utc = 0.0
        self.rev = 0
        self.history_rev = 0
        self._hist_cache: Optional[dict] = None
        self._hist_cache_rev = -1
        self.session: Dict[str, Any] = {"name": None, "meeting": None, "type": None, "status": "Inactive"}
        self.clock: Dict[str, Any] = {"remaining": None, "utc": None, "extrapolating": False}
        self.started_utc: Optional[float] = None
        self.ended_utc: Optional[float] = None
        self.lap_current: Optional[int] = None
        self.lap_total: Optional[int] = None
        self.track_code = "1"
        self.track_msg = ""
        self.weather: Dict[str, Any] = {}
        self.rc: List[dict] = []
        self._rc_seen: set = set()
        self.radio: List[dict] = []
        self._radio_seen: set = set()
        self.events: List[dict] = []
        self.pitstops: List[dict] = []
        self.fastest: Optional[dict] = None
        self.top_speed: Optional[dict] = None
        self.dirty: set = set()
        self.counts = {"msgs": 0, "errors": 0, "laps": 0, "events": 0, "ignored": 0}
        self.topic_counts: Dict[str, int] = {}
        self._overtake_seen: set = set()
        self._kind_c = "unknown"
        self.session_part: Optional[int] = None
        self.cutoff_time: Optional[str] = None
        self._saw_timediff = False
        self._p1: Optional[str] = None
        self._ot_queue: list = []
        self._snapshot = False
        if not keep_track:
            self.tb = TrackBuilder(self.tb.resample if self.tb else 800)
        self._sent_track_version = -1

    # ------------------------------------------------------------------ apply
    def apply(self, msg: Any) -> None:
        try:
            topic = msg.topic
            if topic.endswith(".z"):
                topic = topic[:-2]
            data = msg.data
            utc = to_float(msg.utc)
            self._snapshot = not utc or utc <= 0
            if utc is not None and utc > self.utc:
                self.utc = utc
            self.counts["msgs"] += 1
            tc = self.topic_counts
            if topic in tc:
                tc[topic] += 1
            elif len(tc) < MAX_TOPICS:
                tc[topic] = 1
            h = self._handlers.get(topic)
            if h is None:
                self.counts["ignored"] += 1
                return
            if isinstance(data, (str, bytes)):
                data = self._decode_str(data)
            if not isinstance(data, (dict, list)):
                self.counts["ignored"] += 1
                return
            h(data)
        except Exception as e:  # never raise
            self.counts["errors"] += 1
            if self._err_logs < MAX_ERR_LOGS:
                self._err_logs += 1
                log.warning("engine: bad message on %r skipped: %r", getattr(msg, "topic", None), e, exc_info=self._err_logs <= 3)

    @staticmethod
    def _decode_str(data):
        """Snapshot payloads can arrive as JSON text (or base64+deflate for .z topics)."""
        try:
            if isinstance(data, bytes):
                data = data.decode("utf-8", "replace")
            t = data.lstrip()
            if t[:1] in ("{", "["):
                return json.loads(t)
            if len(t) > 8:
                from ..ingest.messages import decode_z
                return decode_z(t)
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------ helpers
    @property
    def track_version(self) -> int:
        return self.tb.version

    def _drv(self, num: Any) -> Drv:
        num = str(num)
        d = self.drivers.get(num)
        if d is None:
            d = self.drivers[num] = Drv(num)
            self.rev += 1
        return d

    def _kind(self) -> str:
        return self._kind_c

    def _nonrace(self) -> bool:
        k = self._kind()
        if k == "unknown":
            return self._saw_timediff and self.lap_total is None and not self.lap_current
        return k in NONRACE_KINDS

    def _red(self) -> bool:
        return self.track_code == "5" or self.session.get("status") == "Aborted"

    def _cur_lap(self) -> int:
        if self._nonrace():
            return 0
        if self.lap_current:
            return int(self.lap_current)
        mx = max((d.laps for d in self.drivers.values()), default=0)
        return mx + 1 if mx or self.session["status"] == "Started" else 0

    def _event(self, kind: str, text: str, num: Optional[str] = None, utc: Optional[float] = None,
               lap: Optional[int] = None) -> None:
        text = str(text)
        ev = {"utc": utc if utc is not None else self.utc, "lap": lap if lap is not None else self._cur_lap(),
              "kind": kind, "num": num, "text": text}
        self.events.append(ev)
        if len(self.events) > MAX_EVENTS:
            del self.events[: len(self.events) - MAX_EVENTS]
        self.counts["events"] += 1
        self.history_rev += 1

    def _recent_event(self, kind: str, within: float = 5.0, contains: Optional[str] = None) -> bool:
        for ev in reversed(self.events[-8:]):
            if ev["kind"] == kind and abs(self.utc - ev["utc"]) <= within and (contains is None or contains in ev["text"].upper()):
                return True
        return False

    # ------------------------------------------------------------------ handlers
    def _h_driverlist(self, data):
        if not isinstance(data, dict):
            return
        for num, info in data.items():
            if not isinstance(info, dict) or not str(num).isdigit():
                continue
            d = self._drv(num)
            if "Tla" in info and info["Tla"]:
                d.tla = str(info["Tla"])
            fn, ln = info.get("FirstName"), info.get("LastName")
            if fn or ln:
                d.name = ("%s %s" % (fn or "", ln or "")).strip()
            elif info.get("FullName"):
                d.name = str(info["FullName"])
            if info.get("TeamName"):
                d.team = str(info["TeamName"])
            if info.get("TeamColour"):
                c = str(info["TeamColour"]).lstrip("#")
                d.color = "#" + c
            if "Line" in info:
                l = to_int(info["Line"])
                if l is not None:
                    d.line = l
        self.rev += 1

    def _h_timing(self, data):
        if not isinstance(data, dict):
            return
        if "SessionPart" in data:
            part = to_int(data["SessionPart"])
            part = part if part in (1, 2, 3) else None
            if part != self.session_part:
                self.session_part = part
                self._p1 = None
                if part is not None and self._kind() in ("qualifying", "sprint_qualifying") and not self._snapshot:
                    self._event("flag", "Q%d" % part)
                self.rev += 1
        if "CutOffTime" in data:
            self.cutoff_time = clean_text(data["CutOffTime"]) or None
        lines = data.get("Lines")
        if not isinstance(lines, dict):
            return
        for num, ln in lines.items():
            num = str(num)
            if isinstance(ln, dict) and num.isdigit() and len(num) <= 3:
                try:
                    self._timing_line(self._drv(num), ln)
                except Exception as e:
                    self.counts["errors"] += 1
                    if self._err_logs < MAX_ERR_LOGS:
                        self._err_logs += 1
                        log.warning("timing line %s skipped: %r", num, e, exc_info=True)
        q, self._ot_queue = self._ot_queue, []
        for d, rec in q[:50]:
            try:
                self._check_overtakes(d, rec)
            except Exception:
                self.counts["errors"] += 1
        self.rev += 1

    def _timing_line(self, d: Drv, ln: dict) -> None:
        if "Line" in ln:
            l = to_int(ln["Line"])
            if l is not None:
                d.line = l
        if "Position" in ln:
            p = to_int(ln["Position"])
            if p and p > 0:
                d.pos = p
                if d.first_pos is None:
                    d.first_pos = p
        if "GapToLeader" in ln:
            d.gap_text, d.gap_sec = parse_gap(ln["GapToLeader"])
        if "IntervalToPositionAhead" in ln:
            d.int_text, d.int_sec = parse_gap(ln["IntervalToPositionAhead"])
        if "TimeDiffToFastest" in ln:
            self._saw_timediff = True
            d.tdf_seen = True
            d.tdf_text, d.tdf_sec = parse_gap(ln["TimeDiffToFastest"])
        if "TimeDiffToPositionAhead" in ln:
            self._saw_timediff = True
            d.tda_seen = True
            d.tda_text, d.tda_sec = parse_gap(ln["TimeDiffToPositionAhead"])
        lt = ln.get("LastLapTime")
        if isinstance(lt, dict):
            if "Value" in lt:
                sec = parse_laptime(lt["Value"])
                d.last_text = clean_text(lt["Value"])
                d.last_sec = sec
                if sec is not None:
                    d.last_updated = True
                    # a time in the same patch as a lap-counter increment belongs to the lap now completing
                    n_new = to_int(ln.get("NumberOfLaps")) if "NumberOfLaps" in ln else None
                    if d.pending is not None and not (n_new is not None and n_new > d.laps):
                        self._fill_pending(d, sec)
            if "OverallFastest" in lt:
                d.last_overall = to_bool(lt["OverallFastest"])
            if "PersonalFastest" in lt:
                d.last_personal = to_bool(lt["PersonalFastest"])
        bt = ln.get("BestLapTime")
        if isinstance(bt, dict):
            if "Value" in bt:
                d.best_text = clean_text(bt["Value"])
                d.best_sec = parse_laptime(bt["Value"])
                if d.best_sec is None:
                    d.best_text = ""
                    d.best_lap_no = None
            if "Lap" in bt:
                d.best_lap_no = to_int(bt["Lap"])
            if "OverallFastest" in bt:
                d.best_overall = to_bool(bt["OverallFastest"])
            if "PersonalFastest" in bt:
                d.best_personal = to_bool(bt["PersonalFastest"])
        bls = ln.get("BestLapTimes")
        if bls is not None:
            for i, v in indexed_items(bls):
                if i not in (0, 1, 2):
                    continue
                if isinstance(v, dict):
                    if "Value" not in v:
                        continue
                    val = v["Value"]
                    lapno = to_int(v.get("Lap"))
                else:
                    val, lapno = v, None
                sec = parse_laptime(val)
                d.best_laps[i] = {"text": clean_text(val), "sec": sec, "lap": lapno} if sec is not None else None
        if "KnockedOut" in ln:
            d.knocked_out = to_bool(ln["KnockedOut"])
        if "Cutoff" in ln:
            d.cutoff = to_bool(ln["Cutoff"])
        secs = ln.get("Sectors")
        if secs is not None:
            for i, s in indexed_items(secs):
                if i not in (0, 1, 2) or not isinstance(s, dict):
                    continue
                cell = d.sectors[i]
                if "Value" in s:
                    v = s["Value"]
                    sec = parse_laptime(v)
                    if sec is not None:
                        changed = cell.get("sec") != sec
                        cell["text"] = clean_text(v)
                        cell["sec"] = sec
                        d.sec_by_lap.setdefault(d.laps + 1, {})[i] = sec
                        if changed and i < 2 and d.progress is not None and not d.in_pit and not d.pit_out:
                            self.tb.add_sector_boundary(i, d.progress)
                if "OverallFastest" in s:
                    cell["overall"] = to_bool(s["OverallFastest"])
                if "PersonalFastest" in s:
                    cell["personal"] = to_bool(s["PersonalFastest"])
        sp = ln.get("Speeds")
        if isinstance(sp, dict):
            for k, key in (("I1", "i1"), ("I2", "i2"), ("FL", "fl"), ("ST", "st")):
                v = sp.get(k)
                if isinstance(v, dict):
                    v = v.get("Value")
                iv = to_int(v)
                if iv is not None and 0 < iv < 450:
                    d.traps[key] = iv
                    self._top_speed(d, iv)
        if "InPit" in ln:
            self._set_in_pit(d, to_bool(ln["InPit"]))
        if "PitOut" in ln:
            po = to_bool(ln["PitOut"])
            if po and not d.pit_out:
                d.pit_flag = True
                if d.pit_in_utc is not None:
                    self._finish_stop(d)
            d.pit_out = po
        if "Retired" in ln:
            r = to_bool(ln["Retired"])
            if r and not d.retired:
                self._event("retire", "%s retired" % d.tla, d.num, lap=d.laps + 1)
            d.retired = r
        if "Stopped" in ln:
            d.stopped = to_bool(ln["Stopped"])
        if "NumberOfPitStops" in ln:
            n = to_int(ln["NumberOfPitStops"])
            if n is not None and 0 <= n < 100:
                if (n > d.pits and not d.in_pit and len(d.stops) < n and not self._nonrace()
                        and not self._red()):
                    self._add_stop(d, None, d.laps + 1)
                d.pits = n
        # lap counter last, so the snapshot sees everything in this patch
        if "NumberOfLaps" in ln:
            n = to_int(ln["NumberOfLaps"])
            if n is not None and 0 <= n < 10000:
                self._laps_update(d, n)
        # non-race: a driver taking P1 is the "fastest" headline
        if ("Position" in ln or "BestLapTime" in ln) and d.pos == 1 and d.best_sec is not None \
                and self._p1 != d.num and self._nonrace():
            prev, self._p1 = self._p1, d.num
            if prev is not None or not self._snapshot:
                self._event("fastest", "%s P1 %s" % (d.tla, d.best_text or _fmt_time(d.best_sec)), d.num, lap=d.laps)

    # -- laps -----------------------------------------------------------
    def _laps_update(self, d: Drv, n: int) -> None:
        if not d.laps_seen:
            d.laps_seen = True
            if n == 1:
                d.laps = 0  # first ever value is a completed lap 1
            else:
                d.laps = n
                return
        if n > d.laps:
            first = d.laps + 1
            if n - d.laps > 3:
                first = n  # big jump (late join): only record the latest
                d.laps = n - 1
            for k in range(first, n + 1):
                d.laps = k
                self._complete_lap(d, k)
        elif n < d.laps:
            d.laps = n

    def _complete_lap(self, d: Drv, n: int) -> None:
        sec = d.last_sec if d.last_updated else None
        secs = d.sec_by_lap.pop(n, {})
        for k in [k for k in d.sec_by_lap if k < n]:
            del d.sec_by_lap[k]
        s = [secs.get(0), secs.get(1), secs.get(2)]
        if sec is not None and s[2] is None and s[0] is not None and s[1] is not None:
            v = sec - s[0] - s[1]
            if v > 0:
                s[2] = round(v, 3)
        st = d.cur_stint()
        pos = d.pos
        rec = {
            "lap": n, "sec": sec, "s": s,
            "compound": st.compound if st else None,
            "tyreAge": d.stint_age(st, n) if st else None,
            "pos": pos,
            "gapLeader": 0.0 if pos == 1 else d.gap_sec,
            "interval": 0.0 if pos == 1 else d.int_sec,
            "pit": bool(d.pit_flag or d.in_pit),
            "track": d.worst_track,
        }
        d.recs[n] = rec
        if len(d.recs) > MAX_RECS:
            for k in sorted(d.recs)[: len(d.recs) - MAX_RECS]:
                del d.recs[k]
        d.pending = rec if sec is None else None
        d.last_updated = False
        d.pit_flag = d.in_pit
        d.worst_track = self.track_code
        self.counts["laps"] += 1
        self.history_rev += 1
        if sec is not None:
            self._lap_time_known(d, rec)
        if not self._nonrace():
            self._ot_queue.append((d, rec))  # evaluated once the whole patch is applied

    def _fill_pending(self, d: Drv, sec: float) -> None:
        rec = d.pending
        d.pending = None
        d.last_updated = False
        rec["sec"] = sec
        s = rec["s"]
        if s[2] is None and s[0] is not None and s[1] is not None and sec - s[0] - s[1] > 0:
            s[2] = round(sec - s[0] - s[1], 3)
        self.history_rev += 1
        self._lap_time_known(d, rec)

    def _lap_time_known(self, d: Drv, rec: dict) -> None:
        sec = rec["sec"]
        if sec is None or sec <= 0:
            return
        if self._nonrace():
            return  # records come from best laps (see _fastest_record)
        f = self.fastest
        if f is None or sec < f["sec"]:
            self.fastest = {"num": d.num, "sec": sec, "text": d.last_text or _fmt_time(sec), "lap": rec["lap"]}
            if rec["lap"] >= 2 or d.last_overall:
                self._event("fastest", "%s fastest lap %s" % (d.tla, _fmt_time(sec)), d.num, lap=rec["lap"])
            self.rev += 1

    def _check_overtakes(self, a: Drv, rec: dict) -> None:
        if self._nonrace():
            return  # order changes by lap times, not passes
        n = rec["lap"]
        P = rec["pos"]
        prev = a.recs.get(n - 1)
        if not P or prev is None or not prev.get("pos"):
            return
        if P >= prev["pos"] or rec["pit"] or prev["pit"] or rec["track"] not in ("1", "2"):
            return
        found = 0
        for b in self.drivers.values():
            if b is a or b.retired or b.pos is None:
                continue
            bp = b.recs.get(n - 1)
            if bp is None or not bp.get("pos") or bp["pit"]:
                continue
            if bp["pos"] < prev["pos"] and b.pos > P and not b.in_pit and not b.pit_flag:
                key = (a.num, b.num, n)
                if key in self._overtake_seen:
                    continue
                if len(self._overtake_seen) > MAX_SEEN:
                    self._overtake_seen.clear()
                self._overtake_seen.add(key)
                self._event("overtake", "%s passed %s for P%d" % (a.tla, b.tla, P), a.num, lap=n)
                found += 1
                if found >= 3:
                    break

    # -- pits ------------------------------------------------------------
    def _set_in_pit(self, d: Drv, v: bool) -> None:
        if v and not d.in_pit:
            d.in_pit_since = self.utc
            # no "pit stops" in practice/qualifying, nor while the session is red-flagged
            d.pit_in_utc = None if (self._nonrace() or self._red()) else self.utc
            d.pit_in_lap = d.laps + 1
            d.pit_flag = True
        elif not v and d.in_pit and d.pit_in_utc is not None:
            self._finish_stop(d)
        if not v:
            d.in_pit_since = None
        d.in_pit = v

    def _finish_stop(self, d: Drv) -> None:
        dur = self.utc - d.pit_in_utc if d.pit_in_utc is not None else None
        if dur is not None and not (0 < dur < 300):
            dur = None
        lap = d.pit_in_lap
        d.pit_in_utc = None
        if d.stops and d.stops[-1]["durationSec"] is None and abs(self.utc - d.stops[-1]["utc"]) < 60:
            d.stops[-1]["durationSec"] = round(dur, 1) if dur is not None else None
            self.history_rev += 1
            self.rev += 1
            return
        self._add_stop(d, dur, lap)

    def _add_stop(self, d: Drv, dur: Optional[float], lap: int) -> None:
        st = d.cur_stint()
        comp = st.compound if st else None
        stop = {"num": d.num, "lap": lap, "utc": self.utc,
                "durationSec": round(dur, 1) if dur is not None else None,
                "fromCompound": comp, "toCompound": None}
        d.stops.append(stop)
        self.pitstops.append(stop)
        if len(self.pitstops) > MAX_STOPS:
            del self.pitstops[: len(self.pitstops) - MAX_STOPS]
        if len(d.stops) > 100:
            del d.stops[:50]
        txt = "%s pit stop" % d.tla + (" (%.1fs)" % dur if dur is not None else "")
        self._event("pit", txt, d.num, lap=lap)
        self.rev += 1

    # -- TimingAppData ----------------------------------------------------
    def _h_timingapp(self, data):
        lines = data.get("Lines") if isinstance(data, dict) else None
        if not isinstance(lines, dict):
            return
        for num, ln in lines.items():
            if not isinstance(ln, dict) or not str(num).isdigit():
                continue
            d = self._drv(num)
            if "GridPos" in ln:
                g = to_int(ln["GridPos"])
                if g:
                    d.grid = g
            if "Stints" in ln:
                for idx, s in indexed_items(ln["Stints"]):
                    if not isinstance(s, dict):
                        continue
                    st = d.stints.get(idx)
                    if st is None:
                        tl = to_int(s.get("TotalLaps")) or 0
                        sl = to_int(s.get("StartLaps")) or 0
                        start = d.laps if idx > 0 else max(0, d.laps - max(0, tl - sl))
                        st = d.stints[idx] = Stint(idx, start)
                    if "Compound" in s:
                        c = _clean_compound(s["Compound"])
                        if c:
                            st.compound = c
                    if "New" in s:
                        st.new = to_bool(s["New"])
                    if "StartLaps" in s:
                        v = to_int(s["StartLaps"])
                        if v is not None:
                            st.start_laps = v
                    if "TotalLaps" in s:
                        st.total_laps = to_int(s["TotalLaps"])
                cur = d.cur_stint()
                if cur and d.stops and len(d.stints) == len(d.stops) + 1 and cur.compound:
                    d.stops[-1]["toCompound"] = cur.compound
        self.rev += 1
        self.history_rev += 1

    # -- telemetry / position ---------------------------------------------
    def _top_speed(self, d: Drv, speed: int) -> None:
        if speed > d.max_speed and speed < 450:
            d.max_speed = speed
            if self.top_speed is None or speed > self.top_speed["kph"]:
                self.top_speed = {"num": d.num, "kph": speed}

    def _h_cardata(self, data):
        entries = data.get("Entries") if isinstance(data, dict) else None
        if not entries:
            return
        drivers = self.drivers
        dirty = self.dirty
        tb = self.tb
        for ent in as_list(entries):
            cars = ent.get("Cars") if isinstance(ent, dict) else None
            if not isinstance(cars, dict):
                continue
            ts = ent.get("Utc")
            if ts.__class__ is not str:
                ts = ""
            for num, c in cars.items():
                ch = c.get("Channels") if isinstance(c, dict) else None
                if not isinstance(ch, dict):
                    continue
                if num.__class__ is not str:
                    num = str(num)
                d = drivers.get(num)
                if d is None:
                    if not num.isdigit() or len(num) > 3:
                        continue
                    d = self._drv(num)
                if ts:
                    if d.tel_ts and ts < d.tel_ts and ts[:10] == d.tel_ts[:10]:
                        continue  # older/duplicate sample
                    d.tel_ts = ts
                tel = d.tel
                g = ch.get
                v = g("2", g(2))
                if v is not None:
                    v = to_int(v)
                    if v is not None and 0 <= v < 450:
                        tel["speed"] = v
                        if v > d.max_speed:
                            self._top_speed(d, v)
                v = g("0", g(0))
                if v is not None:
                    tel["rpm"] = to_int(v)
                v = g("3", g(3))
                if v is not None:
                    tel["gear"] = to_int(v)
                v = g("4", g(4))
                if v is not None:
                    tel["throttle"] = to_int(v)
                v = g("5", g(5))
                if v is not None:
                    tel["brake"] = norm_brake(v)
                v = g("45", g(45))
                if v is not None:
                    raw = to_int(v)
                    tel["drsRaw"] = raw
                    op = is_drs_open(raw)
                    tel["drs"] = op
                    if tb.track is not None and d.progress is not None:
                        tb.add_drs(d.progress, op)
                dirty.add(num)

    def _h_position(self, data):
        ents = data.get("Position") if isinstance(data, dict) else None
        if not ents:
            return
        tb = self.tb
        touched = []
        for ent in as_list(ents):
            es = ent.get("Entries") if isinstance(ent, dict) else None
            if not isinstance(es, dict):
                continue
            ts = ent.get("Timestamp")
            if ts.__class__ is not str:
                ts = ""
            for num, p in es.items():
                if not isinstance(p, dict):
                    continue
                if num.__class__ is not str:
                    num = str(num)
                x, y = to_float(p.get("X")), to_float(p.get("Y"))
                if x is None or y is None or (x == 0 and y == 0) or abs(x) > 1e7 or abs(y) > 1e7:
                    continue
                d = self.drivers.get(num)
                if d is None:
                    if not num.isdigit() or len(num) > 3:
                        continue
                    d = self._drv(num)
                if ts:
                    if d.xy_ts and ts < d.xy_ts and ts[:10] == d.xy_ts[:10]:
                        continue  # older/duplicate sample
                    d.xy_ts = ts
                status = str(p.get("Status") or "OnTrack")
                z = to_float(p.get("Z")) or 0
                d.xy = {"x": x, "y": y, "z": z, "status": status}
                if tb.track is None:
                    tb.add(num, x, y, lap=d.laps, clean=(status == "OnTrack" and not d.in_pit and not d.pit_out))
                touched.append(d)
                self.dirty.add(num)
        if tb.track is None:
            if tb.maybe_build():
                for d in self.drivers.values():
                    self._project(d)
                self.rev += 1
        else:
            for d in touched:
                self._project(d)

    def _project(self, d: Drv) -> None:
        if d.xy is None or self.tb.track is None:
            return
        prog, idx, _ = self.tb.track.project(d.xy["x"], d.xy["y"], d.hint)
        d.hint = idx
        d.progress = round(prog, 4)
        if d.xy.get("status") == "OnTrack" and not d.in_pit and not d.pit_out:
            self.tb.add_z(prog, d.xy.get("z") or 0.0)

    # -- race control ------------------------------------------------------
    def _h_rc(self, data):
        msgs = data.get("Messages") if isinstance(data, dict) else data
        for m in as_list(msgs):
            if not isinstance(m, dict):
                continue
            text = str(m.get("Message") or "")
            key = (m.get("Utc"), text)
            if key in self._rc_seen:
                continue
            if len(self._rc_seen) > MAX_SEEN:
                self._rc_seen.clear()
            self._rc_seen.add(key)
            utc = parse_iso(m.get("Utc")) or self.utc
            num = m.get("RacingNumber")
            num = str(num).strip() if num not in (None, "") else None
            if num is None:
                mc = _re_car.search(text)
                num = mc.group(1) if mc else None
            lap = to_int(m.get("Lap")) or self._cur_lap()
            flag = m.get("Flag")
            item = {"utc": utc, "lap": lap, "category": str(m.get("Category") or "Other"),
                    "flag": str(flag) if flag not in (None, "") else None,
                    "scope": str(m["Scope"]) if m.get("Scope") not in (None, "") else None,
                    "sector": to_int(m.get("Sector")), "message": text, "num": num}
            self.rc.append(item)
            if len(self.rc) > 400:
                del self.rc[:100]
            try:
                self._rc_events(item)
            except Exception as e:
                self.counts["errors"] += 1
                log.debug("rc event skipped: %r", e)
        self.rev += 1

    def _rc_events(self, it: dict) -> None:
        cat = (it["category"] or "").lower()
        text = it["message"]
        up = text.upper()
        utc, lap = it["utc"], it["lap"]
        d = self.drivers.get(it["num"]) if it["num"] else None
        tla = (d.tla + " ") if d else ""
        if "PENALTY" in up and not any(w in up for w in ("NO PENALTY", "NO FURTHER", "SERVED", "NOT PENALI", "WITHOUT PENALTY")):
            ms = _re_secs.search(up)
            ptxt = ("%s s penalty" % ms.group(1)) if ms else (
                "drive-through" if "DRIVE THROUGH" in up or "DRIVE-THROUGH" in up else
                "stop/go" if "STOP/GO" in up or "STOP AND GO" in up else
                "grid penalty" if "GRID" in up else "penalty")
            self._event("penalty", "%s%s: %s" % (tla, ptxt, text), it["num"], utc=utc, lap=lap)
            return
        if "DELETED" in up and "TIME" in up:  # "TIME 1:23.456 DELETED - TRACK LIMITS"
            self._event("penalty", "%slap time deleted: %s" % (tla, text), it["num"], utc=utc, lap=lap)
            return
        if cat == "safetycar" or up.startswith(("SAFETY CAR", "VIRTUAL SAFETY CAR", "VSC")):
            if cat == "flag":
                return
            kind = "vsc" if "VIRTUAL" in up or "VSC" in up else "sc"
            if not self._recent_event(kind):
                self._event(kind, text, None, utc=utc, lap=lap)
            return
        if cat == "flag":
            flag = (it["flag"] or "").upper()
            if flag == "BLUE":
                return
            if flag == "RED" and self._recent_event("flag", contains="RED"):
                return
            t = text or flag
            if not t:
                return
            scope = it.get("scope")
            self._event("flag", t, it["num"] if scope == "Driver" else None, utc=utc, lap=lap)

    def _h_trackstatus(self, data):
        if not isinstance(data, dict) or "Status" not in data:
            return
        code = str(data.get("Status")).strip()
        self.track_msg = str(data.get("Message") or "")
        prev = self.track_code
        self.track_code = code
        if code != prev:
            sev = TRACK_SEVERITY.get(code, 0)
            for d in self.drivers.values():
                if TRACK_SEVERITY.get(d.worst_track, 0) < sev:
                    d.worst_track = code
            if code == "4" and not self._recent_event("sc"):
                self._event("sc", "SAFETY CAR DEPLOYED")
            elif code == "6" and not self._recent_event("vsc"):
                self._event("vsc", "VIRTUAL SAFETY CAR DEPLOYED")
            elif code == "5" and not self._recent_event("flag", contains="RED"):
                self._event("flag", "RED FLAG")
            elif code == "1" and prev in ("4", "5", "6", "7") and not self._recent_event("flag", 5.0, "CLEAR"):
                self._event("flag", "TRACK CLEAR")
        self.rev += 1

    def _h_weather(self, data):
        if isinstance(data, dict):
            self.weather.update({k: v for k, v in data.items()})
            self.rev += 1

    def _h_lapcount(self, data):
        if not isinstance(data, dict):
            return
        if "CurrentLap" in data:
            v = to_int(data["CurrentLap"])
            self.lap_current = v if v is not None and 0 <= v < 10000 else self.lap_current
        if "TotalLaps" in data:
            v = to_int(data["TotalLaps"])
            self.lap_total = v if v is not None and 0 < v < 10000 else self.lap_total
        self.rev += 1

    def _h_sessionstatus(self, data):
        if not isinstance(data, dict) or "Status" not in data:
            return
        st = str(data["Status"]).strip()
        prev = self.session.get("status")
        self.session["status"] = st
        if st == "Started":
            if self.started_utc is None:
                self.started_utc = self.utc
            self.ended_utc = None
            if prev == "Aborted" and not self._snapshot:
                self._event("flag", "SESSION RESUMED")
        if st in ("Finished", "Finalised", "Ends") and self.ended_utc is None:
            self.ended_utc = self.utc
        if st == "Aborted" and prev != "Aborted" and not self._snapshot \
                and not self._recent_event("flag", contains="RED"):
            self._event("flag", "RED FLAG - SESSION SUSPENDED")
        self.rev += 1

    def _h_sessioninfo(self, data):
        if not isinstance(data, dict):
            return
        m = data.get("Meeting")
        if isinstance(m, dict) and m.get("Name"):
            self.session["meeting"] = str(m["Name"])
        if not isinstance(m, dict) and m is not None:
            data = {k: v for k, v in data.items() if k != "Meeting"}
        if data.get("Name"):
            self.session["name"] = str(data["Name"])
        if data.get("Type"):
            self.session["type"] = str(data["Type"])
        self._kind_c = session_kind(self.session.get("type"), self.session.get("name"))
        deep_merge(self.raw.setdefault("SessionInfo", {}), data)
        self.rev += 1

    def _h_clock(self, data):
        if not isinstance(data, dict):
            return
        c = self.clock
        if "Remaining" in data:
            c["remaining"] = parse_clock(data["Remaining"])
        if "Utc" in data:
            c["utc"] = parse_iso(data["Utc"])
        elif "Remaining" in data:
            c["utc"] = self.utc
        if "Extrapolating" in data:
            c["extrapolating"] = to_bool(data["Extrapolating"])
        if c["utc"] is None:
            c["utc"] = self.utc
        self.rev += 1

    def _h_radio(self, data):
        caps = data.get("Captures") if isinstance(data, dict) else None
        for c in as_list(caps):
            if not isinstance(c, dict) or not c.get("Path"):
                continue
            path = str(c["Path"])
            if path in self._radio_seen:
                continue
            self._radio_seen.add(path)
            num = str(c.get("RacingNumber")) if c.get("RacingNumber") is not None else None
            utc = parse_iso(c.get("Utc")) or self.utc
            self.radio.append({"utc": utc, "num": num, "url": STATIC_URL + path.lstrip("/")})
            if len(self.radio) > 200:
                del self.radio[:50]
            if len(self._radio_seen) > MAX_SEEN:
                self._radio_seen.clear()
            d = self.drivers.get(num) if num else None
            self._event("radio", "Team radio %s" % (d.tla if d else (num or "")), num, utc=utc)
        self.rev += 1

    # ------------------------------------------------------------------ outputs
    def _sorted(self) -> List[Drv]:
        if self._nonrace():
            return sorted(self.drivers.values(),
                          key=lambda d: (d.pos if d.pos else 999,
                                         d.best_sec if d.best_sec is not None else 1e9, d.line, _numkey(d.num)))
        return sorted(self.drivers.values(), key=lambda d: (d.pos if d.pos else 999, d.line, _numkey(d.num)))

    def _cell(self, text, sec, overall, personal) -> dict:
        return {"text": text or "", "sec": sec, "overall": bool(overall), "personal": bool(personal)}

    def _fastest_record(self, nonrace: bool) -> Optional[dict]:
        """Session fastest lap. Race: tracked from completed laps. Non-race: min over current best laps
        (so deleted times drop out and quali parts are covered)."""
        if not nonrace:
            return self.fastest
        best = None
        for d in self.drivers.values():
            cands = [(d.best_sec, d.best_text, d.best_lap_no)]
            cands += [(b["sec"], b["text"], b.get("lap")) for b in d.best_laps if b]
            for sec, text, lap in cands:
                if sec is not None and sec > 0 and (best is None or sec < best["sec"]):
                    best = {"num": d.num, "sec": sec, "text": text or _fmt_time(sec), "lap": lap}
        return best

    def _activity(self, d: Drv, nonrace: bool) -> str:
        if d.retired:
            return "retired"
        if d.in_pit:
            if nonrace and d.in_pit_since is not None and self.utc - d.in_pit_since >= GARAGE_AFTER:
                return "garage"
            return "pit"
        if d.pit_out:
            return "out"
        return "track"

    def _car(self, d: Drv, nonrace: bool = False, rank: int = 0, ctx: Optional[dict] = None) -> dict:
        ctx = ctx or {}
        pos = rank if (nonrace and rank) else d.pos
        grid = None if nonrace else (d.grid or d.first_pos)
        ahead = ctx.get("ahead")
        base = ctx.get("base")
        gfs: Optional[float] = None
        gft: Optional[str] = None
        if nonrace:
            # TimeDiffToFastest / TimeDiffToPositionAhead, falling back to best laps when the feed omits them
            has_t = d.best_sec is not None
            if rank == 1 and has_t:
                gft, gfs = "+0.000", 0.0
            elif d.tdf_text:
                gft, gfs = d.tdf_text, d.tdf_sec
            elif has_t and base is not None:
                gfs = round(d.best_sec - base, 3)
                gft = "+%.3f" % gfs
            else:
                gft, gfs = "", None
            if rank == 1 and has_t:
                it, is_ = "+0.000", 0.0
            elif d.tda_text:
                it, is_ = d.tda_text, d.tda_sec
            elif has_t and ahead is not None and ahead.best_sec is not None:
                is_ = round(d.best_sec - ahead.best_sec, 3)
                it = "+%.3f" % is_
            else:
                it, is_ = "", None
            gt, gs = gft, gfs
        else:
            if pos == 1:
                gt, gs = (d.gap_text, d.gap_sec) if d.gap_text and d.gap_sec is not None else ("+0.000", 0.0)
                it, is_ = (d.int_text, d.int_sec) if d.int_text and d.int_sec is not None else ("+0.000", 0.0)
            else:
                gt, gs, it, is_ = d.gap_text, d.gap_sec, d.int_text, d.int_sec
        best_overall = d.best_overall
        if best_overall is None:
            f = ctx.get("fastest")
            best_overall = bool(f and d.best_sec is not None and abs(f["sec"] - d.best_sec) < 1e-6)
        sectors = []
        for c in d.sectors:
            if c.get("sec") is None:
                sectors.append({"text": None, "sec": None, "overall": False, "personal": False})
            else:
                sectors.append(self._cell(c.get("text"), c.get("sec"), c.get("overall"), c.get("personal")))
        st = d.cur_stint()
        tyre = ({"compound": st.compound, "age": d.stint_age(st), "new": st.new, "stint": st.idx + 1}
                if st else {"compound": None, "age": None, "new": None, "stint": None})
        stints = [{"compound": s.compound, "new": s.new, "startLap": s.start_count,
                   "laps": d.stint_age(s) - s.start_laps if s is st else
                   max(0, (d.stints[s.idx + 1].start_count if (s.idx + 1) in d.stints else d.laps) - s.start_count)}
                  for s in (d.stints[i] for i in sorted(d.stints))]
        tel = d.tel
        return {
            "num": d.num, "tla": d.tla, "name": d.name, "team": d.team, "color": d.color,
            "pos": pos, "grid": grid, "posChange": (grid - pos) if (grid and pos) else 0,
            "gapLeader": gt, "interval": it, "gapLeaderSec": gs, "intervalSec": is_,
            "gapFastest": gft, "gapFastestSec": gfs,
            "knockedOut": d.knocked_out, "cutoff": d.cutoff,
            "bestLaps": [self._cell_best(b) for b in d.best_laps],
            "activity": self._activity(d, nonrace), "timedLaps": d.laps,
            "lap": d.laps, "pits": d.pits, "inPit": d.in_pit, "pitOut": d.pit_out,
            "retired": d.retired, "stopped": d.stopped,
            "last": self._cell(d.last_text, d.last_sec, d.last_overall, d.last_personal),
            "best": self._cell(d.best_text, d.best_sec, best_overall, d.best_personal if d.best_personal is not None else bool(d.best_sec)),
            "sectors": sectors,
            "speedTraps": {"i1": d.traps.get("i1"), "i2": d.traps.get("i2"), "fl": d.traps.get("fl"), "st": d.traps.get("st")},
            "tyre": tyre, "stints": stints,
            "tel": {"speed": tel.get("speed"), "rpm": tel.get("rpm"), "gear": tel.get("gear"),
                    "throttle": tel.get("throttle"), "brake": tel.get("brake"),
                    "drs": bool(tel.get("drs", False)), "drsRaw": tel.get("drsRaw")},
            "xy": dict(d.xy) if d.xy else None,
            "progress": d.progress,
        }

    @staticmethod
    def _cell_best(b: Optional[dict]) -> dict:
        if not b:
            return {"text": None, "sec": None}
        return {"text": b["text"], "sec": b["sec"]}

    def state(self) -> dict:
        """Never raises; always JSON-safe (NaN/inf -> null)."""
        try:
            return _sanitize(self._state())
        except Exception as e:
            self.counts["errors"] += 1
            log.warning("state() failed: %r", e, exc_info=True)
            return {"type": "state", "utc": self.utc, "session": {"name": None, "meeting": None, "type": None,
                    "status": "Inactive", "kind": "unknown", "part": None, "partLabel": None},
                    "clock": {"remainingSec": None, "extrapolating": False, "elapsedSec": None},
                    "lap": {"current": None, "total": None},
                    "trackStatus": {"code": "1", "label": "AllClear", "flag": "green"}, "weather": {},
                    "order": [], "cars": {}, "raceControl": [], "radio": [],
                    "records": {"fastestLap": None, "topSpeed": None}, "pitStops": []}

    def _state(self) -> dict:
        kind = self._kind()
        nonrace = self._nonrace()
        if kind == "unknown" and nonrace:
            kind = "practice"  # TimeDiff* fields without SessionInfo: closest guess
        ds = self._sorted()
        code = self.track_code
        flag = TRACK_FLAGS.get(code, "green")
        label = TRACK_LABELS.get(code, "Unknown")
        if self.session.get("status") == "Aborted":
            flag, label = "red", "Red"
        w = self.weather
        rain = to_float(w.get("Rainfall"))
        clock = self.clock
        rem = clock["remaining"]
        if rem is not None and clock["extrapolating"] and clock["utc"] is not None:
            rem = rem - max(0.0, self.utc - clock["utc"])
        if rem is not None:
            rem = max(0.0, rem)
        elapsed = None
        if self.started_utc is not None:
            elapsed = round((self.ended_utc or self.utc) - self.started_utc, 1)
        cur = self._cur_lap()
        lap_cur: Optional[int] = cur
        if self.lap_total and cur > self.lap_total:
            lap_cur = self.lap_total
        if nonrace:
            lap_cur = None
        part = self.session_part if kind in ("qualifying", "sprint_qualifying") else None
        names = {d.num: d.tla for d in ds}
        fastest = self._fastest_record(nonrace)
        ctx: Dict[str, Any] = {"fastest": fastest, "base": fastest["sec"] if (nonrace and fastest) else None}
        cars = {}
        prev: Optional[Drv] = None
        for i, d in enumerate(ds):
            ctx["ahead"] = prev
            cars[d.num] = self._car(d, nonrace, i + 1 if nonrace else 0, ctx)
            prev = d
        session = dict(self.session)
        session.update({"kind": kind, "part": part, "partLabel": ("Q%d" % part) if part else None})
        return {
            "type": "state", "utc": self.utc,
            "session": session,
            "clock": {"remainingSec": rem, "extrapolating": bool(clock["extrapolating"]), "elapsedSec": elapsed},
            "lap": {"current": lap_cur, "total": self.lap_total if self.lap_total and self.lap_total > 0 else None},
            "trackStatus": {"code": code, "label": label, "flag": flag},
            "weather": {"air": to_float(w.get("AirTemp")), "track": to_float(w.get("TrackTemp")),
                        "humidity": to_float(w.get("Humidity")), "pressure": to_float(w.get("Pressure")),
                        "wind": to_float(w.get("WindSpeed")), "windDir": to_float(w.get("WindDirection")),
                        "rain": bool(rain and rain > 0)},
            "order": [d.num for d in ds],
            "cars": cars,
            "raceControl": list(reversed(self.rc[-MAX_RC:])),
            "radio": [dict(r, tla=names.get(r["num"])) for r in reversed(self.radio[-MAX_RADIO:])],
            "records": {"fastestLap": dict(fastest) if fastest else None,
                        "topSpeed": dict(self.top_speed) if self.top_speed else None},
            "pitStops": [dict(p) for p in self.pitstops],
        }

    def fast_delta(self) -> Optional[dict]:
        if not self.dirty:
            return None
        cars = {}
        for num in self.dirty:
            d = self.drivers.get(num)
            if d is None:
                continue
            c: Dict[str, Any] = {}
            if d.xy:
                c["x"] = d.xy["x"]
                c["y"] = d.xy["y"]
                c["z"] = d.xy["z"]
            t = d.tel
            for k in ("speed", "rpm", "gear", "throttle", "brake", "drs"):
                if k in t:
                    c[k] = t[k]
            if d.progress is not None:
                c["progress"] = d.progress
            if c:
                cars[num] = c
        self.dirty = set()
        if not cars:
            return None
        return _sanitize({"type": "fast", "utc": self.utc, "cars": cars})

    def history(self) -> dict:
        if self._hist_cache is not None and self._hist_cache_rev == self.history_rev:
            return self._hist_cache
        laps = {}
        mx = 0
        for num, d in self.drivers.items():
            if d.recs:
                laps[num] = [d.recs[k] for k in sorted(d.recs)]
                mx = max(mx, max(d.recs))
        h = _sanitize({"type": "history", "maxLap": mx, "laps": laps, "events": list(self.events)})
        self._hist_cache, self._hist_cache_rev = h, self.history_rev
        return h

    def track_dict(self) -> Optional[dict]:
        return self.tb.to_dict()

    def stats(self) -> dict:
        return {"drivers": len(self.drivers), "laps": self.counts["laps"], "events": len(self.events),
                "pitStops": len(self.pitstops), "radio": len(self.radio), "raceControl": len(self.rc),
                "msgs": self.counts["msgs"], "errors": self.counts["errors"], "ignored": self.counts["ignored"],
                "trackBuilt": self.tb.track is not None, "utc": self.utc, "topics": dict(self.topic_counts)}


def _fmt_time(sec: float) -> str:
    m = int(sec // 60)
    return "%d:%06.3f" % (m, sec - 60 * m) if m else "%.3f" % sec


def _numkey(n: str):
    try:
        return int(n)
    except ValueError:
        return 9999


def _sanitize(o: Any) -> Any:
    """Copy of o with NaN/inf floats replaced by None (so json.dumps(allow_nan=False) works)."""
    if isinstance(o, float):
        return o if (o == o and o not in (math.inf, -math.inf)) else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o
