"""Track geometry: build a closed centerline from Position X/Y samples and project cars onto it."""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

MAX_SAMPLES_PER_CAR = 8000
MIN_RUN_POINTS = 60
RESAMPLE_POINTS = 800
DRS_BINS = 400


class Track:
    """Immutable-ish closed polyline with projection."""

    def __init__(self, pts: np.ndarray, length_m: float, start_idx: int = 0,
                 sectors: Optional[List[float]] = None, drs_zones: Optional[List[List[float]]] = None,
                 source: str = "learned", corners: Optional[List[Dict[str, Any]]] = None,
                 marshal_sectors: Optional[List[float]] = None, official_rotation: Optional[float] = None):
        self.pts = np.asarray(pts, dtype=float)  # (M,2), closed implicitly (last connects to first)
        self.n = len(self.pts)
        nxt = np.roll(self.pts, -1, axis=0)
        self.seg = nxt - self.pts
        self.seglen = np.hypot(self.seg[:, 0], self.seg[:, 1])
        self.seglen2 = np.maximum(self.seglen ** 2, 1e-9)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seglen)[:-1]])
        self.total = float(self.seglen.sum()) or 1.0
        self.length = float(length_m)
        self.start_idx = int(start_idx)
        self.sectors = sectors or [1 / 3, 2 / 3]
        self.drs_zones = drs_zones or []
        self.source = source                      # "learned" (from car positions) | "official" (MultiViewer layout)
        self.corners = corners                    # [{"n": 1, "frac": 0.05, "letter": ""}] or None
        self.marshal_sectors = marshal_sectors    # [frac, ...] or None
        self.official_rotation = official_rotation
        mn = self.pts.min(axis=0)
        mx = self.pts.max(axis=0)
        self.bounds = [float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])]
        # Elevation learned from live Position Z samples (running mean per point).
        self._z_sum = np.zeros(self.n)
        self._z_cnt = np.zeros(self.n)

    def add_z(self, progress: float, z: float) -> None:
        """Accumulate an elevation sample (raw Position.Z units) at lap progress 0..1."""
        i = int(progress * self.n) % self.n
        self._z_sum[i] += z
        self._z_cnt[i] += 1

    def elevation(self) -> Optional[List[float]]:
        """Smoothed elevation per point in metres relative to the lowest point; None until sampled."""
        have = self._z_cnt > 0
        if have.sum() < max(10, self.n // 10):
            return None
        idx = np.arange(self.n)
        z = np.interp(idx, idx[have], (self._z_sum[have] / self._z_cnt[have]), period=self.n)
        k = np.ones(15) / 15
        pad = np.concatenate([z[-15:], z, z[:15]])
        z = np.convolve(pad, k, mode="same")[15:-15]
        return [round(float(v) / 10.0, 2) for v in (z - z.min())]

    def project(self, x: float, y: float, hint: Optional[int] = None, window: int = 40) -> Tuple[float, int, float]:
        """Return (progress 0..1, segment index, distance). `hint` = previous segment idx for a local search."""
        pts = self.pts
        n = self.n
        if hint is not None and 0 <= hint < n:
            idx = (np.arange(hint - window // 4, hint + window) % n)
            p = pts[idx]
            s = self.seg[idx]
            l2 = self.seglen2[idx]
        else:
            idx = None
            p, s, l2 = pts, self.seg, self.seglen2
        dx = x - p[:, 0]
        dy = y - p[:, 1]
        t = np.clip((dx * s[:, 0] + dy * s[:, 1]) / l2, 0.0, 1.0)
        ex = dx - t * s[:, 0]
        ey = dy - t * s[:, 1]
        d2 = ex * ex + ey * ey
        k = int(np.argmin(d2))
        dist = math.sqrt(float(d2[k]))
        i = int(idx[k]) if idx is not None else k
        # fall back to global search if the local one is clearly off the line
        if idx is not None and dist > 0.05 * self.total:
            return self.project(x, y, None)
        prog = (self.cum[i] + float(t[k]) * self.seglen[i]) / self.total
        return (prog % 1.0), i, dist

    def to_dict(self) -> Dict[str, Any]:
        d = self._base_dict()
        d["source"] = self.source
        if self.corners:
            d["corners"] = self.corners
        if self.marshal_sectors:
            d["marshalSectors"] = [round(float(v), 4) for v in self.marshal_sectors]
        if self.official_rotation is not None:
            d["officialRotation"] = round(float(self.official_rotation), 2)
        return d

    def _base_dict(self) -> Dict[str, Any]:
        return {
            "points": [[round(float(a), 1), round(float(b), 1)] for a, b in self.pts],
            "length": round(self.length, 1),
            "bounds": [round(v, 1) for v in self.bounds],
            "startIdx": self.start_idx,
            "drsZones": [[round(a, 4), round(b, 4)] for a, b in self.drs_zones],
            "sectors": [round(v, 4) for v in self.sectors],
            "rotation": 0,
            "elevation": self.elevation(),
        }


def _clean_dupes(xy: np.ndarray, eps: float = 1.0) -> np.ndarray:
    if len(xy) < 2:
        return xy
    d = np.hypot(*(xy[1:] - xy[:-1]).T)
    keep = np.concatenate([[True], d > eps])
    return xy[keep]


def _path_len(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.hypot(*(xy[1:] - xy[:-1]).T).sum())


def _closure_err(xy: np.ndarray) -> float:
    """Gap between end and start relative to path length."""
    pl = _path_len(xy)
    if pl <= 0:
        return 1.0
    return float(np.hypot(*(xy[-1] - xy[0]))) / pl


def smooth_closed(xy: np.ndarray, window: int = 7, passes: int = 2) -> np.ndarray:
    n = len(xy)
    if n < window * 2:
        return xy
    k = np.ones(window) / window
    out = xy.copy()
    pad = window
    for _ in range(passes):
        ext = np.concatenate([out[-pad:], out, out[:pad]])
        for c in (0, 1):
            out[:, c] = np.convolve(ext[:, c], k, mode="same")[pad:-pad]
    return out


def resample_closed(xy: np.ndarray, m: int = RESAMPLE_POINTS) -> np.ndarray:
    closed = np.vstack([xy, xy[:1]])
    d = np.hypot(*(closed[1:] - closed[:-1]).T)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = cum[-1]
    targets = np.linspace(0, total, m, endpoint=False)
    xs = np.interp(targets, cum, closed[:, 0])
    ys = np.interp(targets, cum, closed[:, 1])
    return np.column_stack([xs, ys])


def dist_to_polyline(poly: np.ndarray, pts: np.ndarray, chunk: int = 400) -> np.ndarray:
    """Distance from each of ``pts`` (K,2) to the CLOSED polyline ``poly`` (M,2)."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    l2 = np.maximum((ab ** 2).sum(axis=1), 1e-9)
    out = np.empty(len(pts))
    for s in range(0, len(pts), chunk):
        p = pts[s:s + chunk]
        dx = p[:, 0:1] - a[None, :, 0]
        dy = p[:, 1:2] - a[None, :, 1]
        t = np.clip((dx * ab[None, :, 0] + dy * ab[None, :, 1]) / l2[None, :], 0.0, 1.0)
        ex = dx - t * ab[None, :, 0]
        ey = dy - t * ab[None, :, 1]
        out[s:s + chunk] = np.sqrt((ex * ex + ey * ey).min(axis=1))
    return out


def _finite_xy(x: Any, y: Any) -> bool:
    try:
        return math.isfinite(x) and math.isfinite(y)
    except TypeError:
        return False


def parse_circuit_layout(data: Any) -> Optional[Dict[str, Any]]:
    """MultiViewer ``/api/v1/circuits/{key}/{year}`` JSON -> ``{"points": (N,2), "corners": [...], "marshal": [...],
    "rotation": float|None}`` or ``None``.  Lenient: only the ``x``/``y`` arrays are required; every marker is optional.

    Recalled schema: ``{"x":[...], "y":[...], "rotation":92, "corners":[{"number":1,"letter":"","angle":..,
    "length":..,"trackPosition":{"x":..,"y":..}}], "marshalSectors":[{... "number":1, "trackPosition":{..}}],
    "marshalLights":[...], "candidateLap":{...}, "circuitName":..., "circuitKey":..., "year":...}``.  The x/y share
    the coordinate system of the live Position feed (``rotation`` only rotates them for display)."""
    if not isinstance(data, dict):
        return None
    xs, ys = data.get("x"), data.get("y")
    if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != len(ys) or len(xs) < 50:
        return None
    try:
        pts = np.array([[float(a), float(b)] for a, b in zip(xs, ys)], dtype=float)
    except (TypeError, ValueError):
        return None
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 50:
        return None
    if np.hypot(*(pts[-1] - pts[0])) < 1e-6:  # explicitly closed: drop the duplicate end point
        pts = pts[:-1]
    pts = _clean_dupes(pts, 1e-6)
    if len(pts) < 50 or _path_len(pts) <= 0:
        return None

    def markers(key: str) -> List[Dict[str, Any]]:
        out = []
        for i, e in enumerate(data.get(key) or []):
            if not isinstance(e, dict):
                continue
            tp = e.get("trackPosition") if isinstance(e.get("trackPosition"), dict) else e
            try:
                x, y = float(tp.get("x")), float(tp.get("y"))
            except (TypeError, ValueError):
                continue
            if not _finite_xy(x, y):
                continue
            try:
                n = int(e.get("number", i + 1))
            except (TypeError, ValueError):
                n = i + 1
            out.append({"n": n, "letter": str(e.get("letter") or ""), "x": x, "y": y})
        return out

    rot = data.get("rotation")
    try:
        rot = float(rot) if rot is not None else None
    except (TypeError, ValueError):
        rot = None
    return {"points": pts, "corners": markers("corners"), "marshal": markers("marshalSectors"), "rotation": rot}


class TrackBuilder:
    """Accumulates OnTrack samples; builds a `Track` from the cleanest closed lap of any car.

    Practice/qualifying notes: cars parked in the garage or crawling in the pit lane produce long runs of
    (near-)identical points - they are de-duplicated on intake (so they cannot eat the per-car sample cap) and
    candidate laps are ranked by how well they agree with the *other* cars' samples, so a pit-lane detour, an
    out/in lap or a lap with an off cannot win over the racing line a majority of cars drove.  Laps without lap
    tags are found by loop-closure detection from several start points.  Call :meth:`reset` when the circuit
    changes (new meeting) - the engine and the server do."""

    def __init__(self, resample: int = RESAMPLE_POINTS):
        self.resample = resample
        # per car: list of (x, y, lap_tag, clean)
        self._samples: Dict[str, List[Tuple[float, float, Any, bool]]] = {}
        self.track: Optional[Track] = None
        self.version = 0
        self.resets = 0
        self.circuit_key: Any = None   # circuit the current samples/track belong to (set via set_context)
        self.year: Optional[int] = None
        self._clear_state()

    def _clear_state(self) -> None:
        self._since_try = 0
        self._try_every = 150
        self._z_n = 0
        self._drs_open = np.zeros(DRS_BINS)
        self._drs_total = np.zeros(DRS_BINS)
        self._drs_n = 0
        self._drs_last_n = 0
        self._sec_samples: List[List[float]] = [[], []]

    # ---- lifecycle -----------------------------------------------------
    def reset(self) -> None:
        """Forget the track and every collected sample (new circuit / user request).  Bumps ``version``;
        ``track`` becomes ``None`` until a new one is learned (or an official layout accepted)."""
        self._samples.clear()
        self.track = None
        self._clear_state()
        self.resets += 1
        self.version += 1
        log.info("track reset")

    def set_context(self, circuit_key: Any, year: Optional[int] = None) -> bool:
        """Tell the builder which circuit the incoming samples belong to.  A *different* known circuit resets
        the builder first (returns True when it did)."""
        did = False
        if circuit_key is not None and self.circuit_key is not None and str(circuit_key) != str(self.circuit_key):
            self.reset()
            did = True
        if circuit_key is not None:
            self.circuit_key = circuit_key
        if year is not None:
            self.year = year
        return did

    # ---- sample intake -------------------------------------------------
    def add(self, num: str, x: float, y: float, lap: Any = None, clean: bool = True) -> None:
        if self.track is not None:
            return
        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            return
        if not _finite_xy(x, y) or (x == 0.0 and y == 0.0):  # (0,0) = "no fix" in the feed
            return
        lst = self._samples.setdefault(num, [])
        if lst:
            lx, ly, ltag, lclean = lst[-1]
            if abs(x - lx) < 1.0 and abs(y - ly) < 1.0 and ltag == lap and lclean == clean:
                return  # parked / not moving (garage, grid, red flag): nothing new to learn, keep the cap for real laps
        if len(lst) < MAX_SAMPLES_PER_CAR:
            lst.append((x, y, lap, bool(clean)))
            self._since_try += 1

    def maybe_build(self, force: bool = False) -> bool:
        """Try to build; cheap to call often (rate limited with back-off). True if a track was just built."""
        if self.track is not None:
            return False
        if not force and self._since_try < self._try_every:
            return False
        self._since_try = 0
        self._try_every = min(self._try_every * 2, 1500)  # a failing build is O(samples): back off
        try:
            return self.build()
        except Exception:  # pragma: no cover
            log.exception("track build failed")
            return False

    # ---- build ---------------------------------------------------------
    def _candidates(self):
        for num, lst in self._samples.items():
            if len(lst) < MIN_RUN_POINTS:
                continue
            # 1. runs by lap tag (clean only)
            run: List[Tuple[float, float]] = []
            run_tag: Any = object()
            ok = True
            runs: List[np.ndarray] = []
            for x, y, tag, clean in lst:
                if tag != run_tag:
                    if run and ok and run_tag is not None:
                        runs.append(np.array(run))
                    run, run_tag, ok = [], tag, True
                if not clean:
                    ok = False
                run.append((x, y))
            # last (open) run is not a full lap -> skip
            for r in runs:
                r = _clean_dupes(r)
                if len(r) >= MIN_RUN_POINTS and _closure_err(r) < 0.06:
                    yield r
            # 2. closure detection on the clean-only stream, from several start points (the first clean sample may
            #    be on a pit exit road that is never driven again; a later start point still closes)
            clean_pts = _clean_dupes(np.array([(x, y) for x, y, _, c in lst if c]))
            if len(clean_pts) >= MIN_RUN_POINTS:
                for start in range(0, min(len(clean_pts) - MIN_RUN_POINTS, 6 * 120), 120):
                    r = self._closure_run(clean_pts[start:])
                    if r is not None:
                        yield r

    @staticmethod
    def _closure_run(xy: np.ndarray) -> Optional[np.ndarray]:
        n = len(xy)
        if n < MIN_RUN_POINTS:
            return None
        d = np.hypot(*(xy - xy[0]).T)
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(*(xy[1:] - xy[:-1]).T))])
        runmax = np.maximum.accumulate(d)
        ok = (cum > 0) & (runmax > 0.2 * cum) & (d < 0.03 * cum)
        ok[:10] = False
        hits = np.nonzero(ok)[0]
        if len(hits) == 0:
            return None
        return xy[: int(hits[0]) + 1]

    def _observed(self, limit: int = 2000) -> np.ndarray:
        """Clean (on-track, not in pit) samples of every car, sub-sampled to ``limit`` points."""
        pts = [(x, y) for lst in self._samples.values() for x, y, _, c in lst if c]
        if not pts:
            return np.empty((0, 2))
        arr = np.array(pts)
        if len(arr) > limit:
            arr = arr[np.linspace(0, len(arr) - 1, limit).astype(int)]
        return arr

    def build(self) -> bool:
        cands = []
        for r in self._candidates():
            cands.append((round(_closure_err(r), 3), -len(r), r))
        if not cands:
            return False
        cands.sort(key=lambda c: (c[0], c[1]))
        best = cands[0][2]
        if len(cands) > 1:
            # Rank the best few candidates by agreement with all observed samples: laps through the pit lane or with
            # an off-track excursion disagree with what the other cars drove.
            obs = self._observed(1200)
            if len(obs) >= 100:
                top = cands[:6]
                lens = np.array([_path_len(c[2]) for c in top])
                med = float(np.median(lens))
                scored = []
                for (ce, nl, r), ln in zip(top, lens):
                    if med > 0 and abs(ln - med) > 0.25 * med:
                        continue  # length outlier (pit lane detour / part lap)
                    poly = resample_closed(r, 300)
                    cov = float((dist_to_polyline(poly, obs) <= 0.01 * _path_len(np.vstack([poly, poly[:1]]))).mean())
                    scored.append((-round(cov, 2), ce, nl, r))
                if scored:
                    scored.sort(key=lambda c: c[:3])
                    best = scored[0][3]
        sm = smooth_closed(resample_closed(best, self.resample * 2), window=9, passes=2)
        pts = resample_closed(sm, self.resample)
        raw_len = _path_len(np.vstack([pts, pts[:1]]))
        length_m = raw_len / 10.0 if raw_len > 15000 else raw_len
        self.track = Track(pts, length_m)
        self.version += 1
        self._samples.clear()
        log.info("track built: %d pts, %.0f m", len(pts), length_m)
        return True

    # ---- official layout (fallback) ----------------------------------------
    def offer_reference(self, layout: Dict[str, Any], min_samples: int = 150, min_fraction: float = 0.8) -> str:
        """Try to adopt an official circuit polyline (see :func:`parse_circuit_layout`) instead of learning one.

        Returns ``"accepted"``, ``"rejected"`` (the observed on-track samples do not lie on the polyline: wrong circuit /
        different coordinate system - discard it and keep learning), ``"undecided"`` (not enough samples yet - try
        again later) or ``"ignored"`` (a track already exists).  Accepted only if at least ``min_fraction`` of the
        observed clean samples are within a tolerance of the polyline."""
        if self.track is not None:
            return "ignored"
        pts = np.asarray(layout.get("points"), dtype=float)
        if pts.ndim != 2 or len(pts) < 50:
            return "rejected"
        obs = self._observed(3000)
        if len(obs) < min_samples:
            return "undecided"
        plen = _path_len(np.vstack([pts, pts[:1]]))
        tol = max(0.006 * plen, 1e-6)
        frac = float((dist_to_polyline(pts, obs) <= tol).mean())
        if frac < min_fraction:
            log.info("official layout rejected: only %.0f%% of %d observed samples within %.0f of it", frac * 100, len(obs), tol)
            return "rejected"
        poly = resample_closed(pts, self.resample)
        # driving direction: progress must increase along the polyline for the cars we have seen
        score = 0.0
        m = len(poly)
        for lst in self._samples.values():
            seq = np.array([(x, y) for x, y, _, c in lst if c])
            if len(seq) < 20:
                continue
            d = (seq[:, None, :] - poly[None, ::4, :]) ** 2
            idx = d.sum(axis=2).argmin(axis=1) * 4
            step = ((idx[1:] - idx[:-1] + m // 2) % m) - m // 2
            step = step[np.abs(step) < m // 10]
            score += float(np.sign(step).sum())
        if score < 0:
            poly = poly[::-1].copy()
        raw_len = _path_len(np.vstack([poly, poly[:1]]))
        length_m = raw_len / 10.0 if raw_len > 15000 else raw_len
        tr = Track(poly, length_m, source="official", official_rotation=layout.get("rotation"))

        def frac_of(e: Dict[str, Any]) -> float:
            return round(float(tr.project(e["x"], e["y"])[0]), 4)

        corners = [{"n": e["n"], "frac": frac_of(e), **({"letter": e["letter"]} if e.get("letter") else {})}
                   for e in layout.get("corners") or []]
        tr.corners = sorted(corners, key=lambda c: c["frac"]) or None
        tr.marshal_sectors = sorted(frac_of(e) for e in layout.get("marshal") or []) or None
        self.track = tr
        self.version += 1
        self._samples.clear()
        log.info("official circuit layout adopted (%d pts, %.0f m, %.0f%% of samples on it)", len(poly), length_m, frac * 100)
        return "accepted"

    # ---- projection ----------------------------------------------------
    def project(self, x: float, y: float, hint: Optional[int] = None):
        if self.track is None:
            return None
        return self.track.project(x, y, hint)

    def add_z(self, progress: float, z: float) -> None:
        """Feed an elevation sample; bumps `version` when the elevation profile first
        becomes available and again as it is refined, so clients get an updated track."""
        if self.track is None:
            return
        self.track.add_z(progress, z)
        self._z_n = getattr(self, "_z_n", 0) + 1
        if self._z_n in (1500, 15000, 150000):
            self.version += 1

    # ---- DRS zones / sectors ------------------------------------------
    def add_drs(self, progress: float, is_open: bool) -> None:
        if self.track is None:
            return
        b = int(progress * DRS_BINS) % DRS_BINS
        self._drs_total[b] += 1
        if is_open:
            self._drs_open[b] += 1
        self._drs_n += 1
        if self._drs_n - self._drs_last_n >= 400:
            self._drs_last_n = self._drs_n
            self._update_drs()

    def _update_drs(self) -> None:
        tot = self._drs_total
        if tot.sum() < 400 or self._drs_open.sum() < 20:
            return
        ratio = np.where(tot >= 2, self._drs_open / np.maximum(tot, 1), 0.0)
        # smooth a little
        ratio = np.convolve(np.concatenate([ratio[-3:], ratio, ratio[:3]]), np.ones(3) / 3, mode="same")[3:-3]
        peak = float(ratio.max())
        if peak <= 0:
            return
        # DRS only opens for cars within 1s of another, so the absolute ratio is low: threshold relative to the peak
        on = ratio >= max(0.2 * peak, 0.01)
        zones: List[List[float]] = []
        n = DRS_BINS
        if on.all() or not on.any():
            return
        start0 = int(np.argmin(on))  # first False index; rotate so we begin outside a zone
        i = 0
        cur = None
        for k in range(n):
            idx = (start0 + k) % n
            if on[idx] and cur is None:
                cur = idx
            if not on[idx] and cur is not None:
                zones.append([cur / n, idx / n])
                cur = None
        if cur is not None:
            zones.append([cur / n, start0 / n])
        zones = [z for z in zones if ((z[1] - z[0]) % 1.0) > 0.01]
        zones.sort()
        if zones != self.track.drs_zones:
            self.track.drs_zones = zones
            self.version += 1

    def add_sector_boundary(self, idx: int, progress: float) -> None:
        if self.track is None or idx not in (0, 1):
            return
        lst = self._sec_samples[idx]
        if len(lst) < 400:
            lst.append(progress)
        if len(lst) >= 5 and len(lst) % 5 == 0:
            self._update_sectors()

    def _update_sectors(self) -> None:
        s = []
        for lst in self._sec_samples:
            if len(lst) < 5:
                return
            a = np.array(lst)
            # circular median-ish: shift by 0.5 to avoid wrap at start/finish
            sh = (a + 0.5) % 1.0
            s.append(float((np.median(sh) - 0.5) % 1.0))
        if s[0] < s[1] and s[1] - s[0] > 0.05:
            new = [round(s[0], 3), round(s[1], 3)]
            if self.track is not None and any(abs(a - b) > 0.01 for a, b in zip(new, self.track.sectors)):
                self.track.sectors = new
                self.version += 1

    def to_dict(self) -> Optional[Dict[str, Any]]:
        return self.track.to_dict() if self.track is not None else None


# =============================================================================== official layout fallback
MULTIVIEWER_URL = "https://api.multiviewer.app/api/v1/circuits/{key}/{year}"
FETCH_TIMEOUT = 5.0


def default_fetch(url: str, timeout: float = FETCH_TIMEOUT) -> Any:
    """GET ``url`` and decode the JSON body (stdlib only).  Raises on any problem."""
    import json
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "f1-live-dashboard/1.0 (circuit layout fallback)",
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        return json.loads(resp.read(8 << 20).decode("utf-8-sig"))


class OfficialLayoutFallback:
    """Best-effort: fetch the official circuit polyline for the current circuit and let
    :meth:`TrackBuilder.offer_reference` adopt it if it matches what the cars actually drive.

    * only ever used while no track has been learned/adopted yet;
    * the HTTP request runs in a daemon thread with a 5 s timeout and never raises into the caller - every failure is
      only logged (and retried a few times, a flaky link may come up later);
    * the layout is *never trusted blindly*: ``offer_reference`` rejects it unless >= 80 % of the observed on-track
      samples lie on it, in which case we simply keep learning.

    Drive it by calling :meth:`tick` periodically (the server does, ~4 Hz) with the circuit key and year read from
    ``SessionInfo`` (``Meeting.Circuit.Key``, season from ``Path``/``StartDate``)."""

    RETRY_AFTER = (30.0, 120.0, 600.0)  # seconds before the 2nd, 3rd, 4th attempt; then give up for this circuit

    def __init__(self, fetch: Optional[Any] = None, *, timeout: float = FETCH_TIMEOUT, threaded: bool = True,
                 url: str = MULTIVIEWER_URL, clock: Any = None) -> None:
        import threading
        import time

        self._threading = threading
        self._clock = clock or time.monotonic
        self.fetch = fetch or default_fetch
        self.timeout, self.threaded, self.url = timeout, threaded, url
        self._lock = threading.Lock()
        # (key, year) -> {"state": fetching|ready|failed|accepted|rejected|gone, "layout": dict|None, "attempts": int, "next": float}
        self._jobs: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self.last_result: Optional[str] = None

    def forget(self) -> None:
        """Drop all job state (after a manual track reset a new lookup is allowed)."""
        with self._lock:
            self._jobs.clear()

    def state(self, key: Any, year: Any) -> Optional[str]:
        j = self._jobs.get((str(key), int(year)))
        return j["state"] if j else None

    def _run(self, k: Tuple[str, int]) -> None:
        job = self._jobs[k]
        try:
            data = self.fetch(self.url.format(key=k[0], year=k[1]), self.timeout)
            layout = parse_circuit_layout(data)
            with self._lock:
                if layout is None:
                    job["state"] = "gone" if not data else "failed"
                    log.info("circuit layout %s/%s unusable (%s)", k[0], k[1], "empty" if not data else "unrecognised schema")
                else:
                    job["layout"], job["state"] = layout, "ready"
        except Exception as e:  # noqa: BLE001 - silent by design
            with self._lock:
                job["state"] = "failed"
            log.info("circuit layout fetch %s/%s failed: %s: %s", k[0], k[1], type(e).__name__, e)

    def tick(self, tb: TrackBuilder, circuit_key: Any, year: Any) -> Optional[str]:
        """Advance the state machine.  Returns the :meth:`offer_reference` outcome when one was just made."""
        if tb.track is not None or circuit_key is None or year is None:
            return None
        try:
            k = (str(circuit_key), int(year))
        except (TypeError, ValueError):
            return None
        now = self._clock()
        job = self._jobs.get(k)
        if job is None or (job["state"] == "failed" and job["attempts"] < len(self.RETRY_AFTER) + 1 and now >= job["next"]):
            attempts = (job["attempts"] if job else 0) + 1
            job = self._jobs[k] = {"state": "fetching", "layout": None, "attempts": attempts,
                                   "next": now + self.RETRY_AFTER[min(attempts - 1, len(self.RETRY_AFTER) - 1)]}
            if self.threaded:
                t = self._threading.Thread(target=self._run, args=(k,), name="circuit-layout", daemon=True)
                t.start()
            else:
                self._run(k)
        if job["state"] == "ready":
            res = tb.offer_reference(job["layout"])
            if res in ("accepted", "rejected", "ignored"):
                job["state"] = "accepted" if res == "accepted" else "rejected"
                job["layout"] = None
                self.last_result = res
                return res
            return res  # undecided: keep the layout, ask again after more samples
        return None
