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
                 sectors: Optional[List[float]] = None, drs_zones: Optional[List[List[float]]] = None):
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


class TrackBuilder:
    """Accumulates OnTrack samples; builds a `Track` from the cleanest closed lap of any car."""

    def __init__(self, resample: int = RESAMPLE_POINTS):
        self.resample = resample
        # per car: list of (x, y, lap_tag, clean)
        self._samples: Dict[str, List[Tuple[float, float, Any, bool]]] = {}
        self.track: Optional[Track] = None
        self.version = 0
        self._since_try = 0
        self._drs_open = np.zeros(DRS_BINS)
        self._drs_total = np.zeros(DRS_BINS)
        self._drs_n = 0
        self._drs_last_n = 0
        self._sec_samples: List[List[float]] = [[], []]

    # ---- sample intake -------------------------------------------------
    def add(self, num: str, x: float, y: float, lap: Any = None, clean: bool = True) -> None:
        if self.track is not None:
            return
        lst = self._samples.setdefault(num, [])
        if len(lst) < MAX_SAMPLES_PER_CAR:
            lst.append((float(x), float(y), lap, bool(clean)))
        self._since_try += 1

    def maybe_build(self, force: bool = False) -> bool:
        """Try to build; cheap to call often (rate limited). True if a track was just built."""
        if self.track is not None:
            return False
        if not force and self._since_try < 150:
            return False
        self._since_try = 0
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
            # 2. closure detection on clean-only stream
            clean_pts = np.array([(x, y) for x, y, _, c in lst if c])
            if len(clean_pts) >= MIN_RUN_POINTS:
                r = self._closure_run(_clean_dupes(clean_pts))
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

    def build(self) -> bool:
        best = None
        best_score = None
        for r in self._candidates():
            score = (round(_closure_err(r), 3), -len(r))
            if best_score is None or score < best_score:
                best, best_score = r, score
        if best is None:
            return False
        sm = smooth_closed(resample_closed(best, self.resample * 2), window=9, passes=2)
        pts = resample_closed(sm, self.resample)
        raw_len = _path_len(np.vstack([pts, pts[:1]]))
        length_m = raw_len / 10.0 if raw_len > 15000 else raw_len
        self.track = Track(pts, length_m)
        self.version += 1
        self._samples.clear()
        log.info("track built: %d pts, %.0f m", len(pts), length_m)
        return True

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
