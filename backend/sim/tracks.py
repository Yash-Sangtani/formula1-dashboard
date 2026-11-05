"""Synthetic circuits for the race simulator.

A circuit is a closed, smooth, star-shaped polyline built from a polar radius function
``r(theta) = 1 + sum(harmonics) + sum(gaussian bumps)`` (so it can never self-intersect), stretched by
an aspect ratio, smoothed, scaled to a target length and resampled to uniform ~10 m spacing.
Geometry is held in metres; :meth:`Track.units` converts to the F1 position unit (1/10 m).

Index 0 of the polyline sits in the middle of the longest straight (start/finish line).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

STEP_M = 10.0  # polyline spacing
A_LAT = 36.0  # m/s^2 lateral grip used for the reference speed profile
A_ACC = 7.5  # m/s^2 reference acceleration
A_BRK = 35.0  # m/s^2 reference braking
V_MAX = 85.0  # m/s (306 km/h)
V_MIN = 25.0  # m/s (90 km/h)


@dataclass
class Track:
    name: str
    xs: List[float]
    ys: List[float]
    zs: List[float]
    step: float
    length: float
    curv: List[float]  # 1/m, unsigned
    v_ref: List[float]  # m/s reference speed profile
    lap_time_ref: float  # s, time to drive v_ref once around
    sector_fracs: Tuple[float, float]
    drs_zones: List[Tuple[float, float]]
    corners: List[float]  # fractions of lap of curvature peaks
    n: int = field(init=False)

    def __post_init__(self) -> None:
        self.n = len(self.xs)

    # -- geometry ------------------------------------------------------------------------------
    def point_at(self, dist_m: float) -> Tuple[float, float, float]:
        """(x, y, z) in metres at ``dist_m`` metres along the lap (wraps); lies exactly on the polyline."""
        u = (dist_m / self.step) % self.n
        i = int(u)
        f = u - i
        j = i + 1 if i + 1 < self.n else 0
        xs, ys, zs = self.xs, self.ys, self.zs
        return (
            xs[i] + (xs[j] - xs[i]) * f,
            ys[i] + (ys[j] - ys[i]) * f,
            zs[i] + (zs[j] - zs[i]) * f,
        )

    def units(self) -> List[Tuple[int, int, int]]:
        """Polyline vertices in F1 units (1/10 m), as ints."""
        return [(round(x * 10), round(y * 10), round(z * 10)) for x, y, z in zip(self.xs, self.ys, self.zs)]

    def idx(self, dist_m: float) -> int:
        return int(dist_m / self.step) % self.n

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "points": [[x, y] for x, y, _ in self.units()],
            "length": self.length,
            "sectors": list(self.sector_fracs),
            "drsZones": [list(z) for z in self.drs_zones],
            "corners": list(self.corners),
        }


# --------------------------------------------------------------------------- presets
# harmonics: (k, amplitude, phase)   bumps: (theta0, height, width)   all radii relative to 1
_PRESETS: Dict[str, dict] = {
    "bahrain-like": dict(
        length_m=5412.0,
        aspect=1.45,
        harm=[(2, 0.16, 0.3), (3, 0.11, 1.1), (5, 0.07, 2.0), (7, 0.05, 0.4), (9, 0.035, 2.6)],
        bumps=[(0.6, 0.35, 0.13), (2.2, -0.30, 0.12), (3.9, 0.32, 0.11), (5.3, -0.26, 0.12)],
        elev=(14.0, 7.0),
        sectors=(0.30, 0.65),
    ),
    "monza-like": dict(
        length_m=5793.0,
        aspect=2.7,
        harm=[(2, 0.07, 0.0), (3, 0.03, 0.9), (6, 0.03, 0.5), (8, 0.02, 1.7)],
        bumps=[(0.35, 0.30, 0.10), (3.4, -0.22, 0.10), (2.9, 0.22, 0.09), (5.9, -0.20, 0.09)],
        elev=(5.0, 2.0),
        sectors=(0.33, 0.68),
    ),
    "tight-like": dict(
        length_m=3337.0,
        aspect=1.15,
        harm=[(3, 0.13, 0.7), (4, 0.11, 2.1), (6, 0.08, 1.0), (8, 0.06, 0.2), (11, 0.03, 1.3)],
        bumps=[(0.9, 0.26, 0.12), (2.5, -0.24, 0.12), (4.1, 0.26, 0.11), (5.5, -0.22, 0.12)],
        elev=(22.0, 10.0),
        sectors=(0.28, 0.62),
    ),
}
_ALIASES = {"default": "bahrain-like", "synthetic": "bahrain-like", "bahrain": "bahrain-like", "monza": "monza-like"}
_CACHE: Dict[str, Track] = {}


def list_tracks() -> List[str]:
    return sorted(_PRESETS)


def get_track(name: str = "bahrain-like") -> Track:
    key = _ALIASES.get(name, name)
    if key not in _PRESETS:
        raise KeyError("unknown track %r (available: %s)" % (name, ", ".join(list_tracks())))
    tr = _CACHE.get(key)
    if tr is None:
        tr = _CACHE[key] = _build(key, _PRESETS[key])
    return tr


# --------------------------------------------------------------------------- construction
def _resample(xs: Sequence[float], ys: Sequence[float], zs: Sequence[float], step: float):
    """Uniform-arclength resample of a closed polyline; returns (xs, ys, zs, total_length)."""
    n = len(xs)
    cum = [0.0]
    for i in range(n):
        j = (i + 1) % n
        cum.append(cum[-1] + math.hypot(xs[j] - xs[i], ys[j] - ys[i]))
    total = cum[-1]
    m = max(8, round(total / step))
    ox: List[float] = []
    oy: List[float] = []
    oz: List[float] = []
    k = 0
    for q in range(m):
        d = total * q / m
        while cum[k + 1] < d:
            k += 1
        seg = cum[k + 1] - cum[k]
        f = (d - cum[k]) / seg if seg > 0 else 0.0
        j = (k + 1) % n
        ox.append(xs[k] + (xs[j] - xs[k]) * f)
        oy.append(ys[k] + (ys[j] - ys[k]) * f)
        oz.append(zs[k] + (zs[j] - zs[k]) * f)
    return ox, oy, oz, total


def _smooth(vals: List[float], half: int, passes: int) -> List[float]:
    n = len(vals)
    cur = vals
    w = 2 * half + 1
    for _ in range(passes):
        nxt = []
        for i in range(n):
            nxt.append(sum(cur[(i + o) % n] for o in range(-half, half + 1)) / w)
        cur = nxt
    return cur


def _curvature(xs: List[float], ys: List[float], step: float) -> List[float]:
    n = len(xs)
    out = []
    for i in range(n):
        a, b, c = (i - 1) % n, i, (i + 1) % n
        h1 = math.atan2(ys[b] - ys[a], xs[b] - xs[a])
        h2 = math.atan2(ys[c] - ys[b], xs[c] - xs[b])
        d = (h2 - h1 + math.pi) % (2 * math.pi) - math.pi
        out.append(abs(d) / step)
    return out


def _speed_profile(curv: List[float], step: float) -> List[float]:
    n = len(curv)
    v = [min(V_MAX, max(V_MIN, math.sqrt(A_LAT / max(k, 1e-6)))) for k in curv]
    for _ in range(2):  # two sweeps so the wrap-around settles
        for i in range(n - 1, -2, -1):  # braking: look ahead
            a, b = i % n, (i + 1) % n
            v[a] = min(v[a], math.sqrt(v[b] * v[b] + 2 * A_BRK * step))
        for i in range(0, n + 1):  # acceleration: look behind
            a, b = (i - 1) % n, i % n
            v[b] = min(v[b], math.sqrt(v[a] * v[a] + 2 * A_ACC * step))
    return v


def _build(name: str, spec: dict) -> Track:
    # dense polar curve
    npts = 3000
    raw_x: List[float] = []
    raw_y: List[float] = []
    for i in range(npts):
        th = 2 * math.pi * i / npts
        r = 1.0
        for k, a, ph in spec["harm"]:
            r += a * math.cos(k * th + ph)
        for t0, h, w in spec["bumps"]:
            d = (th - t0 + math.pi) % (2 * math.pi) - math.pi
            r += h * math.exp(-((d / w) ** 2))
        r = max(r, 0.25)
        raw_x.append(r * math.cos(th) * spec["aspect"])
        raw_y.append(r * math.sin(th))
    raw_z = [0.0] * npts
    target = spec["length_m"]
    # scale to the target length, resample finely, smooth (rounds off over-sharp corners), then resample
    # to the final spacing and rescale the chord length exactly
    xs, ys, zs, total = _resample(raw_x, raw_y, raw_z, 1.0)
    sc = target / total
    xs = _smooth([x * sc for x in xs], 20, 3) if False else [x * sc for x in xs]
    ys = [y * sc for y in ys]
    xs, ys, zs, total = _resample(xs, ys, zs, 4.0)
    xs = _smooth(xs, 5, 3)
    ys = _smooth(ys, 5, 3)
    xs, ys, zs, total = _resample(xs, ys, zs, STEP_M)
    n = len(xs)
    # elevation: two low harmonics over the lap (metres)
    a1, a2 = spec["elev"]
    zs = [a1 * math.sin(2 * math.pi * i / n + 0.6) + a2 * math.sin(4 * math.pi * i / n + 2.0) for i in range(n)]
    curv = _smooth(_curvature(xs, ys, STEP_M), 1, 2)
    # rotate so index 0 is the middle of the longest straight
    best_i, best_len, run_start, run = 0, 0, None, 0
    for i in range(2 * n):
        if curv[i % n] < 0.0025:
            if run_start is None:
                run_start, run = i, 0
            run += 1
            if run > best_len and run <= n:
                best_len, best_i = run, run_start + run // 2
        else:
            run_start = None
    off = best_i % n
    xs, ys, zs, curv = (lst[off:] + lst[:off] for lst in (xs, ys, zs, curv))
    # final scale to exact target length by chord sum
    chord = sum(math.hypot(xs[(i + 1) % n] - xs[i], ys[(i + 1) % n] - ys[i]) for i in range(n))
    sc = target / chord
    xs = [x * sc for x in xs]
    ys = [y * sc for y in ys]
    step = chord * sc / n
    curv = [c / sc for c in curv]
    v = _speed_profile(curv, step)
    t_ref = sum(step / ((v[i] + v[(i + 1) % n]) / 2) for i in range(n))
    # corners: local maxima of curvature, >= 120 m apart
    cand = sorted(range(n), key=lambda i: -curv[i])
    corners: List[int] = []
    for i in cand:
        if curv[i] < 1 / 220.0:
            break
        if all(min((i - j) % n, (j - i) % n) * step >= 120.0 for j in corners):
            corners.append(i)
    corners.sort()
    # DRS zones: the two longest runs near top speed
    runs: List[Tuple[int, int]] = []
    thr = 0.9 * max(v)
    i = 0
    seen = 0
    # start scanning at a point that is not inside a run
    while i < n and v[i] >= thr:
        i += 1
    start = i
    while seen < n:
        j = (start + seen) % n
        if v[j] >= thr:
            k = seen
            while k < n and v[(start + k) % n] >= thr:
                k += 1
            runs.append(((start + seen) % n, k - seen))
            seen = k
        else:
            seen += 1
    runs = [r for r in runs if r[1] * step >= 350.0]
    runs.sort(key=lambda r: -r[1])
    drs = sorted((r[0] / n, ((r[0] + r[1]) % n) / n) for r in runs[:2])
    return Track(
        name=name,
        xs=xs,
        ys=ys,
        zs=zs,
        step=step,
        length=step * n,
        curv=curv,
        v_ref=v,
        lap_time_ref=t_ref,
        sector_fracs=tuple(spec["sectors"]),  # type: ignore[arg-type]
        drs_zones=drs,
        corners=[c / n for c in corners],
    )
