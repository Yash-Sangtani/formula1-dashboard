import math

import numpy as np
import pytest

from backend.track import TrackBuilder


def loop(n=300, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        r = 4000 + 800 * math.sin(3 * a)
        pts.append((r * math.cos(a) + rng.normal(0, noise), 0.7 * r * math.sin(a) + rng.normal(0, noise)))
    return pts


def test_build_from_tagged_lap_and_project():
    tb = TrackBuilder()
    pts = loop(300, noise=5)
    for lap in (0, 1, 2):
        for x, y in pts:
            tb.add("1", x, y, lap=lap, clean=True)
    # lap 0 is the out-lap (unclean in real life); build uses completed runs
    assert tb.maybe_build(force=True)
    t = tb.track
    assert 600 <= t.n <= 1000
    d = tb.to_dict()
    assert d["length"] > 10000 / 10 and len(d["bounds"]) == 4 and d["startIdx"] == 0
    # projection of sample i should be ~ i/300 (arc-length parameterised, so allow slack) and monotonic
    prev = -1
    ps = []
    hint = None
    for x, y in pts[::10]:
        p, hint, dist = t.project(x, y, hint)
        ps.append(p)
        assert dist < 200
    diffs = np.diff(ps)
    assert (diffs > -1e-6).all() or (np.abs(diffs[diffs < 0]) > 0.9).sum() <= 1
    # hinted and global projection agree
    p1 = t.project(*pts[123])[0]
    p2 = t.project(*pts[123], hint=t.project(*pts[120])[1])[0]
    assert abs(p1 - p2) < 1e-3


def test_pit_laps_excluded_and_closure_detection_without_tags():
    tb = TrackBuilder()
    pts = loop(250)
    for x, y in pts:  # dirty lap (pit lane) with different geometry
        tb.add("44", x * 1.3, y, lap=1, clean=False)
    for x, y in pts + pts[:5]:
        tb.add("1", x, y, lap=None, clean=True)  # untagged: closure detection
    assert tb.maybe_build(force=True)
    assert abs(tb.track.bounds[2] - 4800) < 400  # built from the clean car, not the 1.3x one


def test_not_enough_data_returns_false_and_sets_nothing():
    tb = TrackBuilder()
    for x, y in loop(30)[:20]:
        tb.add("1", x, y, lap=0)
    assert tb.maybe_build(force=True) is False
    assert tb.to_dict() is None and tb.project(1, 2) is None


def test_drs_zones_and_sectors():
    tb = TrackBuilder()
    for lap in (0, 1):
        for x, y in loop(300):
            tb.add("1", x, y, lap=lap)
    assert tb.maybe_build(force=True)
    for rep in range(30):
        for i in range(400):
            f = i / 400
            tb.add_drs(f, 0.30 <= f < 0.40 or 0.80 <= f < 0.90)
    z = tb.track.drs_zones
    assert len(z) == 2
    assert z[0][0] == pytest.approx(0.30, abs=0.02) and z[0][1] == pytest.approx(0.40, abs=0.02)
    for _ in range(10):
        tb.add_sector_boundary(0, 0.31)
        tb.add_sector_boundary(1, 0.66)
    assert tb.track.sectors == pytest.approx([0.31, 0.66], abs=0.01)
