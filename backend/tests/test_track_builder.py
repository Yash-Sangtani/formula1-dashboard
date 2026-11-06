"""TrackBuilder in practice/qualifying conditions, reset/context, and the official-layout fallback (mocked network)."""
from __future__ import annotations

import math
import threading
import time

import numpy as np
import pytest

from backend.track import (OfficialLayoutFallback, TrackBuilder, dist_to_polyline, parse_circuit_layout)


def circuit(n=400, scale=1.0, dx=0.0, dy=0.0):
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n
        r = (4000 + 800 * math.sin(3 * a)) * scale
        out.append((r * math.cos(a) + dx, 0.7 * r * math.sin(a) + dy))
    return out


def drive(tb, num, pts, laps=(1,), lap_tag=True, clean=True, jitter=0.0, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    for lap in laps:
        for x, y in pts:
            tb.add(num, x + rng.normal(0, jitter) + shift, y + rng.normal(0, jitter), lap=lap if lap_tag else None, clean=clean)


def mv_json(pts, n_corners=6, with_marshals=True, closed=False):
    xs = [p[0] for p in pts] + ([pts[0][0]] if closed else [])
    ys = [p[1] for p in pts] + ([pts[0][1]] if closed else [])
    d = {"x": xs, "y": ys, "rotation": 92, "circuitName": "Test", "year": 2025,
         "corners": [{"number": i + 1, "letter": "", "angle": 10.0, "length": 100, "trackPosition": {"x": pts[i * 60][0], "y": pts[i * 60][1]}}
                     for i in range(n_corners)],
         "candidateLap": {"driverNumber": "1"}}
    if with_marshals:
        d["marshalSectors"] = [{"number": i + 1, "angle": 0, "length": 0, "trackPosition": {"x": pts[i * 100][0], "y": pts[i * 100][1]}} for i in range(4)]
    return d


# ----------------------------------------------------------------------------- reset / context
def test_reset_forgets_track_and_samples_and_bumps_version():
    tb = TrackBuilder()
    drive(tb, "1", circuit(), laps=(1, 2))
    assert tb.maybe_build(force=True) and tb.track is not None
    v = tb.version
    tb.reset()
    assert tb.track is None and tb.to_dict() is None and tb.version > v and tb.resets == 1
    assert tb.project(0, 0) is None
    # and it learns a different circuit afterwards
    drive(tb, "1", circuit(scale=0.5), laps=(1, 2))
    assert tb.maybe_build(force=True)
    assert tb.track.bounds[2] < 2600


def test_set_context_resets_only_on_a_different_circuit():
    tb = TrackBuilder()
    assert tb.set_context(63, 2025) is False
    drive(tb, "1", circuit(), laps=(1, 2))
    assert tb.set_context("63", 2025) is False  # same circuit (int vs str) -> keep
    assert tb._samples
    assert tb.set_context(10, 2025) is True
    assert tb.track is None and not tb._samples and tb.circuit_key == 10
    assert tb.set_context(None) is False and tb.circuit_key == 10  # unknown never resets


# ----------------------------------------------------------------------------- practice / quali conditions
def test_garage_cars_zero_fixes_and_stationary_samples_do_not_poison_or_fill_the_cap():
    tb = TrackBuilder()
    for _ in range(20000):  # parked car, flagged clean by mistake: must not consume the per-car sample cap
        tb.add("63", 1234.0, -567.0, lap=0, clean=True)
    assert len(tb._samples["63"]) == 1
    for _ in range(50):
        tb.add("5", 0.0, 0.0, lap=0)  # no-fix
        tb.add("5", float("nan"), 1.0, lap=0)
    assert "5" not in tb._samples
    drive(tb, "1", circuit(), laps=(1, 2))
    assert tb.maybe_build(force=True)
    assert 3500 < tb.track.bounds[2] < 5000


def test_single_car_out_lap_then_flying_lap_without_lap_tags():
    tb = TrackBuilder()
    pit_exit_road = [(4500 + i * 3, -200 - i * 2) for i in range(60)]  # pit exit joins the track, never driven again
    drive(tb, "44", pit_exit_road, lap_tag=False)
    drive(tb, "44", circuit(), laps=(1, 2), lap_tag=False)
    assert tb.maybe_build(force=True)
    assert abs(tb.track.bounds[2] - 4800) < 500


def test_pit_lane_detour_lap_loses_against_the_lap_most_cars_drove():
    tb = TrackBuilder()
    pts = circuit()
    detour = pts[:150] + [(x * 0.6, y * 0.6) for x, y in pts[150:250]] + pts[250:] + pts[:1]  # cuts through the pit lane
    drive(tb, "1", detour, laps=(1,), clean=True)       # same closure quality, "clean" per timing
    for num, seed in (("2", 1), ("3", 2), ("4", 3)):
        drive(tb, num, pts, laps=(1,), jitter=8, seed=seed)
    assert tb.maybe_build(force=True)
    d = dist_to_polyline(tb.track.pts, np.array(pts))
    assert float(np.median(d)) < 100 and float(np.percentile(d, 95)) < 250


def test_rebuild_attempts_back_off_and_stay_cheap_while_no_lap_is_complete():
    tb = TrackBuilder()
    t0 = time.perf_counter()
    for i, (x, y) in enumerate(circuit(400)[:300]):  # part of a lap only
        tb.add("1", x, y, lap=1)
        tb.maybe_build()
    assert tb.track is None
    assert tb._try_every > 150
    assert time.perf_counter() - t0 < 2


def test_to_dict_has_source_and_no_optional_fields_for_learned_tracks():
    tb = TrackBuilder()
    drive(tb, "1", circuit(), laps=(1, 2))
    tb.maybe_build(force=True)
    d = tb.to_dict()
    assert d["source"] == "learned" and "corners" not in d and "marshalSectors" not in d


# ----------------------------------------------------------------------------- official layout
def test_parse_circuit_layout_is_lenient():
    pts = circuit(400)
    lay = parse_circuit_layout(mv_json(pts, closed=True))
    assert len(lay["points"]) == 400  # explicit closing point dropped
    assert [c["n"] for c in lay["corners"]] == [1, 2, 3, 4, 5, 6] and len(lay["marshal"]) == 4 and lay["rotation"] == 92.0
    assert parse_circuit_layout(None) is None
    assert parse_circuit_layout({}) is None
    assert parse_circuit_layout({"x": [1, 2], "y": [1, 2]}) is None
    assert parse_circuit_layout({"x": [0] * 60, "y": [0] * 59}) is None
    odd = mv_json(pts)
    odd["corners"] = [{"number": "x", "trackPosition": {"x": "n/a"}}, "junk", {"trackPosition": {"x": 1, "y": 2}}]
    odd["marshalSectors"] = None
    odd["x"] = [str(v) for v in odd["x"]]  # numeric strings tolerated
    odd["y"] = [str(v) for v in odd["y"]]
    lay = parse_circuit_layout(odd)
    assert lay is not None and len(lay["corners"]) == 1 and lay["marshal"] == []


def test_offer_reference_accepts_matching_layout_and_exposes_corners_and_sectors():
    tb = TrackBuilder()
    pts = circuit(400)
    for num, seed in (("1", 1), ("2", 2)):
        drive(tb, num, pts[: 250], laps=(1,), jitter=10, seed=seed)  # only a part of a lap: no learned track possible
    assert tb.maybe_build(force=True) is False
    lay = parse_circuit_layout(mv_json(pts))
    assert tb.offer_reference(lay) == "accepted"
    t = tb.track
    assert t.source == "official" and tb.version > 0
    d = tb.to_dict()
    assert d["source"] == "official" and d["officialRotation"] == 92.0
    assert [c["n"] for c in d["corners"]] == [1, 2, 3, 4, 5, 6]
    assert all(0 <= c["frac"] <= 1 for c in d["corners"]) and len(d["marshalSectors"]) == 4
    assert d["marshalSectors"] == sorted(d["marshalSectors"])
    # the corner placed at pts[60] projects where the track point nearest to it is
    prog = t.project(*pts[60])[0]
    assert any(abs(c["frac"] - prog) < 0.01 for c in d["corners"])
    # once a track exists the builder stops collecting and later offers are ignored
    assert tb.offer_reference(lay) == "ignored"


def test_offer_reference_rejects_wrong_circuit_and_keeps_learning():
    tb = TrackBuilder()
    pts = circuit(400)
    drive(tb, "1", pts[:300], laps=(1,), jitter=5)
    other = circuit(400, scale=1.4, dx=3000, dy=-1500)
    assert tb.offer_reference(parse_circuit_layout(mv_json(other))) == "rejected"
    assert tb.track is None and tb._samples  # nothing changed; learning continues
    drive(tb, "1", pts, laps=(2, 3), jitter=5)
    assert tb.maybe_build(force=True) and tb.track.source == "learned"


def test_offer_reference_threshold_is_80_percent():
    pts = circuit(400)

    def run(frac_on):
        tb = TrackBuilder()
        n_on = int(300 * frac_on)
        drive(tb, "1", pts[:n_on], laps=(1,), jitter=3)
        for i in range(300 - n_on):  # samples far from the layout (e.g. a different road)
            tb.add("1", 20000 + i * 5, 20000 + i * 7, lap=1)
        return tb.offer_reference(parse_circuit_layout(mv_json(pts)))

    assert run(0.9) == "accepted"
    assert run(0.7) == "rejected"


def test_offer_reference_undecided_without_enough_samples():
    tb = TrackBuilder()
    drive(tb, "1", circuit()[:40])
    assert tb.offer_reference(parse_circuit_layout(mv_json(circuit(400)))) == "undecided"
    assert tb.track is None


def test_offer_reference_reverses_polyline_to_match_driving_direction():
    tb = TrackBuilder()
    pts = circuit(400)
    drive(tb, "1", pts[:300], laps=(1,), jitter=3)  # driven in increasing-angle direction
    rev = pts[::-1]  # layout listed the other way round
    assert tb.offer_reference(parse_circuit_layout(mv_json(rev))) == "accepted"
    t = tb.track
    prog = [t.project(x, y)[0] for x, y in pts[10:300:10]]
    steps = np.diff(prog)
    assert (steps > 0).mean() > 0.95


def test_progress_after_official_layout_feeds_engine_style_projection():
    tb = TrackBuilder()
    pts = circuit(400)
    drive(tb, "1", pts[:200], laps=(1,), jitter=5)
    assert tb.offer_reference(parse_circuit_layout(mv_json(pts))) == "accepted"
    p, idx, dist = tb.project(*pts[100])
    assert 0 <= p < 1 and dist < 120


# ----------------------------------------------------------------------------- fallback fetcher (mocked network)
def make_tb_with_samples():
    tb = TrackBuilder()
    drive(tb, "1", circuit()[:300], laps=(1,), jitter=5)
    return tb


def test_fallback_fetches_in_thread_once_and_adopts_layout():
    calls = []
    release = threading.Event()

    def fetch(url, timeout):
        calls.append((url, timeout))
        release.wait(5)
        return mv_json(circuit(400))

    fb = OfficialLayoutFallback(fetch)
    tb = make_tb_with_samples()
    assert fb.tick(tb, 63, 2025) is None  # starts the thread, does not block
    assert fb.tick(tb, 63, 2025) is None
    release.set()
    for _ in range(200):
        res = fb.tick(tb, 63, 2025)
        if res:
            break
        time.sleep(0.01)
    assert res == "accepted" and tb.track.source == "official"
    assert calls == [("https://api.multiviewer.app/api/v1/circuits/63/2025", 5.0)]
    assert fb.tick(tb, 63, 2025) is None and len(calls) == 1  # nothing more once a track exists


def test_fallback_waits_for_samples_then_decides():
    fb = OfficialLayoutFallback(lambda u, t: mv_json(circuit(400)), threaded=False)
    tb = TrackBuilder()
    assert fb.tick(tb, 63, 2025) == "undecided"
    assert fb.tick(tb, 63, 2025) == "undecided"
    drive(tb, "1", circuit()[:300], laps=(1,), jitter=5)
    assert fb.tick(tb, 63, 2025) == "accepted"


def test_fallback_rejects_wrong_layout_once_and_is_silent():
    n = []
    fb = OfficialLayoutFallback(lambda u, t: n.append(1) or mv_json(circuit(400, scale=2.0, dx=9000)), threaded=False)
    tb = make_tb_with_samples()
    assert fb.tick(tb, 63, 2025) == "rejected"
    assert fb.tick(tb, 63, 2025) is None and len(n) == 1 and tb.track is None


@pytest.mark.parametrize("failure", [
    lambda u, t: (_ for _ in ()).throw(TimeoutError("timed out")),
    lambda u, t: (_ for _ in ()).throw(OSError("network blocked")),
    lambda u, t: None,                         # 404 -> empty
    lambda u, t: {"error": "not found"},       # unknown schema
    lambda u, t: "<html>",                     # not JSON object
    lambda u, t: {"x": ["a"] * 80, "y": [1] * 80},
])
def test_fallback_failures_are_silent_and_leave_the_builder_untouched(failure):
    fb = OfficialLayoutFallback(failure, threaded=False)
    tb = make_tb_with_samples()
    before = (tb.version, dict(tb._samples).keys())
    assert fb.tick(tb, 63, 2025) is None
    assert tb.track is None and (tb.version, dict(tb._samples).keys()) == before


def test_fallback_retries_after_failure_with_backoff():
    now = [0.0]
    seq = iter([OSError("down"), mv_json(circuit(400))])

    def fetch(u, t):
        v = next(seq)
        if isinstance(v, Exception):
            raise v
        return v

    fb = OfficialLayoutFallback(fetch, threaded=False, clock=lambda: now[0])
    tb = make_tb_with_samples()
    assert fb.tick(tb, 63, 2025) is None and fb.state(63, 2025) == "failed"
    now[0] = 10.0
    assert fb.tick(tb, 63, 2025) is None and fb.state(63, 2025) == "failed"  # too early
    now[0] = 31.0
    assert fb.tick(tb, 63, 2025) == "accepted"


def test_fallback_needs_circuit_key_and_year_and_ignores_garbage():
    fb = OfficialLayoutFallback(lambda u, t: 1 / 0, threaded=False)
    tb = make_tb_with_samples()
    assert fb.tick(tb, None, 2025) is None and fb.tick(tb, 63, None) is None and fb.tick(tb, 63, "abc") is None
    assert fb._jobs == {}


def test_default_fetch_uses_5s_timeout_and_decodes_json(monkeypatch):
    import io
    import urllib.request

    from backend.track import default_fetch

    seen = {}

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen["url"], seen["timeout"] = req.full_url, timeout
        return R(b'\xef\xbb\xbf{"x": [1], "y": [2]}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert default_fetch("https://api.multiviewer.app/api/v1/circuits/1/2025", 5.0) == {"x": [1], "y": [2]}
    assert seen == {"url": "https://api.multiviewer.app/api/v1/circuits/1/2025", "timeout": 5.0}
