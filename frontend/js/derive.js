// Derived data shared by several panels (cached per history object).
import { isNum } from './util.js';

const cache = new WeakMap();

/** Personal bests (sectors, lap) per car and overall sector/lap bests, from history.laps.
 *  In/out laps (pit flag) never count as timed laps. */
export function bestsOf(history) {
  if (!history || !history.laps) return { pb: {}, ob: { s: [null, null, null], lap: null }, prog: [] };
  let c = cache.get(history);
  if (c) return c;
  const pb = {}, ob = { s: [null, null, null], lap: null };
  const events = []; // [lap-order index, num, sec] for session-best progression
  for (const [num, arr] of Object.entries(history.laps)) {
    const p = pb[num] = { s: [null, null, null], lap: null, lapNo: null, n: 0 };
    for (const e of arr || []) {
      if (!e) continue;
      for (let i = 0; i < 3; i++) {
        const v = e.s && e.s[i];
        if (isNum(v) && v > 0) {
          if (p.s[i] == null || v < p.s[i]) p.s[i] = v;
          if (ob.s[i] == null || v < ob.s[i].v) ob.s[i] = { v, num };
        }
      }
      if (isNum(e.sec) && e.sec > 0 && !e.pit) {
        p.n++;
        if (p.lap == null || e.sec < p.lap) { p.lap = e.sec; p.lapNo = e.lap; }
        events.push({ lap: e.lap, num, sec: e.sec, utc: e.utc });
      }
    }
    if (p.lap != null && (ob.lap == null || p.lap < ob.lap.v)) ob.lap = { v: p.lap, num };
  }
  c = { pb, ob, events };
  cache.set(history, c);
  return c;
}

export function idealLap(p) { return p && p.s.every(isNum) ? p.s[0] + p.s[1] + p.s[2] : null; }

/** Chip descriptor for a car's on-track activity. */
export function activityOf(car) {
  const a = car.activity;
  if (car.retired || a === 'retired') return { k: 'ret', t: 'RET' };
  if (a === 'garage') return { k: 'garage', t: 'GARAGE' };
  if (a === 'pit' || (!a && car.inPit)) return { k: 'pit', t: 'PIT' };
  if (a === 'in') return { k: 'in', t: 'IN' };
  if (a === 'out' || (!a && car.pitOut)) return { k: 'out', t: 'OUT' };
  if (a === 'track') return { k: 'track', t: 'ON TRACK' };
  return car.stopped ? { k: 'ret', t: 'STOP' } : { k: 'track', t: '' };
}

/** Classify lap-history entries of a timed session as 'out' | 'in' | 'timed'. Uses e.kind when the backend provides it. */
export function classifyLaps(entries) {
  const out = []; let inRun = false;
  for (const e of entries || []) {
    const k = e && typeof e.kind === 'string' ? e.kind.toLowerCase() : '';
    if (k === 'out' || k === 'in' || k === 'timed') { out.push(k); inRun = k !== 'in'; continue; }
    if (e && e.pit) { if (!inRun) { out.push('out'); inRun = true; } else { out.push('in'); inRun = false; } }
    else { out.push('timed'); inRun = true; }
  }
  return out;
}

/** Runs (stints) of a timed session: a run starts at each OUT lap or compound change. Same shape as charts.stintsFromLaps. */
export function runsFromLaps(entries) {
  const cls = classifyLaps(entries), out = []; let cur = null;
  (entries || []).forEach((e, i) => {
    if (!cur || cls[i] === 'out' || (cur.compound && e.compound && e.compound !== cur.compound)) { cur = { compound: e.compound, laps: [], start: e.lap }; out.push(cur); }
    cur.laps.push(e); cur.end = e.lap;
  });
  return out;
}
