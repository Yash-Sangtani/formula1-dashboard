// Fabricated practice / qualifying feed in the CONTRACT §6 shapes. Usage: index.html?mock=fp|quali|sprint_quali
// Goes through store.ingest() so the normalisation layer is exercised exactly like the WebSocket path.
const TEAMS = [
  ['Red Bull Racing', '3671C6', [['1', 'VER', 'Max Verstappen'], ['11', 'PER', 'Sergio Perez']]],
  ['Ferrari', 'E8002D', [['16', 'LEC', 'Charles Leclerc'], ['55', 'SAI', 'Carlos Sainz']]],
  ['Mercedes', '27F4D2', [['44', 'HAM', 'Lewis Hamilton'], ['63', 'RUS', 'George Russell']]],
  ['McLaren', 'FF8000', [['4', 'NOR', 'Lando Norris'], ['81', 'PIA', 'Oscar Piastri']]],
  ['Aston Martin', '229971', [['14', 'ALO', 'Fernando Alonso'], ['18', 'STR', 'Lance Stroll']]],
  ['Alpine', 'FF87BC', [['10', 'GAS', 'Pierre Gasly'], ['31', 'OCO', 'Esteban Ocon']]],
  ['Williams', '64C4FF', [['23', 'ALB', 'Alexander Albon'], ['2', 'SAR', 'Logan Sargeant']]],
  ['RB', '6692FF', [['22', 'TSU', 'Yuki Tsunoda'], ['3', 'RIC', 'Daniel Ricciardo']]],
  ['Kick Sauber', '52E252', [['77', 'BOT', 'Valtteri Bottas'], ['24', 'ZHO', 'Guanyu Zhou']]],
  ['Haas F1 Team', 'B6BABD', [['20', 'MAG', 'Kevin Magnussen'], ['27', 'HUL', 'Nico Hulkenberg']]],
];
const fmt = (s) => { const m = Math.floor(s / 60); return `${m}:${(s - m * 60).toFixed(3).padStart(6, '0')}`; };
const sgn = (v) => '+' + v.toFixed(3);

export function run(store, kind = 'fp', opts = {}) {
  const quali = kind === 'quali' || kind === 'sprint_quali';
  const LAP = opts.lap || 10;                          // seconds per lap (compressed)
  const PARTS = quali ? [{ len: 80, keep: 15 }, { len: 70, keep: 10 }, { len: 60, keep: 10 }] : [{ len: 600, keep: 99 }];
  const track = (() => {
    const pts = [];
    for (let i = 0; i < 360; i++) { const t = (i / 360) * Math.PI * 2; pts.push([Math.round(5200 * Math.cos(t) + 1400 * Math.cos(3 * t)), Math.round(2900 * Math.sin(t) + 900 * Math.sin(2 * t))]); }
    let len = 0; for (let i = 1; i < pts.length; i++) len += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    return { points: pts, length: len, bounds: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)], startIdx: 0, drsZones: [[0.02, 0.12]], sectors: [0.33, 0.66], rotation: 20 };
  })();
  const at = (f) => { f = ((f % 1) + 1) % 1; const i = Math.floor(f * 360), u = f * 360 - i, a = track.points[i], b = track.points[(i + 1) % 360]; return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u]; };
  const cars = []; let g = 0;
  for (const [team, col, drs] of TEAMS) for (const [num, tla, name] of drs) {
    g++;
    cars.push({
      num, tla, name, team, color: '#' + col, pace: 90 + g * 0.09 + Math.random() * 0.4, mode: 'garage', modeT: 1 + Math.random() * 25, f: 0, lapsDone: 0,
      timed: 0, best: null, bestS: [null, null, null], bestLaps: [null, null, null], last: null, sectors: [null, null, null], laps: [], lapNo: 0, compound: quali ? 'SOFT' : ['MEDIUM', 'HARD', 'SOFT'][g % 3],
      tyreAge: 0, stint: 1, ko: false, pushLeft: 0, since: 0, isOutLap: false, isInLap: false,
    });
  }
  let simT = 0, part = 0, partT = 0, speed = 1, paused = false, status = 'Started', redUntil = 0, betweenUntil = 0;
  const tNow = Date.now() / 1000;
  const rc = [], events = [], radio = [];
  const pushRc = (m) => rc.unshift({ utc: tNow + simT, lap: null, category: 'Other', scope: 'Track', sector: null, num: null, flag: null, ...m });
  let fastest = null;
  const teamOrder = () => cars.slice().sort((a, b) => (a.best == null) - (b.best == null) || a.best - b.best || (+a.num) - (+b.num));
  let prevP1 = null;

  function lapDone(c) {
    c.lapNo++; const stage = c.isOutLap ? 'out' : c.isInLap ? 'in' : 'timed';
    const entry = { lap: c.lapNo, sec: null, s: [null, null, null], compound: c.compound, tyreAge: c.tyreAge, pos: null, gapLeader: null, interval: null, pit: stage !== 'timed', track: '1' };
    c.tyreAge++;
    if (stage === 'timed') {
      const evo = quali ? -0.25 * (part + partT / PARTS[part].len) : -0.15 * simT / 600;
      const t = c.pace + evo + (Math.random() - 0.5) * 0.7 + (c.compound === 'SOFT' ? -0.3 : c.compound === 'HARD' ? 0.45 : 0);
      const s = [t * 0.31 + Math.random() * 0.1, t * 0.36 + Math.random() * 0.1, 0]; s[2] = t - s[0] - s[1];
      entry.sec = t; entry.s = s; c.timed++; c.last = t; c.sectors = s.slice();
      for (let i = 0; i < 3; i++) if (c.bestS[i] == null || s[i] < c.bestS[i]) c.bestS[i] = s[i];
      if (c.best == null || t < c.best) c.best = t;
      if (quali) { const bi = c.bestLaps[part]; if (bi == null || t < bi) c.bestLaps[part] = t; }
      if (!fastest || t < fastest.sec) fastest = { num: c.num, sec: t, text: fmt(t), lap: c.lapNo };
      c.pushLeft--;
      if (c.pushLeft <= 0) { c.isInLap = true; }
    } else if (stage === 'out') { c.isOutLap = false; c.pushLeft = quali ? 1 : 2 + Math.floor(Math.random() * 6); }
    else { c.isInLap = false; c.mode = 'pit'; c.modeT = 3; c.f = 0; }
    c.laps.push(entry);
  }

  function step(dt) {
    simT += dt;
    if (simT < redUntil) { status = 'Aborted'; } else if (status === 'Aborted') { status = 'Started'; pushRc({ flag: 'GREEN', category: 'Flag', message: 'GREEN LIGHT - PIT EXIT OPEN' }); }
    if (quali && simT < betweenUntil) { status = 'Finished'; return; }
    if (quali && status === 'Finished' && simT >= betweenUntil) { status = 'Started'; }
    if (status === 'Aborted') return;
    partT += dt;
    if (quali && partT > PARTS[part].len && part < 2) {
      // end of part: eliminate
      const ord = teamOrder(); const keep = PARTS[part].keep;
      ord.forEach((c, i) => { if (i >= keep) c.ko = true; });
      part++; partT = 0; betweenUntil = simT + 6; status = 'Finished';
      pushRc({ category: 'Other', message: `QUALIFYING PART ${part} HAS STARTED` });
      for (const c of cars) if (!c.ko) { c.mode = 'garage'; c.modeT = 1 + Math.random() * 8; c.pushLeft = 0; c.isInLap = false; c.isOutLap = false; }
      return;
    }
    if (!quali && partT > PARTS[0].len) partT = PARTS[0].len;
    if (quali && part === 2 && partT > PARTS[2].len) { partT = PARTS[2].len; status = 'Finished'; }
    for (const c of cars) {
      if (c.ko) { c.mode = 'garage'; continue; }
      if (status === 'Finished') continue;
      c.modeT -= dt;
      if (c.mode === 'garage' && c.modeT <= 0) { c.mode = 'out'; c.f = 0; c.isOutLap = true; c.since = simT; if (!quali) c.compound = ['MEDIUM', 'HARD', 'SOFT'][Math.floor(Math.random() * 3)]; c.tyreAge = 0; }
      else if (c.mode === 'pit' && c.modeT <= 0) { c.mode = 'garage'; c.modeT = 5 + Math.random() * 25; }
      else if (c.mode === 'out') {
        const lapT = c.isOutLap ? LAP * 1.4 : c.isInLap ? LAP * 1.35 : LAP * (c.pace / 90);
        const f0 = c.f; c.f += dt / lapT;
        if (c.f >= 1) { c.f -= 1; lapDone(c); if (c.mode === 'out' && quali && !c.isInLap && !c.isOutLap && c.pushLeft > 0) { /* continue */ } }
        void f0;
      }
    }
    const p1 = teamOrder()[0]; if (p1 && p1.best != null && p1.num !== prevP1) { if (prevP1) events.push({ utc: tNow + simT, lap: null, kind: 'fastest', num: p1.num, text: `${p1.tla} P1 ${fmt(p1.best)}` }); prevP1 = p1.num; }
    if (!quali && simT > 40 && !step.rcd) { step.rcd = 1; redUntil = simT + 6; pushRc({ flag: 'RED', category: 'Flag', message: 'RED FLAG' }); }
  }

  function stateMsg() {
    const ord = quali ? teamOrder().sort((a, b) => (a.ko - b.ko) || ((a.best == null) - (b.best == null)) || a.best - b.best) : teamOrder();
    const lead = ord[0];
    const cs = {};
    const keep = quali ? PARTS[Math.min(part, 2)].keep : 99;
    ord.forEach((c, i) => {
      const gap = c.best != null && lead.best != null ? c.best - lead.best : null;
      const ahead = ord[i - 1], iv = c.best != null && ahead && ahead.best != null ? c.best - ahead.best : null;
      const act = c.ko || c.mode === 'garage' ? 'garage' : c.mode === 'pit' ? 'pit' : c.isOutLap ? 'out' : 'track';
      const cell = (v, o) => v == null ? { text: null, sec: null, overall: false, personal: false } : { text: v.toFixed(3), sec: v, ...o };
      cs[c.num] = {
        num: c.num, tla: c.tla, name: c.name, team: c.team, color: c.color, pos: c.best == null ? null : i + 1, grid: i + 1, posChange: 0,
        gapLeader: gap == null ? '' : i ? sgn(gap) : '', interval: iv == null ? '' : sgn(iv), gapLeaderSec: gap, intervalSec: iv,
        gapFastest: gap == null ? '' : i ? sgn(gap) : '', gapFastestSec: gap, knockedOut: c.ko, cutoff: quali && !c.ko && i >= keep - 1 && i < keep + 1 ? true : false,
        bestLaps: c.bestLaps.map((v) => v == null ? { text: null, sec: null } : { text: fmt(v), sec: v }), activity: act, timedLaps: c.timed,
        lap: c.lapNo, pits: 0, inPit: act === 'pit', pitOut: false, retired: false, stopped: false,
        last: c.last ? { text: fmt(c.last), sec: c.last, overall: !!(fastest && fastest.sec === c.last), personal: c.last === c.best } : { text: null, sec: null, overall: false, personal: false },
        best: c.best ? { text: fmt(c.best), sec: c.best, overall: !!(lead === c), personal: true } : { text: null, sec: null, overall: false, personal: false },
        sectors: c.sectors.map((v, k) => cell(v, { overall: v != null && v === Math.min(...cars.map((x) => x.bestS[k] ?? 1e9)), personal: v != null && v === c.bestS[k] })),
        speedTraps: { st: Math.round(300 + c.pace * -0.1 + c.num * 0.3) }, tyre: { compound: c.compound, age: c.tyreAge, new: c.tyreAge < 2, stint: c.stint }, stints: [],
        tel: { speed: act === 'garage' ? 0 : Math.round(150 + 150 * Math.abs(Math.sin(c.f * 18))), rpm: 10000, gear: 5, throttle: 80, brake: 0, drs: false },
        xy: act === 'garage' ? null : (() => { const p = at(c.f); return { x: p[0], y: p[1], z: 0, status: 'OnTrack' }; })(), progress: act === 'garage' ? null : c.f,
      };
    });
    const len = PARTS[Math.min(part, PARTS.length - 1)].len;
    const rem = Math.max(0, len - partT);
    const names = { fp: 'Practice 1', quali: 'Qualifying', sprint_quali: 'Sprint Qualifying' };
    return {
      type: 'state', utc: tNow + simT,
      session: { name: names[kind] || 'Practice 1', meeting: 'Mock Grand Prix', type: quali ? 'Qualifying' : 'Practice', status, kind: kind === 'fp' ? 'practice' : kind === 'quali' ? 'qualifying' : 'sprint_qualifying', part: quali ? part + 1 : null, partLabel: quali ? 'Q' + (part + 1) : null },
      clock: { remainingSec: rem, extrapolating: status === 'Started', elapsedSec: null }, lap: { current: null, total: null },
      trackStatus: status === 'Aborted' ? { code: '5', label: 'Red', flag: 'red' } : { code: '1', label: 'AllClear', flag: 'green' },
      weather: { air: 24.1, track: 33.0, humidity: 48, pressure: 1011, wind: 1.8, windDir: 200, rain: false },
      order: ord.map((c) => c.num), cars: cs, raceControl: rc.slice(0, 100), radio: radio.slice(0, 30),
      records: { fastestLap: fastest, topSpeed: null }, pitStops: [],
    };
  }
  function historyMsg() {
    const ord = teamOrder(); const laps = {};
    for (const c of cars) laps[c.num] = c.laps.map((e) => ({ ...e, pos: ord.indexOf(c) + 1 }));
    return { type: 'history', maxLap: Math.max(0, ...cars.map((c) => c.lapNo)), laps, events: events.slice(-500) };
  }
  const fastMsg = () => {
    const out = {};
    for (const c of cars) { if (c.mode !== 'out' || c.ko) continue; const p = at(c.f); out[c.num] = { x: Math.round(p[0]), y: Math.round(p[1]), z: 0, speed: Math.round(150 + 150 * Math.abs(Math.sin(c.f * 18))), progress: c.f, throttle: 80, brake: 0, rpm: 10000, gear: 5, drs: false }; }
    return { type: 'fast', utc: tNow + simT, cars: out };
  };
  store.send = (o) => { if (o.speed) speed = o.speed; if (o.paused != null) paused = o.paused; };
  store.conn = 'open';
  store.ingest({ type: 'hello', mode: 'sim', replay: true, phase: 'active', info: { speed: 1 } });
  store.conn = 'open'; store.ingest({ type: 'track', track }); store.emit('conn');
  const warm = opts.warm == null ? 8 : opts.warm; for (let i = 0; i < warm * 4; i++) step(0.25);
  let last = performance.now(), n = 0;
  const tick = () => {
    const now = performance.now(), dt = Math.min(1, (now - last) / 1000); last = now;
    if (!paused) step(dt * speed);
    store.ingest(fastMsg());
    if (n++ % 4 === 0) { store.ingest(stateMsg()); store.ingest(historyMsg()); }
  };
  setInterval(tick, 250);
  store.ingest(stateMsg()); store.ingest(historyMsg());
  window.__mock = { step: (s, k = 4) => { for (let i = 0; i < k; i++) step(s / k); tick(); }, get simT() { return simT; } };
}
