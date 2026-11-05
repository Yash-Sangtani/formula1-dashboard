// Fabricated feed for visual testing without a backend. Usage: /index.html?mock=1
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

export function run(store, kind = 'race') {
  const sprint = kind === 'sprint';
  const LAP = 12; // seconds per lap in the mock (compressed)
  const track = (() => {
    const pts = [];
    for (let i = 0; i < 360; i++) { const t = (i / 360) * Math.PI * 2; pts.push([Math.round(5200 * Math.cos(t) + 1400 * Math.cos(3 * t)), Math.round(2900 * Math.sin(t) + 900 * Math.sin(2 * t))]); }
    let len = 0; for (let i = 1; i < pts.length; i++) len += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    return { points: pts, length: len, bounds: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)], startIdx: 0, drsZones: [[0.02, 0.12], [0.55, 0.64]], sectors: [0.33, 0.66], rotation: 20 };
  })();
  const at = (f) => { f = ((f % 1) + 1) % 1; const i = Math.floor(f * 360), u = f * 360 - i, a = track.points[i], b = track.points[(i + 1) % 360]; return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u]; };
  const cars = []; let g = 0;
  for (const [team, col, drs] of TEAMS) for (const [num, tla, name] of drs) {
    g++; cars.push({ num, tla, name, team, color: '#' + col, grid: g, dist: -g * 0.012, pace: 1 + (Math.random() - 0.5) * 0.05 - g * 0.0015, compound: ['SOFT', 'MEDIUM', 'HARD'][g % 3],
      stintStart: 0, stint: 1, pits: 0, pitUntil: 0, ret: false, lastLap: null, best: null, bestOverall: false, sectors: [null, null, null], lastLapN: 0, lapDelta: 0 });
  }
  const rc = [], radio = []; let flag = 'green', speed = 1, paused = false, simT = 0, fl = null, tNow = Date.now() / 1000;
  const pushRc = (m) => rc.unshift({ utc: tNow + simT, lap: Math.floor(lapOf()) + 1, category: 'Flag', scope: 'Track', sector: null, num: null, ...m });
  const lapOf = () => Math.max(...cars.map((c) => c.dist));
  const events = [
    [8, () => pushRc({ flag: 'YELLOW', scope: 'Sector', sector: 2, message: 'YELLOW IN TRACK SECTOR 2' })],
    [14, () => { radio.unshift({ utc: tNow + simT, num: '44', tla: 'HAM', url: 'about:blank' }); }],
    [20, () => pushRc({ flag: 'CLEAR', scope: 'Sector', sector: 2, message: 'CLEAR IN TRACK SECTOR 2' })],
    [40, () => { flag = 'sc'; pushRc({ category: 'SafetyCar', flag: null, message: 'SAFETY CAR DEPLOYED' }); }],
    [55, () => { flag = 'green'; pushRc({ category: 'SafetyCar', flag: null, message: 'SAFETY CAR IN THIS LAP' }); }],
    [70, () => { cars[17].ret = true; pushRc({ category: 'Other', message: `CAR 3 (RIC) STOPPED ON TRACK`, num: '3' }); }],
    [80, () => { flag = 'vsc'; pushRc({ category: 'SafetyCar', message: 'VIRTUAL SAFETY CAR DEPLOYED' }); }],
    [90, () => { flag = 'green'; pushRc({ flag: 'GREEN', message: 'TRACK CLEAR' }); }],
  ];
  const evDone = new Set();
  const cm = ['SOFT', 'MEDIUM', 'HARD'];
  let order = [];

  function step(dt) {
    simT += dt;
    const slow = flag === 'sc' ? 0.55 : flag === 'vsc' ? 0.7 : 1;
    for (const c of cars) {
      if (c.ret) continue;
      const inPit = c.pitUntil > simT;
      const sp = (inPit ? 0.35 : slow) * c.pace;
      const prevLap = Math.floor(c.dist); c.dist += (dt / LAP) * sp;
      const lapNow = Math.floor(c.dist);
      c.sf = ((c.dist % 1) + 1) % 1;
      if (lapNow > prevLap && lapNow > 0) {
        const t = 90 + (Math.random() - 0.5) * 2.5 + (c.compound === 'SOFT' ? -0.6 : c.compound === 'HARD' ? 0.5 : 0) + (flag !== 'green' ? 25 : 0);
        c.lastLap = t; c.lastLapN = lapNow;
        c.sectors = [t * 0.31 + Math.random() * .3, t * 0.36 + Math.random() * .3, t * 0.33 + Math.random() * .3];
        const pb = !c.best || t < c.best; if (pb) c.best = t;
        c.lastPB = pb;
        if (!fl || t < fl.sec) { fl = { num: c.num, sec: t, text: fmt(t), lap: lapNow }; }
        if (lapNow > 0 && lapNow % (10 + (+c.num % 5)) === 0 && !c.pits) { c.pitUntil = simT + 4; c.pits++; c.stint++; c.compound = cm[(cm.indexOf(c.compound) + 1) % 3]; c.stintStart = lapNow; }
      }
      c.inPit = inPit; c.wasPit = c.pitUntil > 0 && !inPit && simT - c.pitUntil < 3;
    }
    for (const [t, fn] of events) if (simT >= t && !evDone.has(t)) { evDone.add(t); fn(); }
    order = cars.slice().sort((a, b) => (a.ret - b.ret) || b.dist - a.dist);
  }

  function stateMsg() {
    const cs = {};
    order.forEach((c, i) => {
      const lead = order[0], ahead = order[i - 1];
      const gl = (lead.dist - c.dist) * LAP * 7.5, iv = i ? (ahead.dist - c.dist) * LAP * 7.5 : 0;
      const sec = (k) => c.sectors[k] == null ? { text: '', sec: null, overall: false, personal: false } : { text: c.sectors[k].toFixed(3), sec: c.sectors[k], overall: !!(fl && fl.num === c.num && k === 1), personal: k !== 2 };
      cs[c.num] = {
        num: c.num, tla: c.tla, name: c.name, team: c.team, color: c.color, pos: i + 1, grid: c.grid, posChange: c.grid - (i + 1),
        gapLeader: i ? '+' + gl.toFixed(3) : '', interval: i ? '+' + iv.toFixed(3) : '', gapLeaderSec: gl, intervalSec: iv,
        lap: Math.floor(c.dist) + 1, pits: c.pits, inPit: !!c.inPit, pitOut: !!c.wasPit, retired: c.ret, stopped: c.ret,
        last: c.lastLap ? { text: fmt(c.lastLap), sec: c.lastLap, overall: !!(fl && fl.num === c.num && fl.sec === c.lastLap), personal: !!c.lastPB } : { text: '', sec: null, overall: false, personal: false },
        best: c.best ? { text: fmt(c.best), sec: c.best, overall: !!(fl && fl.num === c.num), personal: true } : { text: '', sec: null, overall: false, personal: false },
        sectors: [sec(0), sec(1), sec(2)], speedTraps: {},
        tyre: { compound: c.compound, age: Math.floor(c.dist - c.stintStart) + 1, new: true, stint: c.stint },
        stints: [], tel: { speed: Math.round(c.inPit ? 80 : 200 + 110 * Math.abs(Math.sin(c.dist * 12))), drs: Math.sin(c.dist * 12) > .6, rpm: 0, gear: 5, throttle: 100, brake: 0 },
        xy: (() => { const p = at(c.sf || 0); return { x: p[0], y: p[1], z: 0, status: 'OnTrack' }; })(), progress: c.sf || 0,
      };
    });
    const rem = 5400 - simT;
    return {
      type: 'state', utc: tNow + simT, session: { name: sprint ? 'Sprint' : 'Race', meeting: 'Mock Grand Prix', type: 'Race', status: 'Started', kind: sprint ? 'sprint' : 'race', part: null, partLabel: null },
      clock: { remainingSec: rem, extrapolating: true, elapsedSec: simT }, lap: { current: Math.floor(lapOf()) + 1, total: sprint ? 19 : 57 },
      trackStatus: { code: '1', label: flag, flag: flag }, weather: { air: 28.1, track: 36.4, humidity: 41, pressure: 1010, wind: 2.3, windDir: 120, rain: simT > 100 },
      order: order.map((c) => c.num), cars: cs, raceControl: rc.slice(0, 100), radio: radio.slice(0, 30),
      records: { fastestLap: fl, topSpeed: null }, pitStops: [],
    };
  }
  const fastMsg = () => {
    const out = {};
    for (const c of cars) { if (c.ret) continue; const p = at(c.sf || 0); out[c.num] = { x: Math.round(p[0]), y: Math.round(p[1]), z: 0, speed: Math.round(c.inPit ? 80 : 200 + 110 * Math.abs(Math.sin(c.dist * 12))), drs: Math.sin(c.dist * 12) > .6, progress: c.sf || 0 }; }
    return { type: 'fast', utc: tNow + simT, cars: out };
  };

  store.send = (o) => { if (o.speed) speed = o.speed; if (o.paused != null) paused = o.paused; };
  store.conn = 'open'; store.ingest({ type: 'hello', mode: 'sim', replay: true, phase: 'active', info: { speed: 1 } });
  store.conn = 'open'; store.ingest({ type: 'track', track }); store.emit('conn');
  step(0.01);
  let last = performance.now(), n = 0;
  setInterval(() => {
    const now = performance.now(), dt = (now - last) / 1000; last = now;
    if (!paused) step(dt * speed);
    const m = fastMsg();
    store.ingest(m);
    if (n++ % 4 === 0) store.ingest(stateMsg());
  }, 250);
  store.ingest(stateMsg());
  window.__mock = { step, get simT() { return simT; } };
}
