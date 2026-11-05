// Mock store (same shape/API as js/store.js) with a fabricated 20-car, 30-lap race.
import { mount as mountDD } from '../js/panels/deepdive.js';
import { mount as mountAN } from '../js/panels/analytics.js';

const P = new URLSearchParams(location.search);
let seed = +(P.get('seed') || 7);
const rnd = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
const gauss = () => (rnd() + rnd() + rnd() - 1.5) * 0.8;

const TEAMS = [['Red Bull Racing', '#3671C6'], ['Ferrari', '#E8002D'], ['Mercedes', '#27F4D2'], ['McLaren', '#FF8000'], ['Aston Martin', '#229971'],
  ['Alpine', '#FF87BC'], ['Williams', '#64C4FF'], ['RB', '#6692FF'], ['Kick Sauber', '#52E252'], ['Haas F1 Team', '#B6BABD']];
const DRV = [['1', 'VER', 'Max Verstappen', 0], ['11', 'PER', 'Sergio Perez', 0], ['16', 'LEC', 'Charles Leclerc', 1], ['55', 'SAI', 'Carlos Sainz', 1],
  ['44', 'HAM', 'Lewis Hamilton', 2], ['63', 'RUS', 'George Russell', 2], ['4', 'NOR', 'Lando Norris', 3], ['81', 'PIA', 'Oscar Piastri', 3],
  ['14', 'ALO', 'Fernando Alonso', 4], ['18', 'STR', 'Lance Stroll', 4], ['10', 'GAS', 'Pierre Gasly', 5], ['31', 'OCO', 'Esteban Ocon', 5],
  ['23', 'ALB', 'Alexander Albon', 6], ['2', 'SAR', 'Logan Sargeant', 6], ['22', 'TSU', 'Yuki Tsunoda', 7], ['3', 'RIC', 'Daniel Ricciardo', 7],
  ['77', 'BOT', 'Valtteri Bottas', 8], ['24', 'ZHO', 'Zhou Guanyu', 8], ['20', 'MAG', 'Kevin Magnussen', 9], ['27', 'HUL', 'Nico Hulkenberg', 9]];
const LAPS = 30, TOTAL = 57, SC = [12, 13, 14];
const cmp = ['SOFT', 'MEDIUM', 'HARD'];

function fabricate() {
  const cars = {}, laps = {}, events = [], pitStops = [];
  const cum = {}, retired = new Set(['20']);
  const plan = {};
  DRV.forEach(([num, tla, name, t], i) => {
    const base = 91.6 + t * 0.12 + i * 0.045 + gauss() * 0.25;
    const start = cmp[Math.floor(rnd() * 3)];
    const p1 = 12 + Math.floor(rnd() * 9);
    const two = rnd() < 0.3;
    plan[num] = { base, stops: two ? [p1 - 3, p1 + 9] : [p1], comp: [start, cmp[(cmp.indexOf(start) + 1 + Math.floor(rnd() * 2)) % 3], cmp[Math.floor(rnd() * 3)]], deg: 0.04 + rnd() * 0.06 };
    cum[num] = i * 0.35; laps[num] = [];
    cars[num] = { num, tla, name, team: TEAMS[t][0], color: TEAMS[t][1], grid: i + 1 };
  });
  let fastest = 1e9;
  for (let L = 1; L <= LAPS; L++) {
    const track = SC.includes(L) ? '4' : '1';
    for (const [num] of DRV) {
      if (retired.has(num) && L > 22) continue;
      const p = plan[num], st = p.stops.filter((s) => s < L).length;
      const age = L - (p.stops.filter((s) => s < L).pop() || 0);
      const pit = p.stops.includes(L);
      let sec = p.base + p.deg * age * (st % 2 ? 0.8 : 1) - 0.035 * L + gauss() * 0.22 + (L === 1 ? 5.5 : 0);
      if (track === '4') sec = 118 + gauss();
      if (pit) sec += 21.5 + gauss();
      cum[num] += sec;
      const s1 = sec * (0.31 + gauss() * 0.004), s2 = sec * (0.34 + gauss() * 0.004);
      laps[num].push({ lap: L, sec, s: [s1, s2, sec - s1 - s2], compound: p.comp[st], tyreAge: age, pit, track, _cum: cum[num] });
      if (pit) pitStops.push({ num, lap: L, utc: 0, durationSec: +(22 + rnd() * 2.2 + (rnd() < 0.08 ? 7 : 0)).toFixed(1), fromCompound: p.comp[st], toCompound: p.comp[st + 1] });
    }
    if (SC.includes(L) && L === SC[SC.length - 1]) {
      const ord = Object.keys(cum).filter((n) => laps[n].length === L).sort((a, b) => cum[a] - cum[b]);
      ord.forEach((n, i) => { cum[n] = cum[ord[0]] + i * 0.9; laps[n][L - 1]._cum = cum[n]; });
    }
    const ord = Object.keys(cum).filter((n) => laps[n].length === L).sort((a, b) => cum[a] - cum[b]);
    ord.forEach((n, i) => {
      const e = laps[n][L - 1];
      e.pos = i + 1; e.gapLeader = i ? +(cum[n] - cum[ord[0]]).toFixed(3) : 0; e.interval = i ? +(cum[n] - cum[ord[i - 1]]).toFixed(3) : null;
      if (e.sec < fastest && !e.pit && L > 1 && e.track === '1') { fastest = e.sec; events.push({ utc: L, lap: L, kind: 'fastest', num: n, text: `${cars[n].tla} fastest lap` }); }
      if (L > 1) { const pe = laps[n][L - 2]; if (pe.pos - e.pos >= 2 && !e.pit) events.push({ utc: L, lap: L, kind: 'overtake', num: n, text: `${cars[n].tla} up to P${e.pos}` }); }
      if (e.pit) events.push({ utc: L, lap: L, kind: 'pit', num: n, text: `${cars[n].tla} pits` });
    });
  }
  events.push({ utc: 0, lap: 1, kind: 'flag', num: null, text: 'GREEN LIGHT - PIT EXIT OPEN' });
  events.push({ utc: 11, lap: 12, kind: 'flag', num: null, text: 'YELLOW in TRACK SECTOR 7' });
  events.push({ utc: 11, lap: 12, kind: 'sc', num: null, text: 'SAFETY CAR DEPLOYED' });
  events.push({ utc: 15, lap: 15, kind: 'sc', num: null, text: 'SAFETY CAR IN THIS LAP' });
  events.push({ utc: 15, lap: 15, kind: 'flag', num: null, text: 'GREEN - RACING RESUMES' });
  events.push({ utc: 18, lap: 18, kind: 'penalty', num: '63', text: '5 SECOND TIME PENALTY FOR CAR 63 (RUS)' });
  events.push({ utc: 22, lap: 22, kind: 'retire', num: '20', text: 'MAG retired - engine' });
  events.push({ utc: 24, lap: 24, kind: 'radio', num: '1', text: 'VER team radio' });
  events.sort((a, b) => a.lap - b.lap);
  const order = Object.keys(laps).filter((n) => laps[n].length === LAPS).sort((a, b) => laps[a][LAPS - 1].pos - laps[b][LAPS - 1].pos).concat(['20']);
  const ranks = {};
  for (const [num] of DRV) {
    const arr = laps[num], last = arr[arr.length - 1], c = cars[num];
    const bestE = arr.filter((e) => !e.pit && e.lap > 1).sort((a, b) => a.sec - b.sec)[0];
    const lt = (sec) => { const m = Math.floor(sec / 60); return `${m}:${(sec - m * 60).toFixed(3).padStart(6, '0')}`; };
    const ch = (v) => ({ text: v.toFixed(3), sec: v, overall: false, personal: rnd() < 0.4 });
    Object.assign(c, {
      pos: last.pos, posChange: c.grid - last.pos, gapLeader: '+' + last.gapLeader.toFixed(3), interval: last.interval ? '+' + last.interval.toFixed(3) : '', gapLeaderSec: last.gapLeader, intervalSec: last.interval,
      lap: last.lap, pits: plan[num].stops.filter((s) => s <= LAPS).length, inPit: false, pitOut: false, retired: retired.has(num), stopped: false,
      last: { text: lt(last.sec), sec: last.sec, overall: false, personal: rnd() < 0.3 }, best: { text: lt(bestE.sec), sec: bestE.sec, overall: false, personal: true },
      sectors: last.s.map(ch), speedTraps: { i1: 280 + Math.floor(rnd() * 40), i2: 270 + Math.floor(rnd() * 40), fl: 290 + Math.floor(rnd() * 30), st: 305 + Math.floor(rnd() * 25) },
      tyre: { compound: last.compound, age: last.tyreAge, new: rnd() < 0.5, stint: plan[num].stops.filter((s) => s < LAPS).length + 1 },
      stints: (() => { const stops = plan[num].stops.filter((s) => s < LAPS), pts = [0, ...stops, LAPS]; return pts.slice(0, -1).map((a, i) => ({ compound: plan[num].comp[i], new: true, startLap: a, laps: pts[i + 1] - a })); })(),
      tel: { speed: 200, rpm: 10000, gear: 5, throttle: 80, brake: 0, drs: false }, xy: null, progress: 0.3,
    });
  }
  cars['20'].pos = 20; cars['20'].lap = 22;
  const hist = { type: 'history', maxLap: LAPS, laps: Object.fromEntries(Object.entries(laps).map(([n, a]) => [n, a.map(({ _cum, ...e }) => e)])), events };
  const state = {
    type: 'state', utc: 1, session: { name: 'Race', meeting: 'Mock GP', type: 'Race', status: 'Started' }, clock: {}, lap: { current: LAPS, total: TOTAL },
    trackStatus: { code: '1', label: 'AllClear', flag: 'green' }, weather: {}, order, cars, raceControl: [], radio: [],
    records: { fastestLap: { num: '1', sec: 91.2, text: '1:31.200', lap: 9 }, topSpeed: null }, pitStops,
  };
  return { state, hist };
}

const { state, hist } = fabricate();
const listeners = new Set();
export const store = {
  state, fast: {}, track: null, history: hist, hello: { mode: 'mock' }, conn: 'open', selected: new Set(),
  subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); },
  emit(kind) { for (const fn of listeners) { try { fn(kind); } catch (e) { console.error(e); } } },
  select(num, { additive = false } = {}) {
    if (additive) { this.selected.has(num) ? this.selected.delete(num) : this.selected.add(num); }
    else if (this.selected.size === 1 && this.selected.has(num)) this.selected.clear();
    else this.selected = new Set([num]);
    document.getElementById('sel').textContent = 'selected: ' + [...this.selected].join(',');
    this.emit('selection');
  },
  send() {},
};
const sel = P.get('sel');
if (sel) store.selected = new Set(sel.split(',').filter(Boolean));
document.getElementById('sel').textContent = 'selected: ' + [...store.selected].join(',');
document.querySelectorAll('[data-sel]').forEach((b) => b.addEventListener('click', () => { store.selected = new Set(b.dataset.sel.split(',').filter(Boolean)); document.getElementById('sel').textContent = 'selected: ' + [...store.selected].join(','); store.emit('selection'); }));

const only = P.get('only');
if (only === 'dd') document.getElementById('an-sec').remove();
if (only === 'an') document.getElementById('dd-sec').remove();
if (P.get('tab')) { try { localStorage.setItem('f1.an.tab', P.get('tab')); } catch {} }
if (P.get('empty')) { store.history = { type: 'history', maxLap: 0, laps: {}, events: [] }; if (P.get('empty') === 'nostate') store.state = null; }

const dd = document.getElementById('dd') && mountDD(document.getElementById('dd'), store);
const an = document.getElementById('an') && mountAN(document.getElementById('an'), store);
window.__h = { store, dd, an };

// live 4 Hz telemetry
let t0 = 0;
setInterval(() => {
  t0 += 0.25;
  DRV.forEach(([num], i) => {
    const ph = t0 * 0.5 + i;
    const sp = 200 + 120 * Math.sin(ph);
    const thr = Math.max(0, Math.min(100, 50 + 70 * Math.cos(ph + 0.5)));
    store.fast[num] = { speed: Math.round(sp), rpm: Math.round(8000 + 4500 * ((sp - 80) / 240)), gear: Math.max(1, Math.min(8, Math.round(sp / 42))), throttle: Math.round(thr), brake: thr < 5 ? 100 : 0, drs: sp > 290, progress: (t0 * 0.02 + i * 0.03) % 1 };
  });
  store.fastUtc = t0;
  store.emit('fast');
}, 250);
