// Shared client store: WebSocket connection + latest server messages + selection.
// All incoming frames are normalised here so panels can rely on types (numbers are number|null, objects are objects).
const listeners = new Set();
const MAX_FAST = 64, MAX_RC = 200, MAX_RADIO = 60, MAX_EVENTS = 500, MAX_LAPS_PER_CAR = 600, MAX_PITS = 300;

// ------------------------------------------------------------------ normalisation helpers
const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const num = (v) => {
  if (typeof v === 'number') return isFinite(v) ? v : null;
  if (typeof v === 'string' && v.trim() !== '') { const n = +v; return isFinite(n) ? n : null; }
  return null;
};
const str = (v) => (v == null || (typeof v === 'number' && !isFinite(v)) ? '' : String(v));
const bool = (v) => v === true || v === 'true' || v === 1;
const arr = (v) => (Array.isArray(v) ? v : []);
const cell = (c) => {
  if (!isObj(c)) return { text: '', sec: null, overall: false, personal: false };
  return { text: str(c.text), sec: num(c.sec), overall: bool(c.overall), personal: bool(c.personal) };
};
const ACTIVITY = ['track', 'pit', 'out', 'garage', 'retired'];

function normColor(c) {
  c = str(c);
  if (/^#[0-9a-f]{6}$/i.test(c) || /^#[0-9a-f]{3}$/i.test(c)) return c;
  if (/^[0-9a-f]{6}$/i.test(c)) return '#' + c;
  return '#888888';
}

export function normCar(k, c) {
  if (!isObj(c)) return null;
  const o = { ...c };
  o.num = str(c.num != null ? c.num : k);
  o.tla = str(c.tla) || o.num; o.name = str(c.name); o.team = str(c.team); o.color = normColor(c.color);
  for (const f of ['pos', 'grid', 'lap', 'timedLaps']) o[f] = num(c[f]);
  o.pits = num(c.pits) || 0; o.posChange = num(c.posChange) || 0;
  for (const f of ['gapLeader', 'interval', 'gapFastest']) o[f] = str(c[f]);
  for (const f of ['gapLeaderSec', 'intervalSec', 'gapFastestSec', 'progress']) o[f] = num(c[f]);
  for (const f of ['inPit', 'pitOut', 'retired', 'stopped', 'knockedOut', 'cutoff']) o[f] = bool(c[f]);
  o.last = cell(c.last); o.best = cell(c.best);
  const sc = arr(c.sectors); o.sectors = [cell(sc[0]), cell(sc[1]), cell(sc[2])];
  const bl = arr(c.bestLaps); o.bestLaps = [0, 1, 2].map((i) => { const b = bl[i]; return isObj(b) ? { text: str(b.text), sec: num(b.sec) } : { text: '', sec: null }; });
  const st = isObj(c.speedTraps) ? c.speedTraps : {}; o.speedTraps = { i1: num(st.i1), i2: num(st.i2), fl: num(st.fl), st: num(st.st) };
  const ty = isObj(c.tyre) ? c.tyre : {};
  o.tyre = { compound: str(ty.compound), age: num(ty.age), new: typeof ty.new === 'boolean' ? ty.new : undefined, stint: num(ty.stint) };
  o.stints = arr(c.stints).filter(isObj).slice(-30).map((s) => ({ compound: str(s.compound), new: typeof s.new === 'boolean' ? s.new : undefined, startLap: num(s.startLap) || 0, laps: num(s.laps) || 0 }));
  const t = isObj(c.tel) ? c.tel : {};
  o.tel = { speed: num(t.speed), rpm: num(t.rpm), gear: num(t.gear), throttle: num(t.throttle), brake: num(t.brake), drs: bool(t.drs), drsRaw: num(t.drsRaw) };
  const xy = isObj(c.xy) ? c.xy : null;
  o.xy = xy && num(xy.x) != null && num(xy.y) != null ? { x: num(xy.x), y: num(xy.y), z: num(xy.z) || 0, status: str(xy.status) } : null;
  o.activity = ACTIVITY.includes(c.activity) ? c.activity : null;
  return o;
}

export function normState(m) {
  if (!isObj(m)) return null;
  const s = { ...m };
  s.utc = num(m.utc);
  const se = isObj(m.session) ? m.session : {};
  s.session = { ...se, name: str(se.name), meeting: str(se.meeting), type: str(se.type), status: str(se.status), kind: str(se.kind) || undefined, part: num(se.part), partLabel: str(se.partLabel) || null };
  const cl = isObj(m.clock) ? m.clock : {};
  s.clock = { remainingSec: num(cl.remainingSec), elapsedSec: num(cl.elapsedSec), extrapolating: cl.extrapolating !== false };
  const lp = isObj(m.lap) ? m.lap : {};
  s.lap = { current: num(lp.current), total: num(lp.total) };
  const ts = isObj(m.trackStatus) ? m.trackStatus : {};
  s.trackStatus = { code: str(ts.code), label: str(ts.label), flag: str(ts.flag) || 'green' };
  const w = isObj(m.weather) ? m.weather : {};
  s.weather = { air: num(w.air), track: num(w.track), humidity: num(w.humidity), pressure: num(w.pressure), wind: num(w.wind), windDir: num(w.windDir), rain: bool(w.rain) };
  const cars = {};
  if (isObj(m.cars)) for (const [k, c] of Object.entries(m.cars)) { const n = normCar(k, c); if (n) cars[str(k)] = n; }
  s.cars = cars;
  const seen = new Set(), order = [];
  for (const n of arr(m.order)) { const k = str(n); if (cars[k] && !seen.has(k)) { seen.add(k); order.push(k); } }
  const rest = Object.keys(cars).filter((k) => !seen.has(k)).sort((a, b) => (cars[a].pos == null ? 999 : cars[a].pos) - (cars[b].pos == null ? 999 : cars[b].pos) || a.localeCompare(b, undefined, { numeric: true }));
  s.order = order.concat(rest);
  s.raceControl = arr(m.raceControl).filter(isObj).slice(0, MAX_RC).map((r) => ({ ...r, utc: num(r.utc), lap: num(r.lap), sector: num(r.sector), message: str(r.message), category: str(r.category), flag: str(r.flag), num: r.num == null ? null : str(r.num) }));
  s.radio = arr(m.radio).filter(isObj).slice(0, MAX_RADIO).map((r) => ({ ...r, utc: num(r.utc), num: str(r.num), tla: str(r.tla), url: str(r.url) }));
  const rec = isObj(m.records) ? m.records : {};
  const fl = isObj(rec.fastestLap) ? rec.fastestLap : null;
  s.records = { fastestLap: fl ? { ...fl, num: str(fl.num), sec: num(fl.sec), text: str(fl.text), lap: num(fl.lap) } : null, topSpeed: isObj(rec.topSpeed) ? rec.topSpeed : null };
  s.pitStops = arr(m.pitStops).filter(isObj).slice(-MAX_PITS).map((p) => ({ ...p, num: str(p.num), lap: num(p.lap), utc: num(p.utc), durationSec: num(p.durationSec), fromCompound: str(p.fromCompound), toCompound: str(p.toCompound) }));
  if (m.phase != null) s.phase = str(m.phase);
  return s;
}

export function normHistory(m) {
  if (!isObj(m)) return null;
  const laps = {};
  if (isObj(m.laps)) {
    for (const [k, list] of Object.entries(m.laps)) {
      const out = [];
      for (const e of arr(list)) {
        if (!isObj(e)) continue;
        const lap = num(e.lap); if (lap == null) continue;
        const s = arr(e.s);
        out.push({ ...e, lap, sec: num(e.sec), s: [num(s[0]), num(s[1]), num(s[2])], compound: str(e.compound), tyreAge: num(e.tyreAge), pos: num(e.pos),
          gapLeader: num(e.gapLeader), interval: num(e.interval), pit: bool(e.pit), track: str(e.track) || '1' });
      }
      laps[str(k)] = out.length > MAX_LAPS_PER_CAR ? out.slice(-MAX_LAPS_PER_CAR) : out;
    }
  }
  const events = arr(m.events).filter(isObj).slice(-MAX_EVENTS).map((e) => ({ ...e, utc: num(e.utc), lap: num(e.lap), kind: str(e.kind), num: e.num == null ? null : str(e.num), text: str(e.text) }));
  let maxLap = num(m.maxLap) || 0;
  for (const l of Object.values(laps)) for (const e of l) if (e.lap > maxLap) maxLap = e.lap;
  return { ...m, laps, events, maxLap };
}

function normFast(m) {
  const out = {};
  if (!isObj(m.cars)) return out;
  for (const [k, v] of Object.entries(m.cars)) {
    if (!isObj(v)) continue;
    const o = {};
    for (const f of ['x', 'y', 'z', 'speed', 'rpm', 'gear', 'throttle', 'brake', 'progress']) if (f in v) o[f] = num(v[f]);
    if ('drs' in v) o.drs = bool(v.drs);
    out[str(k)] = o;
  }
  return out;
}

function normTrack(t) {
  if (!isObj(t) || !Array.isArray(t.points)) return null;
  const pts = t.points.filter((p) => Array.isArray(p) && num(p[0]) != null && num(p[1]) != null).map((p) => [+p[0], +p[1]]);
  if (pts.length < 3) return null;
  const f2 = (a) => arr(a).filter((z) => Array.isArray(z) && num(z[0]) != null && num(z[1]) != null);
  const secs = arr(t.sectors).map(num).filter((v) => v != null);
  return { ...t, points: pts, length: num(t.length) || 0, startIdx: Math.max(0, Math.min(pts.length - 1, num(t.startIdx) || 0)), rotation: num(t.rotation) || 0,
    drsZones: f2(t.drsZones), sectors: secs.length >= 2 ? secs.slice(0, 2) : undefined };
}

// ------------------------------------------------------------------ store
export const store = {
  state: null, fast: {}, track: null, history: null, hello: null, control: null,
  conn: 'connecting',
  phase: null,            // 'waiting' | 'active' ... as reported by the backend (hello/state/api/status)
  lastMsgAt: 0,           // performance.now() of the last frame of any kind
  lastStateAt: 0,
  feedAge: null,          // upstream feed age in seconds, from /api/status (null if unknown)
  retry: { attempt: 0, nextAt: 0 },
  selected: new Set(),
  subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); },
  emit(kind) { for (const fn of [...listeners]) { try { fn(kind); } catch (e) { reportError(fn._panel || 'subscriber', e); } } },
  select(num, { additive = false } = {}) {
    num = str(num);
    if (additive) { this.selected.has(num) ? this.selected.delete(num) : this.selected.add(num); }
    else if (this.selected.size === 1 && this.selected.has(num)) { this.selected.clear(); }
    else { this.selected = new Set([num]); }
    try { localStorage.setItem('f1.selected', JSON.stringify([...this.selected])); } catch {}
    this.emit('selection');
  },
  send(obj) { try { if (this._ws && this._ws.readyState === 1) this._ws.send(JSON.stringify(obj)); } catch {} },
  /** Drop all session data (called on `hello`, i.e. every (re)connect). Selection is kept. */
  reset() {
    this.state = null; this.fast = {}; this.history = null; this.control = null;
    this.emit('reset');
  },
  /** Feed a decoded server frame (also used by mocks and the fuzz test). */
  ingest(m) {
    if (!isObj(m)) return;
    this.lastMsgAt = performance.now();
    switch (m.type) {
      case 'hello':
        this.reset();
        this.hello = m; if (m.phase != null) this.phase = str(m.phase);
        this.emit('conn'); break;
      case 'state': {
        const s = normState(m); if (!s) break;
        if (s.utc == null) s.utc = this.state && this.state.utc != null ? this.state.utc : Date.now() / 1000;
        if (s.phase != null) this.phase = s.phase;
        this.state = s; this.lastStateAt = this.lastMsgAt; this.emit('state'); break;
      }
      case 'fast': {
        const f = normFast(m);
        for (const [k, v] of Object.entries(f)) {
          if (!(k in this.fast) && Object.keys(this.fast).length >= MAX_FAST) continue;
          this.fast[k] = { ...(this.fast[k] || {}), ...v };
        }
        this.fastUtc = num(m.utc); this.emit('fast'); break;
      }
      case 'track': this.track = normTrack(m.track) || this.track; this.emit('track'); break;
      case 'history': { const hh = normHistory(m); if (hh) { this.history = hh; this.emit('history'); } break; }
      case 'control': this.control = { speed: num(m.speed), paused: typeof m.paused === 'boolean' ? m.paused : undefined }; this.emit('control'); break;
      default: break;
    }
  },
  _ws: null,
};
try { const sel = JSON.parse(localStorage.getItem('f1.selected') || '[]'); if (Array.isArray(sel)) store.selected = new Set(sel.map(String)); } catch {}

// ------------------------------------------------------------------ error reporting (visible badge + capped console output)
const errCounts = new Map();
export function reportError(name, e) {
  const n = (errCounts.get(name) || 0) + 1; errCounts.set(name, n);
  if (n <= 3 || n % 50 === 0) console.error('[panel error] ' + name + (n > 1 ? ' (x' + n + ')' : ''), e);
  try { window.dispatchEvent(new CustomEvent('f1-panel-error', { detail: { name, message: String((e && e.message) || e) } })); } catch {}
}

// ------------------------------------------------------------------ connection with backoff
let ws = null, statusTimer = 0, statusFails = 0, retryTimer = 0;

function connect() {
  clearTimeout(retryTimer);
  let sock;
  try {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    sock = new WebSocket(`${proto}://${location.host}/ws${location.search.includes('token=') ? location.search : ''}`);
  } catch (e) { scheduleRetry(); return; }
  ws = sock; store._ws = sock; store.conn = 'connecting'; store.emit('conn');
  let opened = false;
  sock.onopen = () => { opened = true; store.conn = 'open'; store.retry.attempt = 0; store.lastMsgAt = performance.now(); store.emit('conn'); pollStatus(true); };
  sock.onclose = () => {
    if (ws !== sock) return;
    store.conn = 'closed'; store.emit('conn'); scheduleRetry();
  };
  sock.onerror = () => { /* onclose follows */ };
  sock.onmessage = (ev) => {
    let m;
    try { m = JSON.parse(ev.data); } catch (e) { return; }
    try { store.ingest(m); } catch (e) { reportError('store.ingest', e); }
  };
}
function scheduleRetry() {
  const a = ++store.retry.attempt;
  const delay = Math.min(15000, 1000 * 2 ** Math.min(a - 1, 4)) * (0.8 + Math.random() * 0.4);
  store.retry.nextAt = performance.now() + delay;
  store.emit('conn');
  clearTimeout(retryTimer); retryTimer = setTimeout(connect, delay);
}

async function pollStatus(first) {
  clearTimeout(statusTimer);
  const next = () => { statusTimer = setTimeout(pollStatus, store.state && !/wait/i.test(store.phase || '') ? 8000 : 3000); };
  if (store.conn !== 'open') { next(); return; }
  try {
    const ctl = new AbortController(); const to = setTimeout(() => ctl.abort(), 4000);
    const r = await fetch('/api/status' + (location.search.includes('token=') ? location.search : ''), { cache: 'no-store', signal: ctl.signal });
    clearTimeout(to);
    if (r.ok) {
      const j = await r.json(); statusFails = 0;
      if (j && typeof j === 'object') {
        if (j.phase != null && String(j.phase) !== store.phase) { store.phase = String(j.phase); store.emit('conn'); }
        store.feedAge = num(j.lastMessageAgeSec);
        if (j.control && store.control == null) { store.control = { speed: num(j.control.speed), paused: j.control.paused }; }
        store.emit('status');
      }
    } else statusFails++;
  } catch (e) { statusFails++; }
  // half-open socket: server unreachable for 3 polls while we think we are connected -> force a reconnect
  if (statusFails >= 3 && ws && ws.readyState === 1) { statusFails = 0; try { ws.close(); } catch {} }
  next();
}

export async function fetchTrackFallback() {
  try {
    const r = await fetch('/api/track', { cache: 'no-store' }); if (!r.ok) return;
    const j = await r.json(); const t = normTrack(j && j.track);
    if (t && !store.track) { store.track = t; store.emit('track'); }
  } catch {}
}

export function start() {
  connect();
  store.subscribe((k) => { if (k === 'conn' && store.hello && !store.track) fetchTrackFallback(); });
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && store.conn === 'closed') { store.retry.attempt = 0; connect(); }
  });
  window.addEventListener('online', () => { if (store.conn === 'closed') { store.retry.attempt = 0; connect(); } });
}
