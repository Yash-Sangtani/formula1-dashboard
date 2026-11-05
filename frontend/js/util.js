// Shared helpers for the dashboard shell and panels.
import { store } from './store.js';

export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];

export function h(tag, props, ...kids) {
  const e = document.createElement(tag);
  if (props) for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'dataset') Object.assign(e.dataset, v);
    else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
    else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) e.append(k.nodeType ? k : document.createTextNode(String(k)));
  return e;
}

export const ls = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

// Update text only when changed (cached on the node).
export function setText(node, v) {
  v = v == null ? '' : String(v);
  if (node._t !== v) { node._t = v; node.textContent = v; }
}
// Set className only when changed.
export function setClass(node, cls) {
  if (node._c !== cls) { node._c = cls; node.className = cls; }
}
export function setStyle(node, prop, v) {
  const k = '_s' + prop;
  if (node[k] !== v) { node[k] = v; node.style[prop] = v; }
}

export const COMPOUNDS = {
  SOFT: { l: 'S', c: '#ff3b3b' }, MEDIUM: { l: 'M', c: '#ffd300' }, HARD: { l: 'H', c: '#f2f2f2' },
  INTERMEDIATE: { l: 'I', c: '#3ec14f' }, WET: { l: 'W', c: '#2f80ff' },
};
export function compound(c) { return COMPOUNDS[String(c || '').toUpperCase()] || { l: '?', c: '#888' }; }

// colour class for a timing cell {text, overall, personal}
export function timeClass(t) {
  if (!t || !t.text) return 't-none';
  if (t.overall) return 't-purple';
  if (t.personal) return 't-green';
  return 't-yellow';
}

export function fmtClock(sec) {
  if (sec == null || !isFinite(sec)) return '--:--:--';
  sec = Math.max(0, Math.round(sec));
  const hh = Math.floor(sec / 3600), mm = Math.floor(sec / 60) % 60, ss = sec % 60;
  return `${hh}:${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
}
export function fmtUtc(utc) {
  if (utc == null) return '';
  const d = typeof utc === 'number' ? new Date(utc * (utc < 1e12 ? 1000 : 1)) : new Date(utc);
  if (isNaN(d)) return '';
  return d.toISOString().slice(11, 19);
}
export function textOn(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex || ''); if (!m) return '#fff';
  const n = parseInt(m[1], 16), r = n >> 16, g = (n >> 8) & 255, b = n & 255;
  return (r * 299 + g * 587 + b * 114) / 1000 > 150 ? '#000' : '#fff';
}
export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

// Replay control state shared by header + keyboard shortcuts.
export const SPEEDS = [1, 2, 5, 10, 20];
const rsubs = new Set();
export const replay = {
  speed: 1, paused: false,
  on(fn) { rsubs.add(fn); return () => rsubs.delete(fn); },
  _emit() { for (const f of rsubs) f(); },
  setSpeed(s) { this.speed = s; store.send({ type: 'control', speed: s }); this._emit(); },
  setPaused(p) { this.paused = p; store.send({ type: 'control', paused: p }); this._emit(); },
  toggle() { this.setPaused(!this.paused); },
  step(dir) {
    let i = SPEEDS.indexOf(this.speed); if (i < 0) i = 0;
    this.setSpeed(SPEEDS[clamp(i + dir, 0, SPEEDS.length - 1)]);
  },
  available() { return !!store.hello && store.hello.replay !== false && store.hello.mode !== 'live'; },
  sync(hello) {
    const i = hello && hello.info;
    if (i && typeof i.speed === 'number') this.speed = i.speed;
    if (i && typeof i.paused === 'boolean') this.paused = i.paused;
    this._emit();
  },
};
