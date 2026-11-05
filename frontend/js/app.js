import { store, start } from './store.js';
import { ls, replay, SPEEDS, h } from './util.js';
import * as header from './panels/header.js';
import * as tower from './panels/tower.js';
import * as trackmap from './panels/trackmap.js';
import * as racecontrol from './panels/racecontrol.js';
import * as radio from './panels/radio.js';
import * as layout from './panels/layout.js';
import * as alerts from './panels/alerts.js';

const $ = (s) => document.querySelector(s);
const DEFAULT_UI = { view: 'race', tower: 'auto', side: true };
const ui = Object.assign({}, DEFAULT_UI, ls.get('f1.ui', {}));
const subs = new Set();

function applyUi() {
  const b = document.body;
  b.dataset.view = ui.view; b.dataset.tower = ui.tower;
  b.classList.toggle('noside', !ui.side);
  ls.set('f1.ui', ui);
  for (const f of subs) f();
  mountLazy();
  window.dispatchEvent(new Event('resize'));
}
const ctl = {
  ui,
  set(patch) { Object.assign(ui, DEFAULT_UI, patch); applyUi(); },
  reset() { Object.assign(ui, DEFAULT_UI); applyUi(); },
  onChange(f) { subs.add(f); },
  bigScreen() {
    const on = ui.view === 'all' && document.fullscreenElement;
    if (on) { document.exitFullscreen(); return; }
    this.set({ ...ui, view: 'all' });
    document.documentElement.requestFullscreen && document.documentElement.requestFullscreen().catch(() => {});
  },
};

function safeMount(name, mod, el, ...args) {
  try { mod.mount(el, store, ...args); } catch (e) { console.error('mount ' + name, e); el.append(h('div', { class: 'empty' }, name + ' failed: ' + e.message)); }
}

// ---- fixed panels
safeMount('header', header, $('#panel-header'));
safeMount('layout', layout, $('#toolbar .tb-views'), ctl);
safeMount('alerts', alerts, $('#toolbar .tb-alerts'));
safeMount('tower', tower, $('#panel-tower .pb'));
safeMount('trackmap', trackmap, $('#panel-map .pb'));
safeMount('racecontrol', racecontrol, $('#rc-body'));
safeMount('radio', radio, $('#radio-body'));

// ---- lazily loaded panels owned by another module
const lazy = { deepdive: false, analytics: false };
async function loadLazy(name) {
  const host = $(`#panel-${name} .pb`);
  lazy[name] = true;
  try {
    const mod = await import(`./panels/${name}.js`);
    host.textContent = '';
    mod.mount(host, store);
  } catch (e) {
    console.warn(name + ' unavailable', e);
    host.textContent = '';
    host.append(h('div', { class: 'placeholder' }, h('b', null, name === 'deepdive' ? 'Deep dive' : 'Analytics'), ' panel not available yet.'));
  }
}
function visible(el) { return el.offsetWidth > 0 && el.offsetHeight > 0; }
function mountLazy() {
  for (const n of Object.keys(lazy)) if (!lazy[n] && visible($('#panel-' + n))) loadLazy(n);
}

// ---- team colour theming from primary selection
function theme() {
  const prim = [...store.selected][0];
  const c = prim && store.state && store.state.cars && store.state.cars[prim];
  document.documentElement.style.setProperty('--team', c && c.color ? c.color : 'var(--red)');
}

// ---- keyboard
function help() {
  let o = $('#help');
  if (o) { o.remove(); return; }
  const rows = [['1 2 3 4', 'Race / Deep dive / Analytics / All'], ['j / k', 'Select next / previous car'], ['Space', 'Pause / resume replay'],
    ['+ / -', 'Replay speed up / down'], ['F', 'Big-screen fullscreen'], ['Esc', 'Clear selection / close'], ['?', 'This help'],
    ['Click', 'Select car (Shift/Ctrl adds)']];
  o = h('div', { id: 'help', onclick: () => o.remove() }, h('div', { class: 'help-card' }, h('h3', null, 'Keyboard shortcuts'),
    h('table', null, rows.map(([k, d]) => h('tr', null, h('td', null, h('kbd', null, k)), h('td', null, d))))));
  document.body.append(o);
}
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const t = e.target; if (t && /INPUT|TEXTAREA|SELECT/.test(t.tagName) && t.type !== 'checkbox') return;
  const k = e.key;
  const views = ['race', 'deep', 'analytics', 'all'];
  if (k >= '1' && k <= '4') ctl.set({ ...ui, view: views[+k - 1] });
  else if (k === 'j' || k === 'k') {
    const order = (store.state && store.state.order) || []; if (!order.length) return;
    const cur = order.indexOf([...store.selected][0]);
    const i = cur < 0 ? 0 : Math.min(order.length - 1, Math.max(0, cur + (k === 'j' ? 1 : -1)));
    store.selected = new Set(); store.select(order[i]);
  } else if (k === ' ') { if (replay.available()) { e.preventDefault(); replay.toggle(); } }
  else if (k === '+' || k === '=') replay.step(1);
  else if (k === '-' || k === '_') replay.step(-1);
  else if (k === 'f' || k === 'F') ctl.bigScreen();
  else if (k === '?') help();
  else if (k === 'Escape') { const o = $('#help'); if (o) o.remove(); else if (store.selected.size) { store.selected = new Set(); try { localStorage.setItem('f1.selected', '[]'); } catch {} store.emit('selection'); } }
});

store.subscribe((k) => { if (k === 'selection' || k === 'state') theme(); });
window.addEventListener('resize', () => requestAnimationFrame(mountLazy));
applyUi(); theme();

if (new URLSearchParams(location.search).has('mock')) import('../dev/mock.js').then((m) => m.run(store));
else start();
