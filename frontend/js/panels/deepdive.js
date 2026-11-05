// Per-car deep dive panel (stage 3). export function mount(el, store)
import {
  Chart, COLORS, esc, h, rafThrottle, isNum, fmtLap, fmtSec, fmtDelta, median, quantile,
  tyreColor, tyreLetter, indexHistory, teamStyles, raceBands, stintsFromLaps,
} from '../charts.js';
import { sessionOf } from '../session.js';
import { activityOf, classifyLaps, runsFromLaps } from '../derive.js';

const MAX_PIN = 4;
const WINDOW_S = 20;
const SEC_COLORS = { pu: COLORS.purple, gr: COLORS.green, ye: COLORS.yellow };
const bestsCache = new WeakMap();

/** Personal bests per car and overall bests (sectors + lap) derived from history.laps */
function bests(history) {
  if (!history || !history.laps) return { pb: {}, ob: { s: [null, null, null], lap: null } };
  let c = bestsCache.get(history);
  if (c) return c;
  const pb = {}, ob = { s: [null, null, null], lap: null };
  for (const [num, arr] of Object.entries(history.laps)) {
    const p = pb[num] = { s: [null, null, null], lap: null, lapNo: null };
    for (const e of arr || []) {
      for (let i = 0; i < 3; i++) {
        const v = e.s && e.s[i];
        if (isNum(v) && (p.s[i] == null || v < p.s[i])) p.s[i] = v;
        if (isNum(v) && (ob.s[i] == null || v < ob.s[i].v)) ob.s[i] = { v, num };
      }
      if (isNum(e.sec) && !e.pit && (p.lap == null || e.sec < p.lap)) { p.lap = e.sec; p.lapNo = e.lap; }
    }
    if (p.lap != null && (ob.lap == null || p.lap < ob.lap.v)) ob.lap = { v: p.lap, num };
  }
  c = { pb, ob };
  bestsCache.set(history, c);
  return c;
}
const secCls = (v, pbv, obv) => (!isNum(v) ? 'na' : obv != null && v <= obv + 5e-4 ? 'pu' : pbv != null && v <= pbv + 5e-4 ? 'gr' : 'ye');

function refs(root) {
  const o = {};
  root.querySelectorAll('[data-r]').forEach((e) => { o[e.dataset.r] = e; });
  return o;
}

export function mount(el, store) {
  el.classList.add('dd');
  const root = h('div', 'dd-root');
  el.appendChild(root);
  let key = '', cols = [], cmp = null, pins = [];
  const bufs = {};
  const dirty = { state: true, fast: true, hist: true };
  const cars = () => (store.state && store.state.cars) || {};
  const hist = () => store.history;

  // ------------------------------------------------------------ helpers
  function pinned() {
    const c = cars();
    return [...store.selected].filter((n) => c[n]).slice(0, MAX_PIN);
  }
  function lapsOf(n) { const hh = hist(); return (hh && hh.laps && hh.laps[n]) || []; }
  function carBehind(n) {
    const st = store.state; if (!st || !st.order) return null;
    const i = st.order.indexOf(n);
    return i >= 0 && i + 1 < st.order.length ? st.order[i + 1] : null;
  }
  function carAhead(n) {
    const st = store.state; if (!st || !st.order) return null;
    const i = st.order.indexOf(n);
    return i > 0 ? st.order[i - 1] : null;
  }
  /** gap trend over the last 3 laps from history interval values (positive = growing) */
  function trend(n) {
    const a = lapsOf(n).filter((e) => isNum(e.interval)).slice(-4);
    if (a.length < 2) return null;
    return a[a.length - 1].interval - a[0].interval;
  }
  function trendHtml(d, behind) {
    if (d == null) return '<span class="dd-tr flat">·</span>';
    if (Math.abs(d) < 0.12) return '<span class="dd-tr flat" title="stable over 3 laps">■</span>';
    const shrinking = d < 0;
    // gap ahead shrinking = good (green); gap behind shrinking = bad (red)
    const good = behind ? !shrinking : shrinking;
    return `<span class="dd-tr ${good ? 'good' : 'bad'}" title="${fmtDelta(d, 2)}s over last 3 laps">${shrinking ? '▼' : '▲'} ${Math.abs(d).toFixed(1)}</span>`;
  }

  // ------------------------------------------------------------ structure
  function rebuild() {
    for (const c of cols) c.destroy();
    if (cmp) cmp.destroy();
    cols = []; cmp = null; root.innerHTML = '';
    const st = store.state;
    if (!st || !st.cars) { root.appendChild(h('div', 'dd-empty', 'Waiting for session data…')); return; }
    if (!pins.length) { buildPicker(); return; }
    const single = pins.length === 1;
    root.classList.toggle('dd-single', single);
    if (store.selected.size > MAX_PIN) {
      root.appendChild(h('div', 'dd-note', `Showing the first ${MAX_PIN} of ${store.selected.size} selected drivers.`));
    }
    const grid = h('div', 'dd-cols');
    grid.style.setProperty('--n', pins.length);
    root.appendChild(grid);
    for (const n of pins) { const c = makeCol(n, single); cols.push(c); grid.appendChild(c.root); }
    if (!single) { cmp = makeCompare(); root.appendChild(cmp.root); }
  }

  function buildPicker() {
    const st = store.state;
    const w = h('div', 'dd-picker');
    w.innerHTML = '<div class="dd-picker-h">Select a driver for the deep dive <small>(shift/ctrl-click to pin up to 4)</small></div>';
    const g = h('div', 'dd-pgrid');
    for (const n of st.order || Object.keys(st.cars)) {
      const c = st.cars[n]; if (!c) continue;
      const b = h('button', 'dd-pbtn');
      b.type = 'button';
      b.style.setProperty('--tc', c.color || '#888');
      b.innerHTML = `<i>${c.pos ?? ''}</i><b>${esc(c.tla || n)}</b><span>${esc(c.name || '')}</span><em>#${esc(n)}</em>`;
      b.addEventListener('click', (e) => store.select(n, { additive: e.shiftKey || e.ctrlKey || e.metaKey }));
      g.appendChild(b);
    }
    w.appendChild(g);
    root.appendChild(w);
  }

  // ------------------------------------------------------------ one car column
  function makeCol(num, single) {
    const r = h('div', 'dd-col' + (single ? ' dd-col-single' : ''));
    const c0 = cars()[num];
    const timed = sessionOf(store.state).timed;
    r.style.setProperty('--tc', c0.color || '#888');
    r.innerHTML = `
      <div class="dd-head dd-card" data-r="head"></div>
      <div class="dd-card dd-tel">
        <div class="dd-h">Live telemetry</div>
        <div class="dd-gauges">
          <div class="dd-speed"><b data-r="speed">0</b><small>km/h</small></div>
          <div class="dd-gear"><b data-r="gear">N</b><small>gear</small></div>
          <div class="dd-drs" data-r="drs">DRS</div>
          <div class="dd-rpm"><div class="dd-leds" data-r="leds">${'<span></span>'.repeat(15)}</div>
            <div class="dd-bar"><i data-r="rpmfill"></i></div><small data-r="rpm">0 rpm</small></div>
          <div class="dd-pedals">
            <div class="dd-pedal"><label>THR</label><div class="dd-bar thr"><i data-r="thr"></i></div><em data-r="thrv">0</em></div>
            <div class="dd-pedal"><label>BRK</label><div class="dd-bar brk"><i data-r="brk"></i></div><em data-r="brkv">0</em></div>
          </div>
        </div>
        <div class="dd-strip-l">Last ${WINDOW_S}s <span><i style="background:${esc(c0.color || '#888')}"></i>speed</span><span><i style="background:${COLORS.green}"></i>throttle</span><span><i style="background:${COLORS.red}"></i>brake</span></div>
        <div class="dd-strip" data-r="strip1"></div>
        <div class="dd-strip dd-strip2" data-r="strip2"></div>
      </div>
      <div class="dd-card dd-sectors"><div class="dd-h">Sectors</div><div data-r="sectors"></div></div>
      <div class="dd-card dd-stints"><div class="dd-h">Tyre stints</div><div data-r="stints"></div>
        ${timed ? '' : '<div class="dd-h dd-h2">Pit stops</div><div data-r="pits"></div>'}
        <div class="dd-h dd-h2">Speed traps</div><div data-r="traps"></div></div>
      ${single ? `<div class="dd-card dd-wide"><div class="dd-h">Lap pace <small>tyre-coloured stints, pit laps marked</small></div><div class="dd-chart" data-r="pace"></div></div>
      ${timed ? `<div class="dd-card dd-wide"><div class="dd-h">Gap to session best <small>seconds, timed laps only</small></div><div class="dd-chart" data-r="gaps"></div></div>
      <div class="dd-card dd-wide"><div class="dd-h">Lap list <small>OUT / IN laps flagged</small></div><div class="dd-scroll dd-laplist" data-r="laplist"></div></div>`
      : `<div class="dd-card dd-wide"><div class="dd-h">Gap ahead / behind <small>seconds, per lap</small></div><div class="dd-chart" data-r="gaps"></div></div>`}` : ''}`;
    const R = refs(r);
    const col = { root: r, num, R, charts: {} };
    R.head.addEventListener('click', (e) => {
      const b = e.target.closest('[data-act]');
      if (b && b.dataset.act === 'rm') store.select(num, { additive: true });
    });
    R.head.addEventListener('change', (e) => {
      if (e.target.dataset.act === 'add' && e.target.value) store.select(e.target.value, { additive: true });
    });
    const tc = c0.color || '#888';
    col.charts.speed = new Chart(R.strip1, {
      height: 84, left: 34, right: 14, top: 6, bottom: 4, xDomain: [-WINDOW_S, 0], yDomain: [0, 360],
      yTicks: [0, 100, 200, 300].map((v) => ({ v, label: String(v) })), xTicks: [], hoverHighlight: false,
      xFmt: (v) => v.toFixed(1) + 's', yFmt: (v) => Math.round(v) + ' km/h', series: [{ id: 'spd', label: 'Speed', color: tc, fill: tc + '22', points: [] }],
    });
    col.charts.pedals = new Chart(R.strip2, {
      height: 62, left: 34, right: 14, top: 4, bottom: 16, xDomain: [-WINDOW_S, 0], yDomain: [0, 105],
      yTicks: [0, 50, 100].map((v) => ({ v, label: v + '%' })), xTicks: [-20, -15, -10, -5, 0].map((v) => ({ v, label: v ? v + 's' : 'now' })),
      xFmt: (v) => v.toFixed(1) + 's', yFmt: (v) => Math.round(v) + '%', hoverHighlight: false,
      series: [
        { id: 'thr', label: 'Throttle', color: COLORS.green, fill: COLORS.green + '22', points: [] },
        { id: 'brk', label: 'Brake', color: COLORS.red, fill: COLORS.red + '33', step: true, points: [] },
      ],
    });
    if (single) {
      col.charts.pace = new Chart(R.pace, { height: 240, left: 46, xName: 'Lap', xFmt: (v) => String(Math.round(v)), yFmt: fmtLap, legend: true, series: [] });
      col.charts.gaps = new Chart(R.gaps, {
        height: 200, left: 40, xName: 'Lap', xFmt: (v) => String(Math.round(v)), yFmt: (v) => v.toFixed(timed ? 3 : 2) + 's', yDomain: [0, null], legend: true, series: [],
        hlines: [{ y: timed ? 0 : 1, color: 'rgba(46,204,113,.5)', label: '' }],
      });
    }
    col.destroy = () => Object.values(col.charts).forEach((c) => c.destroy());
    col.renderState = () => renderState(col);
    col.renderGauges = () => renderGauges(col);
    col.renderStrip = () => renderStrip(col);
    col.timed = timed;
    col.renderCharts = () => renderCharts(col, single);
    return col;
  }

  function renderState(col) {
    const st = store.state, n = col.num, c = st.cars[n]; if (!c) return;
    const { R } = col;
    const keepHead = R.head.contains(document.activeElement) && document.activeElement.tagName === 'SELECT';
    const ahead = carAhead(n), behind = carBehind(n);
    const A = ahead && st.cars[ahead], B = behind && st.cars[behind];
    const X = sessionOf(st), timed = X.timed;
    const act = activityOf(c);
    const status = c.retired ? '<span class="dd-pill dnf">OUT</span>' : timed ? (c.knockedOut ? '<span class="dd-pill dnf">KNOCKED OUT</span>' : act.t && act.k !== 'track' ? `<span class="dd-pill ${act.k === 'out' ? 'out' : 'pit'}">${esc(act.t)}</span>` : '')
      : c.inPit ? '<span class="dd-pill pit">PIT</span>' : c.pitOut ? '<span class="dd-pill out">PIT EXIT</span>' : c.stopped ? '<span class="dd-pill dnf">STOPPED</span>' : '';
    const t = c.tyre || {};
    const addable = (st.order || []).filter((x) => !pins.includes(x));
    const opts = addable.map((x) => `<option value="${esc(x)}">${esc(st.cars[x].tla)} – P${st.cars[x].pos}</option>`).join('');
    if (!keepHead) R.head.innerHTML = `
      <div class="dd-pos">P${c.pos ?? '–'}<small>${c.posChange ? (c.posChange > 0 ? '▲' + c.posChange : '▼' + -c.posChange) : ''}</small></div>
      <div class="dd-id"><b>${esc(c.tla || n)}</b><span>${esc(c.name || '')}</span><span class="dd-team">${esc(c.team || '')} · #${esc(n)}</span></div>
      <div class="dd-tyre"><span class="dd-tdot" style="background:${tyreColor(t.compound)}">${tyreLetter(t.compound)}</span>
        <span>${esc(t.compound || '–')}<small>${t.age != null ? t.age + ' laps' : ''}${t.new === false ? ' · used' : t.new ? ' · new' : ''}</small></span></div>
      ${timed ? `<div class="dd-gaps">
        <div><label>To fastest</label><b>${c.best && c.best.text ? (c.pos === 1 || !(c.gapFastest || c.gapLeader) ? 'FASTEST' : esc(c.gapFastest || c.gapLeader)) : '–'}</b></div>
        <div><label>To car ahead ${A ? esc(A.tla) : ''}</label><b>${ahead && c.best && c.best.text ? esc(c.interval || '–') : '–'}</b></div>
        ${X.isQuali ? `<div><label>Q1 / Q2 / Q3</label><b class="dd-qs">${(c.bestLaps || []).map((b) => esc(b.text || '–')).join(' · ')}</b></div>` : ''}
      </div>` : `<div class="dd-gaps">
        <div><label>Ahead ${A ? esc(A.tla) : ''}</label><b>${ahead ? esc(c.interval || '–') : 'LEADER'}</b>${ahead ? trendHtml(trend(n), false) : ''}</div>
        <div><label>Behind ${B ? esc(B.tla) : ''}</label><b>${behind ? esc(B.interval || '–') : '–'}</b>${behind ? trendHtml(trend(behind), true) : ''}</div>
      </div>`}
      <div class="dd-lapinfo"><div><label>${timed ? 'Timed laps' : 'Lap'}</label><b>${(timed ? (c.timedLaps != null ? c.timedLaps : c.lap) : c.lap) ?? '–'}</b></div><div><label>Last</label><b>${esc((c.last && c.last.text) || '–')}</b></div><div><label>Best</label><b>${esc((c.best && c.best.text) || '–')}</b></div></div>
      ${status}
      <div class="dd-ctl">${pins.length < MAX_PIN && addable.length ? `<select data-act="add" title="Pin another driver"><option value="">+ add</option>${opts}</select>` : ''}<button type="button" data-act="rm" title="Unpin">×</button></div>`;
    // sectors
    const hh = hist(), B2 = bests(hh), pb = B2.pb[n] || { s: [null, null, null] };
    const rows = [0, 1, 2].map((i) => {
      const cur = c.sectors && c.sectors[i];
      const v = cur && cur.sec, p = pb.s[i], o = B2.ob.s[i];
      const cls = isNum(v) ? (cur.overall ? 'pu' : cur.personal ? 'gr' : secCls(v, p, o && o.v)) : 'na';
      const pcls = isNum(p) && o && p <= o.v + 5e-4 ? 'pu' : 'gr';
      return `<tr><th>S${i + 1}</th><td class="${cls}">${isNum(v) ? v.toFixed(3) : '–'}</td><td class="${isNum(p) ? pcls : 'na'}">${fmtSec(p)}</td>
        <td class="pu2">${o ? o.v.toFixed(3) : '–'}<small>${o ? esc(st.cars[o.num] ? st.cars[o.num].tla : o.num) : ''}</small></td><td class="dl">${isNum(v) && isNum(p) ? fmtDelta(v - p) : ''}</td></tr>`;
    }).join('');
    const rec = st.records && st.records.fastestLap;
    const lcls = c.last && c.last.overall ? 'pu' : c.last && c.last.personal ? 'gr' : c.last && c.last.sec ? 'ye' : 'na';
    R.sectors.innerHTML = `<table class="dd-tbl"><thead><tr><th></th><th>Last</th><th>Personal best</th><th>Overall best</th><th>Δ PB</th></tr></thead><tbody>${rows}
      <tr class="lap"><th>Lap</th><td class="${lcls}">${esc((c.last && c.last.text) || '–')}</td><td class="${c.best && c.best.overall ? 'pu' : 'gr'}">${esc((c.best && c.best.text) || '–')}</td>
      <td class="pu2">${rec ? esc(rec.text) : '–'}<small>${rec && st.cars[rec.num] ? esc(st.cars[rec.num].tla) : ''}</small></td><td class="dl">${c.last && isNum(c.last.sec) && c.best && isNum(c.best.sec) ? fmtDelta(c.last.sec - c.best.sec) : ''}</td></tr></tbody></table>`;
    // stints
    let stints = (c.stints || []).map((s) => ({ compound: s.compound, laps: s.laps, startLap: s.startLap, isNew: s.new }));
    if (!stints.length) stints = (timed ? runsFromLaps : stintsFromLaps)(lapsOf(n)).map((s) => ({ compound: s.compound, laps: s.laps.length, startLap: s.start - 1 }));
    const total = Math.max((st.lap && st.lap.current) || 0, c.lap || 0, stints.reduce((a, s) => Math.max(a, (s.startLap || 0) + (s.laps || 0)), 0), 1);
    R.stints.innerHTML = stints.length ? `<div class="dd-sbar">${stints.map((s, i) => `<div class="dd-seg" style="flex:${Math.max(s.laps || 0, 0.6)};background:${tyreColor(s.compound)}" title="${esc(s.compound)} stint ${i + 1}: ${s.laps} laps">${tyreLetter(s.compound)} ${s.laps ?? ''}</div>`).join('')}
      <div class="dd-seg dd-rest" style="flex:${Math.max(0, (st.lap && st.lap.total ? st.lap.total : total) - stints.reduce((a, s) => a + (s.laps || 0), 0))}"></div></div>` : '<div class="dd-none">No stint data yet</div>';
    // pits
    const ps = (st.pitStops || []).filter((p) => p.num === n);
    if (R.pits) R.pits.innerHTML = ps.length ? `<table class="dd-tbl dd-pit"><thead><tr><th>#</th><th>Lap</th><th>Time</th><th>Tyres</th></tr></thead><tbody>${ps.map((p, i) => `<tr><th>${i + 1}</th><td>${p.lap}</td><td>${isNum(p.durationSec) ? p.durationSec.toFixed(1) + 's' : '–'}</td>
      <td><span class="dd-tdot sm" style="background:${tyreColor(p.fromCompound)}">${tyreLetter(p.fromCompound)}</span> → <span class="dd-tdot sm" style="background:${tyreColor(p.toCompound)}">${tyreLetter(p.toCompound)}</span></td></tr>`).join('')}</tbody></table>` : '<div class="dd-none">No stops yet</div>';
    // speed traps
    const names = [['i1', 'I1'], ['i2', 'I2'], ['fl', 'FL'], ['st', 'ST']];
    R.traps.innerHTML = '<div class="dd-traps">' + names.map(([k, lab]) => {
      const v = c.speedTraps && c.speedTraps[k];
      const all = Object.values(st.cars).map((x) => x.speedTraps && x.speedTraps[k]).filter(isNum);
      const rank = isNum(v) ? all.filter((x) => x > v).length + 1 : null;
      return `<div class="${rank === 1 ? 'top' : ''}"><label>${lab}</label><b>${isNum(v) ? v : '–'}</b><small>${rank ? 'P' + rank + '/' + all.length : ''}</small></div>`;
    }).join('') + '</div>';
  }

  function telOf(n) {
    const t = (cars()[n] && cars()[n].tel) || {}, f = store.fast[n] || {};
    const pick = (k) => (f[k] != null ? f[k] : t[k]);
    return { speed: pick('speed'), rpm: pick('rpm'), gear: pick('gear'), throttle: pick('throttle'), brake: pick('brake'), drs: f.drs != null ? f.drs : t.drs };
  }
  function renderGauges(col) {
    const t = telOf(col.num), { R } = col;
    R.speed.textContent = isNum(t.speed) ? Math.round(t.speed) : '–';
    R.gear.textContent = !isNum(t.gear) ? '–' : t.gear <= 0 ? 'N' : t.gear;
    R.drs.classList.toggle('on', !!t.drs);
    const rpm = isNum(t.rpm) ? t.rpm : 0, maxr = rpm > 12600 ? 13500 : 12500;
    R.rpmfill.style.width = Math.min(100, (rpm / maxr) * 100) + '%';
    R.rpm.textContent = Math.round(rpm) + ' rpm';
    const lit = Math.floor(Math.max(0, Math.min(1, (rpm - 8500) / (maxr - 8500 - 300))) * 15 + 0.001);
    const leds = R.leds.children;
    for (let i = 0; i < 15; i++) leds[i].className = i < lit ? (i < 5 ? 'on g' : i < 10 ? 'on r' : 'on b') : '';
    const thr = isNum(t.throttle) ? Math.max(0, Math.min(100, t.throttle)) : 0;
    const brk = isNum(t.brake) ? (t.brake <= 1 ? t.brake * 100 : Math.min(100, t.brake)) : 0;
    R.thr.style.width = thr + '%'; R.thrv.textContent = Math.round(thr) + '%';
    R.brk.style.width = brk + '%'; R.brkv.textContent = brk > 0 ? 'ON' : '0%';
  }
  function pushSample(n) {
    const f = store.fast[n]; if (!f) return;
    const b = bufs[n] || (bufs[n] = []);
    const t = performance.now() / 1000;
    const brk = isNum(f.brake) ? (f.brake <= 1 ? f.brake * 100 : f.brake) : 0;
    b.push({ t, s: f.speed, th: f.throttle, br: brk });
    while (b.length && t - b[0].t > WINDOW_S + 2) b.shift();
  }
  function renderStrip(col) {
    const b = bufs[col.num] || [];
    if (!b.length) return;
    const t0 = b[b.length - 1].t;
    const sel = (k) => b.map((q) => [q.t - t0, isNum(q[k]) ? q[k] : null]);
    col.charts.speed.update({ series: [{ ...col.charts.speed.o.series[0], points: sel('s') }] });
    col.charts.pedals.update({ series: [{ ...col.charts.pedals.o.series[0], points: sel('th') }, { ...col.charts.pedals.o.series[1], points: sel('br') }] });
  }

  function paceDomain(vals) {
    const v = vals.filter(isNum);
    if (!v.length) return [null, null];
    const mn = Math.min(...v), md = median(v);
    const typical = v.filter((x) => x < md * 1.03);
    return [mn - 0.4, Math.max(...typical) + 0.6];
  }
  function renderCharts(col, single) {
    if (!single) return;
    const n = col.num, laps = lapsOf(n), st = store.state;
    const hh = hist(), ix = indexHistory(hh);
    const stints = (col.timed ? runsFromLaps : stintsFromLaps)(laps);
    const series = stints.map((s, i) => ({
      id: 'st' + i, label: `Stint ${i + 1} ${s.compound || ''}`, color: tyreColor(s.compound), markers: true, r: 2.4,
      points: s.laps.map((e) => [e.lap, isNum(e.sec) ? e.sec : null, e]),
    }));
    const cls = classifyLaps(laps);
    const secs = laps.map((e) => e.sec);
    const pbl = bests(hh).pb[n];
    col.charts.pace.update({
      series, yDomain: paceDomain(secs), xDomain: [1, Math.max(ix.maxLap, 2)],
      bands: raceBands(hh), vlines: laps.map((e, i) => [e, cls[i]]).filter(([e]) => e.pit).map(([e, k]) => ({ x: e.lap, color: 'rgba(255,255,255,.35)', label: col.timed ? k.toUpperCase() : 'PIT' })),
      hlines: pbl && pbl.lap ? [{ y: pbl.lap, color: 'rgba(46,204,113,.5)' }] : [],
      tooltip: (s, p) => `<b>Lap ${p[0]}</b> ${fmtLap(p[1])}<div>${esc(p[2].compound || '')} · age ${p[2].tyreAge ?? '–'}${p[2].pit ? ' · pit' : ''}</div><div>S ${(p[2].s || []).map((x) => (isNum(x) ? x.toFixed(2) : '–')).join(' / ')}</div>`,
    });
    if (col.timed) { renderTimedExtras(col, laps, cls, ix, hh); return; }
    // gaps
    const pa = [], pbk = [];
    for (const e of laps) {
      pa.push([e.lap, isNum(e.interval) ? e.interval : null]);
      const nx = ix.byLapPos.get(e.lap) && ix.byLapPos.get(e.lap).get((e.pos || 0) + 1);
      pbk.push([e.lap, nx && isNum(nx[1].interval) ? nx[1].interval : null]);
    }
    const all = pa.concat(pbk).map((q) => q[1]).filter(isNum);
    const top = all.length ? Math.max(4, Math.min(Math.max(...all) * 1.08, (quantile(all, 0.92) || 5) * 1.6)) : 5;
    col.charts.gaps.update({
      xDomain: [1, Math.max(ix.maxLap, 2)], yDomain: [0, top], bands: raceBands(hh),
      series: [
        { id: 'ahead', label: 'Gap ahead', color: '#4fc3f7', points: pa, markers: true, r: 1.8 },
        { id: 'behind', label: 'Gap behind', color: '#ff8a65', points: pbk, markers: true, r: 1.8 },
      ],
    });
  }

  /** practice / qualifying: delta to session best per timed lap + lap list with OUT / IN flagged */
  function renderTimedExtras(col, laps, cls, ix, hh) {
    const B = bests(hh), n = col.num, ob = B.ob.lap && B.ob.lap.v, pbl = B.pb[n] && B.pb[n].lap;
    const pts = laps.map((e, i) => [e.lap, cls[i] === 'timed' && isNum(e.sec) && isNum(ob) ? e.sec - ob : null, e]);
    const vals = pts.map((q) => q[1]).filter(isNum);
    const top = vals.length ? Math.max(1, Math.min(Math.max(...vals) * 1.1, (quantile(vals, 0.85) || 1) * 2)) : 2;
    col.charts.gaps.update({ xDomain: [1, Math.max(ix.maxLap, 2)], yDomain: [0, top], bands: raceBands(hh),
      series: [{ id: 'dsb', label: 'Gap to session best', color: '#4fc3f7', points: pts, markers: true, r: 2.2 }] });
    if (!col.R.laplist) return;
    const rows = laps.slice().reverse().slice(0, 80).map((e) => {
      const i = laps.indexOf(e), k = cls[i], timedLap = k === 'timed' && isNum(e.sec);
      const dpb = timedLap && isNum(pbl) ? e.sec - pbl : null, dsb = timedLap && isNum(ob) ? e.sec - ob : null;
      const sc = (j) => { const v = e.s && e.s[j]; return isNum(v) ? v.toFixed(3) : '–'; };
      const tag = k === 'timed' ? '' : `<span class="dd-lapflag ${k}">${k.toUpperCase()}</span>`;
      return `<tr class="${k !== 'timed' ? 'dd-sl' : ''}"><th>${e.lap}</th><td class="${timedLap && isNum(pbl) && e.sec <= pbl + 5e-4 ? 'pu' : ''}">${isNum(e.sec) ? fmtLap(e.sec) : '–'}</td><td>${dpb == null ? '' : dpb <= 5e-4 ? 'PB' : fmtDelta(dpb)}</td><td>${dsb == null ? '' : dsb <= 5e-4 ? 'BEST' : fmtDelta(dsb)}</td>
        <td>${sc(0)}</td><td>${sc(1)}</td><td>${sc(2)}</td><td><span class="dd-tdot sm" style="background:${tyreColor(e.compound)}">${tyreLetter(e.compound)}</span>${e.tyreAge != null ? ' ' + e.tyreAge : ''}</td><td>${tag}</td></tr>`;
    }).join('');
    col.R.laplist.innerHTML = rows ? `<table class="dd-tbl dd-lapl"><thead><tr><th>Lap</th><th>Time</th><th>Δ PB</th><th>Δ best</th><th>S1</th><th>S2</th><th>S3</th><th>Tyre</th><th></th></tr></thead><tbody>${rows}</tbody></table>` : '<div class="dd-none">No laps yet</div>';
  }

  // ------------------------------------------------------------ comparison (>= 2 pinned)
  function makeCompare() {
    const timed = sessionOf(store.state).timed;
    const r = h('div', 'dd-compare');
    r.innerHTML = `<div class="dd-cmp-h">Driver comparison <small data-r="sub"></small></div>
      <div class="dd-cmp-grid">
      <div class="dd-card"><div class="dd-h">Lap times</div><div class="dd-chart" data-r="c1"></div></div>
      <div class="dd-card"><div class="dd-h">Lap delta vs reference <small>positive = reference faster</small></div><div class="dd-chart" data-r="c2"></div></div>
      ${timed ? `<div class="dd-card"><div class="dd-h">Best-lap progression <small>running personal best</small></div><div class="dd-chart" data-r="c3"></div></div>
      <div class="dd-card"><div class="dd-h">Gap to session best <small>timed laps</small></div><div class="dd-chart" data-r="c4"></div></div>`
      : `<div class="dd-card"><div class="dd-h">Cumulative gap to reference <small>positive = behind reference</small></div><div class="dd-chart" data-r="c3"></div></div>
      <div class="dd-card"><div class="dd-h">Interval to car ahead</div><div class="dd-chart" data-r="c4"></div></div>`}
      <div class="dd-card dd-wide2"><div class="dd-h">Best-lap &amp; sector comparison</div><div data-r="tbl" class="dd-scroll"></div></div>
      </div>`;
    const R = refs(r);
    const base = { height: 210, left: 46, xName: 'Lap', xFmt: (v) => String(Math.round(v)), legend: true, series: [] };
    const charts = {
      c1: new Chart(R.c1, { ...base, yFmt: fmtLap }),
      c2: new Chart(R.c2, { ...base, yFmt: (v) => fmtDelta(v, 2) + 's', hlines: [{ y: 0, color: '#666' }] }),
      c3: new Chart(R.c3, { ...base, yFmt: timed ? fmtLap : (v) => fmtDelta(v, 1) + 's', hlines: timed ? [] : [{ y: 0, color: '#666' }] }),
      c4: new Chart(R.c4, { ...base, yFmt: (v) => v.toFixed(timed ? 3 : 1) + 's', yDomain: [0, null] }),
    };
    return {
      root: r, R, charts, destroy: () => Object.values(charts).forEach((c) => c.destroy()),
      render: () => renderCompare({ R, charts, timed }),
    };
  }
  function renderCompare({ R, charts, timed }) {
    const st = store.state, hh = hist(), ix = indexHistory(hh);
    const sty = teamStyles(st.cars), ref = pins[0], rc = st.cars[ref];
    R.sub.textContent = 'reference: ' + rc.tla;
    const xd = [1, Math.max(ix.maxLap, 2)];
    const mk = (n, id, pts, extra) => ({ id: id + n, label: st.cars[n].tla, color: sty[n].color, dash: sty[n].dash, points: pts, ...extra });
    const lapsRef = new Map(lapsOf(ref).map((e) => [e.lap, e]));
    const allSecs = [];
    const s1 = pins.map((n) => { const pts = lapsOf(n).map((e) => { allSecs.push(e.sec); return [e.lap, isNum(e.sec) ? e.sec : null, e]; }); return mk(n, 'p', pts, { markers: true, r: 1.8 }); });
    charts.c1.update({ series: s1, xDomain: xd, yDomain: paceDomain(allSecs), bands: raceBands(hh) });
    const s2 = [], s3 = [];
    for (const n of pins.slice(1)) {
      const d = [], g = [];
      for (const e of lapsOf(n)) {
        const re = lapsRef.get(e.lap);
        const ok = re && isNum(e.sec) && isNum(re.sec) && !e.pit && !re.pit;
        d.push([e.lap, ok ? e.sec - re.sec : null, e]);
        g.push([e.lap, re && isNum(e.gapLeader) && isNum(re.gapLeader) ? e.gapLeader - re.gapLeader : null, e]);
      }
      s2.push(mk(n, 'd', d, { markers: true, r: 2 }));
      s3.push(mk(n, 'g', g));
    }
    const dv = s2.flatMap((s) => s.points.map((q) => q[1])).filter(isNum);
    const lim = dv.length ? Math.max(1, Math.min(4, quantile(dv.map(Math.abs), 0.9) * 1.4)) : 2;
    charts.c2.update({ series: s2, xDomain: xd, yDomain: [-lim, lim], bands: raceBands(hh) });
    if (timed) {
      const obl = bests(hh).ob.lap && bests(hh).ob.lap.v;
      const prog = pins.map((n) => { let m = null; return mk(n, 'r', lapsOf(n).map((e) => { if (isNum(e.sec) && !e.pit && (m == null || e.sec < m)) m = e.sec; return [e.lap, m]; }), { markers: true, r: 1.8 }); });
      const pv = prog.flatMap((q) => q.points.map((z) => z[1])).filter(isNum);
      charts.c3.update({ series: prog, xDomain: xd, yDomain: pv.length ? [Math.min(...pv) - 0.2, Math.min(Math.max(...pv), Math.min(...pv) + 3) + 0.2] : [null, null] });
      charts.c4.update({ xDomain: xd, series: pins.map((n) => mk(n, 'g', lapsOf(n).map((e) => [e.lap, isNum(e.sec) && !e.pit && isNum(obl) ? e.sec - obl : null]), { markers: true, r: 1.8 })) });
    } else {
    charts.c3.update({ series: s3, xDomain: xd, bands: raceBands(hh) });
    charts.c4.update({
      xDomain: xd, bands: raceBands(hh),
      series: pins.map((n) => mk(n, 'i', lapsOf(n).map((e) => [e.lap, isNum(e.interval) ? e.interval : null]))),
    });
    }
    // table
    const B = bests(hh);
    const cell = (n, val, refVal, fmt, lowerBetter = true, dig = 3) => {
      const d = isNum(val) && isNum(refVal) && n !== ref ? val - refVal : null;
      const cls = d == null ? '' : (lowerBetter ? d < 0 : d > 0) ? 'gr' : d === 0 ? '' : 'rd';
      return `<td><b>${isNum(val) ? fmt(val) : '–'}</b>${d != null ? `<small class="${cls}">${fmtDelta(d, dig)}</small>` : ''}</td>`;
    };
    const head = pins.map((n) => `<th style="border-bottom:2px solid ${st.cars[n].color}">${esc(st.cars[n].tla)}${n === ref ? ' <small>ref</small>' : ''}</th>`).join('');
    const row = (label, f) => `<tr><th>${label}</th>${pins.map((n) => f(n)).join('')}</tr>`;
    const avg5 = (n) => { const v = lapsOf(n).filter((e) => isNum(e.sec) && !e.pit && String(e.track ?? '1') === '1').slice(-5).map((e) => e.sec); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
    R.tbl.innerHTML = `<table class="dd-tbl dd-cmp"><thead><tr><th></th>${head}</tr></thead><tbody>
      ${row('Best lap', (n) => cell(n, B.pb[n] && B.pb[n].lap, B.pb[ref] && B.pb[ref].lap, fmtLap))}
      ${row('on lap', (n) => `<td>${B.pb[n] && B.pb[n].lapNo ? B.pb[n].lapNo : '–'}</td>`)}
      ${[0, 1, 2].map((i) => row('Best S' + (i + 1), (n) => cell(n, B.pb[n] && B.pb[n].s[i], B.pb[ref] && B.pb[ref].s[i], (v) => v.toFixed(3)))).join('')}
      ${row('Ideal lap', (n) => { const p = B.pb[n]; const v = p && p.s.every(isNum) ? p.s.reduce((a, b) => a + b, 0) : null; const rp = B.pb[ref]; const rv = rp && rp.s.every(isNum) ? rp.s.reduce((a, b) => a + b, 0) : null; return cell(n, v, rv, fmtLap); })}
      ${row('Last 5 avg', (n) => cell(n, avg5(n), avg5(ref), fmtLap))}
      ${row('Last sectors', (n) => { const c = st.cars[n]; return `<td class="sc">${[0, 1, 2].map((i) => { const x = c.sectors && c.sectors[i]; const o = B.ob.s[i]; const p = B.pb[n] && B.pb[n].s[i]; return `<span class="${secCls(x && x.sec, p, o && o.v)}">${x && isNum(x.sec) ? x.sec.toFixed(2) : '–'}</span>`; }).join('')}</td>`; })}
      ${row('Top speed (ST)', (n) => cell(n, st.cars[n].speedTraps && st.cars[n].speedTraps.st, st.cars[ref].speedTraps && st.cars[ref].speedTraps.st, (v) => String(v), false, 0))}
      ${row(timed ? 'Timed laps' : 'Pit stops', (n) => `<td>${timed ? (st.cars[n].timedLaps ?? st.cars[n].lap ?? '–') : (st.cars[n].pits ?? 0)}</td>`)}
      ${row('Tyre', (n) => `<td><span class="dd-tdot sm" style="background:${tyreColor(st.cars[n].tyre && st.cars[n].tyre.compound)}">${tyreLetter(st.cars[n].tyre && st.cars[n].tyre.compound)}</span> ${st.cars[n].tyre && st.cars[n].tyre.age != null ? st.cars[n].tyre.age + 'L' : ''}</td>`)}
    </tbody></table>`;
  }

  // ------------------------------------------------------------ frame loop
  function frame() {
    if (el.offsetParent === null) return; // hidden: ResizeObserver re-triggers when shown
    const st = store.state;
    pins = st ? pinned() : [];
    const k = (st ? 's' : 'n') + '|' + sessionOf(st).kind + '|' + pins.join(',') + '|' + (!pins.length && st ? (st.order || []).join(',') : '') + '|' + store.selected.size;
    if (k !== key) { key = k; rebuild(); dirty.state = dirty.fast = dirty.hist = true; }
    if (!cols.length) return;
    if (dirty.state || dirty.hist) { cols.forEach((c) => c.renderState()); }
    if (dirty.fast || dirty.state) { cols.forEach((c) => { c.renderGauges(); }); }
    if (dirty.fast) cols.forEach((c) => c.renderStrip());
    if (dirty.hist || dirty.state) {
      if (dirty.hist || !frame.charted) { cols.forEach((c) => c.renderCharts(pins.length === 1)); if (cmp) cmp.render(); frame.charted = true; }
    }
    dirty.state = dirty.fast = dirty.hist = false;
  }
  const schedule = rafThrottle(frame, 'deepdive');
  const unsub = store.subscribe((kind) => {
    if (kind === 'reset') { for (const k2 of Object.keys(bufs)) delete bufs[k2]; key = '~'; dirty.state = dirty.hist = dirty.fast = true; frame.charted = false; schedule(); return; }
    if (kind === 'fast') {
      for (const n of pins) pushSample(n);
      dirty.fast = true;
    } else if (kind === 'state') dirty.state = true;
    else if (kind === 'history') { dirty.hist = true; frame.charted = false; } else if (kind === 'selection') { dirty.state = dirty.hist = dirty.fast = true; frame.charted = false; } else return;
    schedule();
  });
  const ro = new ResizeObserver(() => { dirty.state = dirty.hist = dirty.fast = true; frame.charted = false; schedule(); });
  ro.observe(el);
  schedule();
  return { destroy() { unsub(); ro.disconnect(); schedule.cancel(); cols.forEach((c) => c.destroy()); if (cmp) cmp.destroy(); } };
}
