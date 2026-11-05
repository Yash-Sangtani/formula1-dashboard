// Race analytics panel (stage 4). export function mount(el, store)
import { reportError } from '../store.js';
import {
  Chart, COLORS, TYRE, esc, h, rafThrottle, isNum, fmtLap, fmtDelta, median, quantile, linreg,
  tyreColor, tyreLetter, indexHistory, teamStyles, raceBands, stintsFromLaps, heatColor,
} from '../charts.js';

const FUEL_S_PER_LAP = 0.035; // fuel-burn correction applied to lap times
const KINDS = {
  flag: ['Flags', '#f1c40f'], sc: ['Safety car', '#ff9f0a'], vsc: ['VSC', '#ffcc66'], pit: ['Pit', '#4fc3f7'],
  fastest: ['Fastest lap', COLORS.purple], retire: ['Retire', '#ff453a'], penalty: ['Penalty', '#ff6fa8'],
  overtake: ['Overtake', '#2ecc71'], radio: ['Radio', '#8a8a8a'],
};
const TABS = [
  ['trace', 'Race trace'], ['pos', 'Positions'], ['strat', 'Strategy'], ['pace', 'Pace'],
  ['pits', 'Pit stops'], ['events', 'Events'], ['speeds', 'Speeds & bests'],
];
const intFmt = (v) => String(Math.round(v));

export function mount(el, store) {
  el.classList.add('an');
  const bar = h('div', 'an-tabs');
  const body = h('div', 'an-body');
  el.append(bar, body);
  let active = 'trace';
  try { active = localStorage.getItem('f1.an.tab') || 'trace'; } catch { /* ignore */ }
  if (!TABS.some((t) => t[0] === active)) active = 'trace';
  let hlLap = null;      // lap highlighted via event click
  let refDriver = 'leader';
  const views = {};

  const S = () => store.state;
  const HIST = () => store.history;
  const hasLaps = () => { const hh = HIST(); return !!(hh && hh.laps && indexHistory(hh).maxLap >= 1); };
  const cars = () => (S() && S().cars) || {};
  const order = () => (S() && S().order) || Object.keys(cars());
  const selected = () => store.selected;
  const lapsOf = (n) => (HIST() && HIST().laps && HIST().laps[n]) || [];
  const pick = (id) => store.select(id, { additive: true });

  TABS.forEach(([id, label]) => {
    const b = h('button', 'an-tab', label); b.type = 'button'; b.dataset.tab = id;
    b.addEventListener('click', () => setTab(id));
    bar.appendChild(b);
  });
  const lapBadge = h('div', 'an-lap');
  bar.appendChild(lapBadge);

  function setTab(id) {
    active = id;
    try { localStorage.setItem('f1.an.tab', id); } catch { /* ignore */ }
    render(true);
  }

  // ---------------------------------------------------------- shared bits
  function empty(root, show, msg) {
    let m = root.querySelector(':scope > .an-msg');
    if (!m) { m = h('div', 'an-msg'); root.prepend(m); }
    m.textContent = msg || 'Waiting for lap data — charts appear after the first completed lap.';
    m.style.display = show ? '' : 'none';
    Array.from(root.children).forEach((c) => { if (c !== m) c.classList.toggle('an-hidden', show); });
  }
  function lapMarks() { return hlLap != null ? [{ x: hlLap, color: COLORS.accent, label: 'L' + hlLap, dash: [] }] : []; }
  function driverSeries(n, pts, extra = {}) {
    const c = cars()[n] || {}, sty = teamStyles(cars())[n] || { color: '#888' };
    const sel = selected().has(n);
    return { id: n, label: c.tla || n, short: c.tla || n, color: sty.color, dash: sty.dash, points: pts, width: sel ? 3 : 1.5, ...extra };
  }
  const xRange = () => [1, Math.max(indexHistory(HIST()).maxLap, 2)];
  function legendSync(chart) { return (id, hidden) => { chart._userHidden = new Set(hidden); }; }
  function card(title, extra) { const c = h('div', 'an-card'); if (title) c.innerHTML = `<div class="an-h">${title}${extra ? ` <small>${extra}</small>` : ''}</div>`; return c; }
  const tc = (n) => (cars()[n] && cars()[n].tla) || n;
  const gapOf = (e) => (e && isNum(e.gapLeader) ? e.gapLeader : e && e.pos === 1 ? 0 : null);
  function orderedNums() { return order().filter((n) => cars()[n]); }

  // ---------------------------------------------------------- (a) race trace
  views.trace = (() => {
    const root = h('div', 'an-view');
    const tb = h('div', 'an-tools');
    tb.innerHTML = '<label>Reference <select data-r="ref"></select></label><label class="an-chk"><input type="checkbox" data-r="clip" checked> clip lapped cars</label><span class="an-hint">hover a line to highlight · click to pin driver · double-click a legend chip to isolate</span>';
    const host = h('div', 'an-chart');
    root.append(tb, host);
    const sel = tb.querySelector('[data-r=ref]'), clip = tb.querySelector('[data-r=clip]');
    let chart = null, lastOpts = '';
    sel.addEventListener('change', () => { refDriver = sel.value; views.trace.render(); });
    clip.addEventListener('change', () => views.trace.render());
    return {
      root,
      render() {
        empty(root, !hasLaps());
        if (!hasLaps()) return;
        const sig = order().join(',') + '|' + refDriver;
        if (sig !== lastOpts) {
          lastOpts = sig;
          sel.innerHTML = '<option value="leader">Leader</option>' + orderedNums().map((n) => `<option value="${esc(n)}">${esc(tc(n))}</option>`).join('');
        }
        if (refDriver !== 'leader' && !cars()[refDriver]) refDriver = 'leader';
        sel.value = refDriver;
        const ix = indexHistory(HIST());
        if (!chart) {
          chart = new Chart(host, {
            legend: true, endLabels: true, yInvert: true, right: 38, left: 48, tooltipMode: 'x', xName: 'Lap', xFmt: intFmt,
            onClick: (s) => s && pick(s.id), onLegend: legendSync(),
          });
        }
        const vals = [];
        const series = orderedNums().map((n) => {
          const pts = lapsOf(n).map((e) => {
            const r = refDriver === 'leader' ? 0 : gapOf(ix.byLap.get(e.lap).get(refDriver));
            const g = gapOf(e);
            const v = isNum(g) && isNum(r) ? g - r : null;
            if (v != null) vals.push(v);
            return [e.lap, v, e];
          });
          return driverSeries(n, pts);
        });
        const q = quantile(vals, 0.92);
        const hiCap = clip.checked && q != null ? Math.max(30, q * 1.3) : null;
        const mn = vals.length ? Math.min(...vals) : 0;
        chart.update({
          series, xDomain: xRange(), yDomain: [Math.min(0, mn - 1), hiCap], bands: raceBands(HIST()), vlines: lapMarks(),
          yFmt: (v) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(0) + 's', yLabel: refDriver === 'leader' ? 'gap to leader' : 'gap to ' + tc(refDriver),
        });
        chart.hidden = chart._userHidden || chart.hidden;
      },
    };
  })();

  // ---------------------------------------------------------- (b) positions
  views.pos = (() => {
    const root = h('div', 'an-view');
    const host = h('div', 'an-chart');
    root.append(h('div', 'an-tools', '<span class="an-hint">position by lap · dashed = second car of the same team</span>'), host);
    let chart = null;
    return {
      root,
      render() {
        empty(root, !hasLaps());
        if (!hasLaps()) return;
        if (!chart) chart = new Chart(host, { legend: true, endLabels: true, yInvert: true, right: 38, left: 30, tooltipMode: 'x', xName: 'Lap', xFmt: intFmt, onClick: (s) => s && pick(s.id), onLegend: legendSync() });
        const n = orderedNums().length || 20;
        chart.update({
          series: orderedNums().map((k) => driverSeries(k, lapsOf(k).map((e) => [e.lap, e.pos ?? null, e]))),
          xDomain: xRange(), yDomain: [0.5, n + 0.5], yTicks: Array.from({ length: n }, (_, i) => ({ v: i + 1, label: 'P' + (i + 1) })).filter((t) => t.v % (n > 12 ? 2 : 1) === 1 || t.v === 1 || n <= 12),
          bands: raceBands(HIST()), vlines: lapMarks(), yFmt: (v) => 'P' + Math.round(v),
        });
        chart.hidden = chart._userHidden || chart.hidden;
      },
    };
  })();

  // ---------------------------------------------------------- (c) tyre strategy
  views.strat = (() => {
    const root = h('div', 'an-view');
    const leg = h('div', 'an-tools', Object.entries(TYRE).map(([k, c]) => `<span class="an-tyre"><i style="background:${c}"></i>${k[0] + k.slice(1).toLowerCase()}</span>`).join('') + '<span class="an-tyre"><b class="an-ring"></b>pit stop</span><span class="an-hint">bar label = stint length (laps)</span>');
    const host = h('div', 'an-chart an-chart-auto');
    root.append(leg, host);
    let chart = null;
    return {
      root,
      render() {
        const st = S();
        const nums = orderedNums();
        empty(root, !st || !nums.length, 'Waiting for session data…');
        if (!st || !nums.length) return;
        const total = (st.lap && st.lap.total) || Math.max(indexHistory(HIST()).maxLap, (st.lap && st.lap.current) || 0, 10);
        const pts = [], pits = [];
        nums.forEach((n, i) => {
          const c = st.cars[n];
          let stints = (c.stints || []).map((s) => ({ compound: s.compound, a: s.startLap || 0, b: (s.startLap || 0) + (s.laps || 0), isNew: s.new }));
          if (!stints.length) stints = stintsFromLaps(lapsOf(n)).map((s) => ({ compound: s.compound, a: s.start - 1, b: s.end }));
          stints.forEach((s, k) => {
            const len = s.b - s.a;
            pts.push([s.a, s.b, i, { color: tyreColor(s.compound), text: len > 0 ? String(len) : '', num: n, tip: `<b>${esc(c.tla)}</b> stint ${k + 1}<div>${esc(s.compound || '?')}${s.isNew === false ? ' (used)' : ''} · laps ${s.a + 1}–${s.b} (${len})</div>` }]);
            if (k < stints.length - 1) pits.push([s.b, i, { num: n, tip: `<b>${esc(c.tla)}</b> pit stop, end of lap ${s.b}` }]);
          });
        });
        if (!chart) {
          chart = new Chart(host, {
            left: 64, right: 12, yInvert: true, xName: 'Lap', xFmt: intFmt, onClick: (s, p) => { if (p) { const m = p[3] || p[2]; if (m && m.num) pick(m.num); } },
            tooltip: (s, p) => (p[3] ? p[3].tip : p[2] && p[2].tip),
          });
        }
        host.style.height = Math.max(240, nums.length * 24 + 40) + 'px';
        chart.update({
          series: [
            { id: 'stints', type: 'hbar', barH: 0.72, color: '#888', points: pts, legend: false },
            { id: 'pits', type: 'scatter', color: '#fff', r: 3, points: pits, legend: false },
          ],
          xDomain: [0, total], yDomain: [-0.5, nums.length - 0.5],
          yTicks: nums.map((n, i) => ({ v: i, label: `${st.cars[n].pos ?? ''} ${st.cars[n].tla}` })),
          vlines: [...(st.lap && st.lap.current ? [{ x: st.lap.current, color: COLORS.accent, label: 'NOW', dash: [] }] : []), ...lapMarks()],
          bands: raceBands(HIST()), gridX: true,
        });
      },
    };
  })();

  // ---------------------------------------------------------- (d) lap-time / stint pace
  views.pace = (() => {
    const root = h('div', 'an-view');
    const tb = h('div', 'an-tools');
    tb.innerHTML = `<label class="an-chk"><input type="checkbox" data-r="excl" checked> exclude pit / SC / out laps</label>
      <label class="an-chk"><input type="checkbox" data-r="clip" checked> clip outliers</label>
      <label class="an-chk"><input type="checkbox" data-r="fuel"> fuel-corrected (${FUEL_S_PER_LAP}s/lap)</label>
      <label>Colour <select data-r="col"><option value="team">team</option><option value="tyre">tyre</option></select></label>`;
    const host = h('div', 'an-chart an-chart-s');
    const tcard = card('Stint pace', 'average, best and degradation slope (s/lap, + = slower with age)');
    const tw = h('div', 'an-scroll'); tcard.appendChild(tw);
    root.append(tb, host, tcard);
    const R = {}; tb.querySelectorAll('[data-r]').forEach((e) => { R[e.dataset.r] = e; e.addEventListener('change', () => views.pace.render()); });
    let chart = null;
    return {
      root,
      render() {
        empty(root, !hasLaps());
        if (!hasLaps()) return;
        const excl = R.excl.checked, clip = R.clip.checked, fuel = R.fuel.checked, byTyre = R.col.value === 'tyre';
        const adj = (e) => e.sec + (fuel ? FUEL_S_PER_LAP * (e.lap - 1) : 0);
        const valid = (e, prev) => isNum(e.sec) && e.lap > 1 && (!excl || (!e.pit && !(prev && prev.pit) && String(e.track ?? '1') === '1'));
        const all = [];
        const perCar = {};
        for (const n of orderedNums()) {
          const arr = lapsOf(n);
          perCar[n] = arr.map((e, i) => ({ e, ok: valid(e, arr[i - 1]) }));
          perCar[n].forEach((q) => { if (q.ok) all.push(adj(q.e)); });
        }
        const med = median(all);
        const lim = clip && med ? med * 1.07 : Infinity;
        if (!chart) {
          chart = new Chart(host, {
            legend: true, left: 48, xName: 'Lap', xFmt: intFmt, onClick: (s) => s && pick(s.id), onLegend: legendSync(),
            tooltip: (s, p) => `<b>${esc(s.label)}</b> lap ${p[0]}<div>${fmtLap(p[2].raw)} ${fuel ? '→ ' + fmtLap(p[1]) + ' corr.' : ''}</div><div>${esc(p[2].e.compound || '')} · age ${p[2].e.tyreAge ?? '–'}</div>`,
          });
        }
        const series = orderedNums().map((n) => driverSeries(n, perCar[n].filter((q) => q.ok && adj(q.e) <= lim).map((q) => [q.e.lap, adj(q.e), { e: q.e, raw: q.e.sec, color: byTyre ? tyreColor(q.e.compound) : undefined }]), { type: 'scatter', r: selected().has(n) ? 3.6 : 2.4 }));
        const vis = series.flatMap((s) => s.points.map((p) => p[1]));
        chart.update({
          series, xDomain: xRange(), yDomain: vis.length ? [Math.min(...vis) - 0.3, clip && med ? Math.max(...vis) + 0.3 : null] : [null, null],
          bands: raceBands(HIST()), vlines: lapMarks(), yFmt: fmtLap,
        });
        chart.hidden = chart._userHidden || chart.hidden;
        // stint table
        const rows = [];
        for (const n of orderedNums()) {
          const arr = lapsOf(n), idx = new Map(arr.map((e, i) => [e, i]));
          stintsFromLaps(arr).forEach((s, k) => {
            const good = s.laps.filter((e) => valid(e, arr[idx.get(e) - 1]) && adj(e) <= lim);
            if (!good.length) { rows.push({ n, k, s, cnt: 0 }); return; }
            const ts = good.map(adj), avg = ts.reduce((a, b) => a + b, 0) / ts.length;
            const reg = good.length >= 4 ? linreg(good.map((e) => e.lap), ts) : null;
            rows.push({ n, k, s, cnt: good.length, avg, best: Math.min(...ts), slope: reg && reg.slope });
          });
        }
        const sorted = new Map(orderedNums().map((n, i) => [n, i]));
        rows.sort((a, b) => sorted.get(a.n) - sorted.get(b.n) || a.k - b.k);
        tw.innerHTML = `<table class="an-tbl"><thead><tr><th>P</th><th>Driver</th><th>Stint</th><th>Tyre</th><th>Laps</th><th>Avg</th><th>Best</th><th>Deg s/lap</th></tr></thead><tbody>${rows.map((r) => {
          const c = cars()[r.n];
          const d = r.slope;
          return `<tr data-n="${esc(r.n)}"><td>${c.pos ?? ''}</td><td><i class="an-dot" style="background:${c.color}"></i>${esc(c.tla)}</td><td>${r.k + 1}</td>
            <td><span class="an-tdot" style="background:${tyreColor(r.s.compound)}">${tyreLetter(r.s.compound)}</span></td><td>${r.cnt}/${r.s.laps.length}</td>
            <td>${r.cnt ? fmtLap(r.avg) : '–'}</td><td>${r.cnt ? fmtLap(r.best) : '–'}</td>
            <td class="${isNum(d) ? (d > 0.08 ? 'bad' : d < 0 ? 'good' : '') : ''}">${isNum(d) ? (d >= 0 ? '+' : '−') + Math.abs(d).toFixed(3) : '–'}</td></tr>`;
        }).join('')}</tbody></table>`;
      },
    };
  })();
  views.pace.root.addEventListener('click', (e) => { const tr = e.target.closest('tr[data-n]'); if (tr) pick(tr.dataset.n); });

  // ---------------------------------------------------------- (e) pit stops
  views.pits = (() => {
    const root = h('div', 'an-view');
    const sum = h('div', 'an-tools');
    const bcard = card('Pit stop durations', 'stationary + lane time as reported');
    const host = h('div', 'an-chart an-chart-s'); bcard.appendChild(host);
    const tcard = card('Stops &amp; undercut / overcut check', 'positions gained/lost over the 3 laps after the stop');
    const tw = h('div', 'an-scroll'); tcard.appendChild(tw);
    root.append(sum, bcard, tcard);
    let chart = null, sortKey = 'lap', sortDir = 1;
    tw.addEventListener('click', (e) => {
      const th = e.target.closest('th[data-k]');
      if (th) { if (sortKey === th.dataset.k) sortDir = -sortDir; else { sortKey = th.dataset.k; sortDir = 1; } views.pits.render(); return; }
      const tr = e.target.closest('tr[data-lap]');
      if (tr) { hlLap = +tr.dataset.lap === hlLap ? null : +tr.dataset.lap; render(); }
    });
    return {
      root,
      render() {
        const st = S();
        let stops = ((st && st.pitStops) || []).map((p) => ({ ...p }));
        if (!stops.length && hasLaps()) {
          for (const n of orderedNums()) for (const e of lapsOf(n)) if (e.pit) stops.push({ num: n, lap: e.lap, durationSec: null });
        }
        empty(root, !stops.length, 'No pit stops yet.');
        if (!stops.length) return;
        const ix = indexHistory(HIST()), maxLap = ix.maxLap;
        for (const p of stops) {
          const before = p.lap > 1 ? ix.byLap.get(p.lap - 1)?.get(p.num)?.pos : st.cars[p.num]?.grid;
          const target = Math.min(p.lap + 3, maxLap);
          const after = ix.byLap.get(target)?.get(p.num)?.pos;
          p.before = before ?? null; p.after = after ?? null; p.span = target - p.lap;
          p.delta = isNum(before) && isNum(after) ? before - after : null;
          p.complete = p.lap + 3 <= maxLap;
          p.tla = tc(p.num);
        }
        const durs = stops.map((p) => p.durationSec).filter(isNum);
        const dmed = median(durs);
        sum.innerHTML = `<span class="an-kpi"><b>${stops.length}</b> stops</span>` + (durs.length ? `<span class="an-kpi"><b>${Math.min(...durs).toFixed(1)}s</b> fastest (${esc(stops.find((p) => p.durationSec === Math.min(...durs)).tla)})</span><span class="an-kpi"><b>${(durs.reduce((a, b) => a + b, 0) / durs.length).toFixed(1)}s</b> average</span>` : '<span class="an-hint">durations not available</span>');
        const sorter = { lap: (p) => p.lap, driver: (p) => p.tla, dur: (p) => p.durationSec ?? 1e9, delta: (p) => p.delta ?? -1e9 }[sortKey] || ((p) => p.lap);
        const list = stops.slice().sort((a, b) => { const x = sorter(a), y = sorter(b); return (x > y ? 1 : x < y ? -1 : 0) * sortDir || a.lap - b.lap; });
        const th = (k, t) => `<th data-k="${k}" class="sortable${sortKey === k ? ' on' : ''}">${t}${sortKey === k ? (sortDir > 0 ? ' ▲' : ' ▼') : ''}</th>`;
        tw.innerHTML = `<table class="an-tbl"><thead><tr>${th('driver', 'Driver')}${th('lap', 'Lap')}${th('dur', 'Time')}<th>Tyres</th><th>Pos before</th><th>Pos +3L</th>${th('delta', 'Δ pos')}<th>Hint</th></tr></thead><tbody>${list.map((p) => {
          let hint = '<span class="muted">–</span>';
          if (isNum(p.delta)) {
            const d = p.delta;
            hint = !p.complete ? '<span class="muted">pending…</span>'
              : d > 0 ? `<span class="good">+${d} · undercut/overcut paid off</span>` : d < 0 ? `<span class="bad">${d} · lost track position</span>` : '<span class="muted">position held</span>';
          }
          const slow = isNum(p.durationSec) && dmed && p.durationSec > dmed + 3;
          return `<tr data-lap="${p.lap}" class="${hlLap === p.lap ? 'hl' : ''}"><td><i class="an-dot" style="background:${esc(st.cars[p.num]?.color || '#888')}"></i>${esc(p.tla)}</td><td>${p.lap}</td>
            <td class="${slow ? 'bad' : ''}">${isNum(p.durationSec) ? p.durationSec.toFixed(1) + 's' : '–'}</td>
            <td>${p.fromCompound ? `<span class="an-tdot" style="background:${tyreColor(p.fromCompound)}">${tyreLetter(p.fromCompound)}</span>→<span class="an-tdot" style="background:${tyreColor(p.toCompound)}">${tyreLetter(p.toCompound)}</span>` : '–'}</td>
            <td>${p.before ?? '–'}</td><td>${p.after ?? '–'}${!p.complete && p.after ? '*' : ''}</td><td class="${isNum(p.delta) ? (p.delta > 0 ? 'good' : p.delta < 0 ? 'bad' : '') : ''}">${isNum(p.delta) ? (p.delta > 0 ? '+' : '') + p.delta : '–'}</td><td>${hint}</td></tr>`;
        }).join('')}</tbody></table>`;
        // chart
        const withDur = stops.filter((p) => isNum(p.durationSec)).sort((a, b) => a.lap - b.lap);
        bcard.style.display = withDur.length ? '' : 'none';
        if (!withDur.length) return;
        if (!chart) chart = new Chart(host, { left: 40, bottom: 34, yFmt: (v) => v.toFixed(0) + 's', tooltip: (s, p) => p[2] && p[2].tip, onClick: (s, p) => p[2] && p[2].num && pick(p[2].num) });
        const cap = Math.min(Math.max(...withDur.map((p) => p.durationSec)) * 1.05, (dmed || 25) * 2.5);
        chart.update({
          series: [{ id: 'dur', type: 'bar', barWidth: 0.7, color: '#888', legend: false, points: withDur.map((p, i) => [i, Math.min(p.durationSec, cap), { color: st.cars[p.num]?.color || '#888', num: p.num, tip: `<b>${esc(p.tla)}</b> lap ${p.lap}<div>${p.durationSec.toFixed(2)}s</div>` }]) }],
          xDomain: [-0.6, withDur.length - 0.4], yDomain: [Math.max(0, Math.min(...withDur.map((p) => p.durationSec)) - 4), cap + 0.5],
          xTicks: withDur.map((p, i) => ({ v: i, label: p.tla })), xMinGap: 22,
          hlines: dmed ? [{ y: dmed, color: 'rgba(255,255,255,.35)' }] : [],
        });
      },
    };
  })();

  // ---------------------------------------------------------- (f) events
  const evFilter = new Set(Object.keys(KINDS).filter((k) => k !== 'radio'));
  views.events = (() => {
    const root = h('div', 'an-view');
    const chips = h('div', 'an-tools an-chips');
    const host = h('div', 'an-chart an-chart-s');
    const lcard = card('Event log', 'click an event to highlight its lap in the other charts');
    const list = h('div', 'an-evlist'); lcard.appendChild(list);
    root.append(chips, host, lcard);
    let chart = null;
    chips.addEventListener('click', (e) => { const b = e.target.closest('[data-k]'); if (!b) return; evFilter.has(b.dataset.k) ? evFilter.delete(b.dataset.k) : evFilter.add(b.dataset.k); views.events.render(); });
    list.addEventListener('click', (e) => { const r = e.target.closest('[data-lap]'); if (!r) return; hlLap = +r.dataset.lap === hlLap ? null : +r.dataset.lap; render(); });
    return {
      root,
      render() {
        const evs = (HIST() && HIST().events) || [];
        empty(root, !evs.length, 'No race events yet.');
        if (!evs.length) return;
        const counts = {}; evs.forEach((e) => { counts[e.kind] = (counts[e.kind] || 0) + 1; });
        const kinds = Object.keys(KINDS);
        chips.innerHTML = kinds.map((k) => `<button type="button" data-k="${k}" class="an-fchip${evFilter.has(k) ? ' on' : ''}" style="--c:${KINDS[k][1]}"><i></i>${KINDS[k][0]} <small>${counts[k] || 0}</small></button>`).join('') + (hlLap != null ? `<span class="an-hint">lap ${hlLap} highlighted — click again to clear</span>` : '');
        const shown = evs.filter((e) => evFilter.has(e.kind));
        const rows = kinds.filter((k) => evFilter.has(k));
        if (!chart) chart = new Chart(host, { left: 82, bottom: 22, yInvert: true, xName: 'Lap', xFmt: intFmt, onClick: (s, p) => { if (p[2] && p[2].lap != null) { hlLap = p[2].lap === hlLap ? null : p[2].lap; render(); } }, tooltip: (s, p) => `<b>Lap ${p[0]}</b> · ${KINDS[p[2].kind]?.[0] || p[2].kind}<div>${esc(p[2].text || '')}</div>` });
        const maxLap = Math.max(indexHistory(HIST()).maxLap, ...shown.map((e) => e.lap || 0), 2);
        chart.update({
          series: rows.map((k, i) => ({ id: k, type: 'scatter', color: KINDS[k][1], r: 3.5, legend: false, points: shown.filter((e) => e.kind === k && e.lap != null).map((e) => [e.lap, i, e]) })),
          xDomain: [0, maxLap], yDomain: [-0.5, Math.max(0.5, rows.length - 0.5)], yTicks: rows.map((k, i) => ({ v: i, label: KINDS[k][0] })), gridX: true,
          bands: raceBands(HIST()), vlines: lapMarks(),
        });
        const rev = shown.slice().reverse();
        list.innerHTML = rev.length ? rev.map((e) => {
          const k = KINDS[e.kind] || [e.kind, '#888'];
          const c = e.num && cars()[e.num];
          return `<div class="an-ev${e.lap === hlLap ? ' hl' : ''}" data-lap="${e.lap ?? ''}" style="--c:${k[1]}"><span class="an-ev-lap">${e.lap != null ? 'L' + e.lap : ''}</span><span class="an-ev-k">${esc(k[0])}</span>${c ? `<span class="an-ev-d" style="color:${c.color}">${esc(c.tla)}</span>` : ''}<span class="an-ev-t">${esc(e.text || '')}</span></div>`;
        }).join('') : '<div class="an-none">No events match the filters.</div>';
      },
    };
  })();

  // ---------------------------------------------------------- (g) speeds & bests
  views.speeds = (() => {
    const root = h('div', 'an-view');
    const grid = h('div', 'an-grid');
    root.appendChild(grid);
    function ranked(title, items, fmt, best) {
      // items: [{n, v}] sorted ascending/descending by caller
      const top = items[0] && items[0].v;
      const lo = items.length ? items[items.length - 1].v : 0;
      return `<div class="an-card"><div class="an-h">${title}</div>${items.length ? items.map((it, i) => {
        const w = best === 'max' ? (lo && top ? 35 + 65 * (it.v - lo + 1) / (top - lo + 1) : 100) : (top ? 35 + 65 * (1 - (it.v - top) / Math.max(top * 0.03, 0.001)) : 100);
        const c = cars()[it.n];
        return `<div class="an-rank" data-n="${esc(it.n)}"><span class="an-rk">${i + 1}</span><span class="an-rn"><i class="an-dot" style="background:${c.color}"></i>${esc(c.tla)}</span><span class="an-rbar"><i style="width:${Math.max(8, Math.min(100, w))}%;background:${c.color}"></i></span><span class="an-rv ${i === 0 ? 'pu' : ''}">${fmt(it.v)}</span></div>`;
      }).join('') : '<div class="an-none">no data</div>'}</div>`;
    }
    grid.addEventListener('click', (e) => { const r = e.target.closest('[data-n]'); if (r) pick(r.dataset.n); });
    return {
      root,
      render() {
        const st = S();
        empty(root, !st || !Object.keys(cars()).length, 'Waiting for session data…');
        if (!st) return;
        const nums = orderedNums();
        const trap = (k) => nums.map((n) => ({ n, v: cars()[n].speedTraps && cars()[n].speedTraps[k] })).filter((x) => isNum(x.v)).sort((a, b) => b.v - a.v).slice(0, 10);
        const pb = {};
        for (const n of nums) {
          const p = pb[n] = { s: [null, null, null], lap: null };
          for (const e of lapsOf(n)) {
            for (let i = 0; i < 3; i++) { const v = e.s && e.s[i]; if (isNum(v) && (p.s[i] == null || v < p.s[i])) p.s[i] = v; }
            if (isNum(e.sec) && !e.pit && (p.lap == null || e.sec < p.lap)) p.lap = e.sec;
          }
          if (cars()[n].best && isNum(cars()[n].best.sec) && (p.lap == null || cars()[n].best.sec < p.lap)) p.lap = cars()[n].best.sec;
        }
        const sec = (i) => nums.map((n) => ({ n, v: pb[n].s[i] })).filter((x) => isNum(x.v)).sort((a, b) => a.v - b.v).slice(0, 10);
        const laps = nums.map((n) => ({ n, v: pb[n].lap })).filter((x) => isNum(x.v)).sort((a, b) => a.v - b.v);
        const ideal = nums.map((n) => ({ n, v: pb[n].s.every(isNum) ? pb[n].s.reduce((a, b) => a + b, 0) : null })).filter((x) => isNum(x.v)).sort((a, b) => a.v - b.v).slice(0, 10);
        const spd = (v) => v + ' km/h', secf = (v) => v.toFixed(3);
        const lapsFull = laps.length ? `<div class="an-card an-card-wide"><div class="an-h">Best laps <small>gap to fastest · potential from ideal lap</small></div><div class="an-scroll"><table class="an-tbl"><thead><tr><th>#</th><th>Driver</th><th>Best lap</th><th>Gap</th><th>Ideal</th><th>Potential</th><th>Top speed</th></tr></thead><tbody>${laps.map((x, i) => {
          const p = pb[x.n], id = p.s.every(isNum) ? p.s.reduce((a, b) => a + b, 0) : null, c = cars()[x.n];
          return `<tr data-n="${esc(x.n)}"><td>${i + 1}</td><td><i class="an-dot" style="background:${c.color}"></i>${esc(c.tla)}</td><td class="${i === 0 ? 'pu' : ''}">${fmtLap(x.v)}</td><td>${i ? fmtDelta(x.v - laps[0].v) : ''}</td><td>${id ? fmtLap(id) : '–'}</td><td>${id ? fmtDelta(id - x.v) : '–'}</td><td>${(c.speedTraps && c.speedTraps.st) ?? '–'}</td></tr>`;
        }).join('')}</tbody></table></div></div>` : '';
        grid.innerHTML = ranked('Speed trap (ST)', trap('st'), spd, 'max') + ranked('Finish line (FL)', trap('fl'), spd, 'max') + ranked('Intermediate 1', trap('i1'), spd, 'max') + ranked('Intermediate 2', trap('i2'), spd, 'max')
          + ranked('Best sector 1', sec(0), secf, 'min') + ranked('Best sector 2', sec(1), secf, 'min') + ranked('Best sector 3', sec(2), secf, 'min') + ranked('Ideal laps', ideal, fmtLap, 'min') + lapsFull;
      },
    };
  })();

  // ---------------------------------------------------------- main render loop
  for (const id of Object.keys(views)) { views[id].root.classList.add('an-hidden-view'); body.appendChild(views[id].root); }
  function render(force) {
    if (el.offsetParent === null) return;
    for (const b of bar.querySelectorAll('.an-tab')) b.classList.toggle('on', b.dataset.tab === active);
    for (const [id, v] of Object.entries(views)) v.root.classList.toggle('an-hidden-view', id !== active);
    const st = S();
    lapBadge.textContent = st && st.lap && st.lap.current ? `Lap ${st.lap.current}${st.lap.total ? '/' + st.lap.total : ''}` : '';
    try { views[active].render(force); } catch (e) { reportError('analytics:' + active, e); }
  }
  const schedule = rafThrottle(() => render(), 'analytics');
  const unsub = store.subscribe((k) => { if (k === 'state' || k === 'history' || k === 'selection') schedule(); });
  const ro = new ResizeObserver(() => schedule());
  ro.observe(el);
  schedule();
  return { destroy() { unsub(); ro.disconnect(); schedule.cancel(); }, setTab };
}
