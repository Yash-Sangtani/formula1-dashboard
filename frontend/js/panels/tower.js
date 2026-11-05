import { setText, setClass, setStyle, timeClass, compound, isNum } from '../util.js';
import { sessionOf } from '../session.js';
import { bestsOf, activityOf } from '../derive.js';

const ROW_H = 32;
const RACE_HEAD = `<span class="c-pos">P</span><span class="c-chg"></span><span class="c-bar"></span><span class="c-tla">DRV</span>
<span class="c-gap">GAP</span><span class="c-int">INT</span><span class="c-last">LAST</span><span class="c-best">BEST</span>
<span class="c-sec"><i>S1</i><i>S2</i><i>S3</i></span><span class="c-tyre">TYRE</span><span class="c-pit">PIT</span><span class="c-drs">DRS</span><span class="c-spd">KPH</span>`;
const RACE_BODY = `<span class="c-pos"></span><span class="c-chg"></span><span class="c-bar"></span><span class="c-tla"></span>
<span class="c-gap"></span><span class="c-int"></span><span class="c-last"></span><span class="c-best"></span>
<span class="c-sec"><i></i><i></i><i></i></span><span class="c-tyre"><b></b><em></em><small></small></span>
<span class="c-pit"></span><span class="c-drs"><i></i></span><span class="c-spd"></span>`;
// practice / qualifying: best lap + gap to fastest are the primary columns
const T_HEAD = (q) => `<span class="c-pos">P</span><span class="c-bar"></span><span class="c-tla">DRV</span>
<span class="c-best">BEST</span><span class="c-gap" title="Gap to fastest">GAP</span>${q ? '<span class="c-q"><i>Q1</i><i>Q2</i><i>Q3</i></span>' : ''}
<span class="c-sec" title="Personal best sectors"><i>S1</i><i>S2</i><i>S3</i></span><span class="c-tyre">TYRE</span><span class="c-laps" title="Timed laps">LAPS</span><span class="c-act"></span>`;
const T_BODY = (q) => `<span class="c-pos"></span><span class="c-bar"></span><span class="c-tla"></span>
<span class="c-best"></span><span class="c-gap"></span>${q ? '<span class="c-q"><i></i><i></i><i></i></span>' : ''}
<span class="c-sec"><i></i><i></i><i></i></span><span class="c-tyre"><b></b><em></em><small></small></span><span class="c-laps"></span><span class="c-act"></span>`;

export function mount(el, store) {
  el.classList.add('tower');
  el.innerHTML = `<div class="row head"></div><div class="tw-scroll"><div class="tw-body"></div></div><div class="tw-empty">Waiting for timing data...</div>`;
  const head = el.querySelector('.row.head'), body = el.querySelector('.tw-body'), scroller = el.querySelector('.tw-scroll'), empty = el.querySelector('.tw-empty');
  const lineCut = document.createElement('div'); lineCut.className = 'cut-line'; lineCut.hidden = true; lineCut.innerHTML = '<span>ELIMINATION ZONE</span>';
  const lineKo = document.createElement('div'); lineKo.className = 'cut-line ko'; lineKo.hidden = true; lineKo.innerHTML = '<span>KNOCKED OUT</span>';
  body.append(lineCut, lineKo);
  const rows = new Map(); // num -> ref
  let mode = ''; // 'race' | 'practice' | 'quali'

  function setMode(m) {
    if (m === mode) return;
    mode = m;
    for (const ref of rows.values()) ref.e.remove();
    rows.clear();
    el.dataset.mode = m; el.classList.toggle('tm-timed', m !== 'race'); el.classList.toggle('tm-quali', m === 'quali');
    head.innerHTML = m === 'race' ? RACE_HEAD : T_HEAD(m === 'quali');
  }
  setMode('race');

  function makeRow(num) {
    const e = document.createElement('div'); e.className = 'row'; e.dataset.num = num;
    e.innerHTML = mode === 'race' ? RACE_BODY : T_BODY(mode === 'quali');
    const q = (s) => e.querySelector(s);
    const ref = {
      e, num, idx: -1, pos: q('.c-pos'), chg: q('.c-chg'), bar: q('.c-bar'), tla: q('.c-tla'), gap: q('.c-gap'),
      int: q('.c-int'), last: q('.c-last'), best: q('.c-best'), sec: [...e.querySelectorAll('.c-sec i')], q: [...e.querySelectorAll('.c-q i')],
      tyreB: q('.c-tyre b'), tyreE: q('.c-tyre em'), tyreS: q('.c-tyre small'), tyre: q('.c-tyre'), pit: q('.c-pit'), drs: q('.c-drs i'), spd: q('.c-spd'),
      laps: q('.c-laps'), act: q('.c-act'),
    };
    body.append(e); rows.set(num, ref); return ref;
  }

  function updateFast(ref, car) {
    if (!ref.spd) return;
    const f = store.fast[ref.num];
    const drs = f && f.drs != null ? f.drs : car.tel && car.tel.drs;
    if (ref.drs) setClass(ref.drs, drs ? 'on' : '');
    const sp = f && f.speed != null ? f.speed : car.tel && car.tel.speed;
    setText(ref.spd, car.retired || car.inPit ? '' : isNum(sp) ? Math.round(sp) : '');
  }

  function tyreCell(ref, car) {
    const ty = car.tyre || {}, cp = compound(ty.compound);
    setText(ref.tyreB, cp.l); setStyle(ref.tyreB, 'borderColor', cp.c); setStyle(ref.tyreB, 'color', cp.c);
    setText(ref.tyreE, ty.age != null ? ty.age : '');
    setText(ref.tyreS, ty.stint && mode === 'race' ? 'S' + ty.stint : '');
    ref.tyre.title = (ty.compound || '') + (ty.new === false ? ' (used)' : ty.new ? ' (new)' : '');
  }

  function updateRace(ref, car, idx) {
    setText(ref.pos, car.retired ? '' : car.pos != null ? car.pos : idx + 1);
    const pc = car.posChange | 0;
    setText(ref.chg, pc > 0 ? '▲' + pc : pc < 0 ? '▼' + -pc : '');
    setClass(ref.chg, 'c-chg ' + (pc > 0 ? 'up' : pc < 0 ? 'down' : ''));
    const lead = car.pos === 1;
    setText(ref.gap, car.retired ? 'OUT' : lead ? 'LEADER' : car.gapLeader || '');
    setText(ref.int, car.retired || lead ? '' : car.interval || '');
    const l = car.last || {}, b = car.best || {};
    setText(ref.last, l.text || ''); setClass(ref.last, 'c-last ' + timeClass(l));
    setText(ref.best, b.text || ''); setClass(ref.best, 'c-best ' + (b.text ? (b.overall ? 't-purple' : 't-none') : 't-none'));
    for (let i = 0; i < 3; i++) {
      const s = (car.sectors && car.sectors[i]) || {};
      setText(ref.sec[i], s.text || ''); setClass(ref.sec[i], timeClass(s));
    }
    tyreCell(ref, car);
    if (car.retired) { setText(ref.pit, 'RET'); setClass(ref.pit, 'c-pit badge ret'); }
    else if (car.inPit) { setText(ref.pit, 'IN PIT'); setClass(ref.pit, 'c-pit badge pit'); }
    else if (car.pitOut) { setText(ref.pit, 'OUT'); setClass(ref.pit, 'c-pit badge out'); }
    else { setText(ref.pit, car.pits ? car.pits + 'P' : ''); setClass(ref.pit, 'c-pit'); }
    updateFast(ref, car);
    setClass(ref.e, 'row' + (car.retired ? ' retired' : '') + (car.inPit ? ' inpit' : ''));
  }

  // best lap to show: the feed's BestLapTime, else the most recent quali-part best (feeds reset BestLapTime per part)
  function dispBest(car) {
    if (car.best && car.best.text) return { text: car.best.text, sec: car.best.sec, overall: car.best.overall };
    for (let i = 2; i >= 0; i--) { const b = car.bestLaps && car.bestLaps[i]; if (b && b.text) return { text: b.text, sec: b.sec, overall: false }; }
    return null;
  }
  function updateTimed(ref, car, idx, B, ob, lead) {
    const db = dispBest(car), hasBest = !!db;
    setText(ref.pos, car.pos != null && hasBest ? car.pos : hasBest ? idx + 1 : '–');
    setText(ref.best, hasBest ? db.text : '—');
    setClass(ref.best, 'c-best ' + (hasBest ? (idx === 0 || db.overall ? 't-purple' : '') : 't-none'));
    const first = idx === 0 && hasBest;
    const calc = hasBest && isNum(db.sec) && isNum(lead) && db.sec > lead ? '+' + (db.sec - lead).toFixed(3) : '';
    setText(ref.gap, !hasBest ? '' : first ? 'FASTEST' : car.gapFastest || car.gapLeader || calc);
    setClass(ref.gap, 'c-gap' + (first ? ' lead' : ''));
    if (mode === 'quali') {
      for (let i = 0; i < 3; i++) {
        const b = car.bestLaps && car.bestLaps[i];
        setText(ref.q[i], b && b.text ? b.text : '');
        setClass(ref.q[i], b && b.text ? (i === (car._part || 0) - 1 ? 'cur' : '') : 'na');
      }
    }
    const pb = B.pb[car.num] || { s: [null, null, null] };
    for (let i = 0; i < 3; i++) {
      let v = pb.s[i], cls = 't-none';
      if (v != null) cls = ob.s[i] && v <= ob.s[i].v + 5e-4 ? 't-purple' : 't-green';
      else { const s = car.sectors && car.sectors[i]; if (s && s.text) { v = s.sec; cls = timeClass(s); } }
      setText(ref.sec[i], v != null ? (+v).toFixed(3) : ''); setClass(ref.sec[i], cls);
    }
    tyreCell(ref, car);
    const nl = car.timedLaps != null ? car.timedLaps : car.lap;
    setText(ref.laps, nl != null && nl > 0 ? nl : '');
    const a = activityOf(car);
    setText(ref.act, a.t); setClass(ref.act, 'c-act act-' + a.k);
    setClass(ref.e, 'row' + (car.retired ? ' retired' : '') + (car.knockedOut ? ' ko' : '') + (car.cutoff && !car.knockedOut ? ' danger' : '') +
      (a.k === 'garage' ? ' garage' : '') + (a.k === 'pit' ? ' inpit' : '') + (!hasBest ? ' notime' : ''));
  }

  function updateRow(ref, car, idx, ctx) {
    if (ref.idx !== idx) { ref.idx = idx; ref.e.style.transform = `translateY(${idx * ROW_H}px)`; }
    setStyle(ref.bar, 'background', car.color || '#888');
    setText(ref.tla, car.tla || car.num);
    if (mode === 'race') updateRace(ref, car, idx); else updateTimed(ref, car, idx, ctx.B, ctx.B.ob, ctx.lead);
    ref.car = car;
  }

  function applySel() {
    const prim = [...store.selected][0];
    for (const [num, ref] of rows) {
      const s = store.selected.has(num);
      ref.e.classList.toggle('sel', s);
      ref.e.classList.toggle('primary', num === prim);
      if (s) ref.e.style.setProperty('--tc', (ref.car && ref.car.color) || '#e10600');
    }
  }

  function onState() {
    const s = store.state;
    if (!s || !s.cars) { onReset(); return; }
    const X = sessionOf(s);
    setMode(X.isQuali ? 'quali' : X.timed ? 'practice' : 'race');
    const order = s.order && s.order.length ? s.order : Object.keys(s.cars);
    empty.hidden = order.length > 0;
    empty.textContent = store.phase === 'waiting' ? (store.statusMsg || 'Waiting for the session to start...') : 'Waiting for timing data...';
    const ctx = { B: mode === 'race' ? null : bestsOf(store.history), lead: null };
    if (mode !== 'race') for (const num of order) { const c = s.cars[num]; const d = c && dispBest(c); if (d && isNum(d.sec)) { ctx.lead = d.sec; break; } }
    const seen = new Set();
    let firstCut = -1, firstKo = -1;
    order.forEach((num, i) => {
      const car = s.cars[num]; if (!car) return;
      seen.add(num);
      car._part = X.part;
      if (mode === 'quali') {
        if (car.knockedOut && firstKo < 0) firstKo = i;
        if (car.cutoff && !car.knockedOut && firstCut < 0) firstCut = i;
      }
      updateRow(rows.get(num) || makeRow(num), car, i, ctx);
    });
    for (const [num, ref] of rows) if (!seen.has(num)) { ref.e.remove(); rows.delete(num); }
    lineCut.hidden = firstCut < 0; lineKo.hidden = firstKo < 0;
    if (firstCut >= 0) lineCut.style.transform = `translateY(${firstCut * ROW_H - 1}px)`;
    if (firstKo >= 0) { lineKo.style.transform = `translateY(${firstKo * ROW_H - 1}px)`; lineKo.firstChild.textContent = 'KNOCKED OUT' + (X.partLabel ? ' · ' + X.partLabel : ''); }
    body.style.height = order.length * ROW_H + 'px';
    applySel();
  }
  function onReset() {
    for (const ref of rows.values()) ref.e.remove();
    rows.clear(); body.style.height = '0'; empty.hidden = false; lineCut.hidden = lineKo.hidden = true;
    empty.textContent = store.phase === 'waiting' ? (store.statusMsg || 'Waiting for the session to start...') : 'Waiting for timing data...';
  }
  function onFast() {
    if (!store.state || mode !== 'race') return;
    for (const ref of rows.values()) if (ref.car) updateFast(ref, ref.car);
  }

  body.addEventListener('click', (ev) => {
    const r = ev.target.closest('.row'); if (!r) return;
    store.select(r.dataset.num, { additive: ev.shiftKey || ev.ctrlKey || ev.metaKey });
  });
  let fastT = 0;
  store.subscribe((k) => {
    if (k === 'state' || k === 'history') onState();
    else if (k === 'reset') onReset();
    else if (k === 'conn') { if (!store.state) onReset(); }
    else if (k === 'fast') { const n = performance.now(); if (n - fastT > 240) { fastT = n; onFast(); } }
    else if (k === 'selection') {
      applySel();
      const p = rows.get([...store.selected][0]);
      if (p) {
        const top = p.idx * ROW_H, vt = scroller.scrollTop, vh = scroller.clientHeight;
        if (top < vt) scroller.scrollTop = top; else if (top + ROW_H > vt + vh) scroller.scrollTop = top + ROW_H - vh;
      }
    }
  });
  if (store.state) onState();
}
