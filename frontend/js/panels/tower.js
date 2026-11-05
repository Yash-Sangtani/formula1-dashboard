import { setText, setClass, setStyle, timeClass, compound } from '../util.js';

const ROW_H = 32;
const HEAD = `<span class="c-pos">P</span><span class="c-chg"></span><span class="c-bar"></span><span class="c-tla">DRV</span>
<span class="c-gap">GAP</span><span class="c-int">INT</span><span class="c-last">LAST</span><span class="c-best">BEST</span>
<span class="c-sec"><i>S1</i><i>S2</i><i>S3</i></span><span class="c-tyre">TYRE</span><span class="c-pit">PIT</span><span class="c-drs">DRS</span><span class="c-spd">KPH</span>`;
const BODY = `<span class="c-pos"></span><span class="c-chg"></span><span class="c-bar"></span><span class="c-tla"></span>
<span class="c-gap"></span><span class="c-int"></span><span class="c-last"></span><span class="c-best"></span>
<span class="c-sec"><i></i><i></i><i></i></span><span class="c-tyre"><b></b><em></em><small></small></span>
<span class="c-pit"></span><span class="c-drs"><i></i></span><span class="c-spd"></span>`;

export function mount(el, store) {
  el.classList.add('tower');
  el.innerHTML = `<div class="row head">${HEAD}</div><div class="tw-scroll"><div class="tw-body"></div></div><div class="tw-empty">Waiting for timing data...</div>`;
  const body = el.querySelector('.tw-body'), scroller = el.querySelector('.tw-scroll'), empty = el.querySelector('.tw-empty');
  const rows = new Map(); // num -> ref

  function makeRow(num) {
    const e = document.createElement('div'); e.className = 'row'; e.dataset.num = num; e.innerHTML = BODY;
    const q = (s) => e.querySelector(s);
    const sec = e.querySelectorAll('.c-sec i');
    const ref = {
      e, num, idx: -1, pos: q('.c-pos'), chg: q('.c-chg'), bar: q('.c-bar'), tla: q('.c-tla'), gap: q('.c-gap'),
      int: q('.c-int'), last: q('.c-last'), best: q('.c-best'), sec: [...sec], tyreB: q('.c-tyre b'),
      tyreE: q('.c-tyre em'), tyreS: q('.c-tyre small'), tyre: q('.c-tyre'), pit: q('.c-pit'), drs: q('.c-drs i'), spd: q('.c-spd'),
    };
    body.append(e); rows.set(num, ref); return ref;
  }

  function updateFast(ref, car) {
    const f = store.fast[ref.num];
    const drs = f && f.drs != null ? f.drs : car.tel && car.tel.drs;
    setClass(ref.drs, drs ? 'on' : '');
    const sp = f && f.speed != null ? f.speed : car.tel && car.tel.speed;
    setText(ref.spd, car.retired || car.inPit ? '' : sp != null ? Math.round(sp) : '');
  }

  function updateRow(ref, car, idx) {
    if (ref.idx !== idx) { ref.idx = idx; ref.e.style.transform = `translateY(${idx * ROW_H}px)`; }
    setText(ref.pos, car.retired ? '' : car.pos != null ? car.pos : idx + 1);
    const pc = car.posChange | 0;
    setText(ref.chg, pc > 0 ? '▲' + pc : pc < 0 ? '▼' + -pc : '');
    setClass(ref.chg, 'c-chg ' + (pc > 0 ? 'up' : pc < 0 ? 'down' : ''));
    setStyle(ref.bar, 'background', car.color || '#888');
    setText(ref.tla, car.tla || car.num);
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
    const ty = car.tyre || {}, cp = compound(ty.compound);
    setText(ref.tyreB, cp.l); setStyle(ref.tyreB, 'borderColor', cp.c); setStyle(ref.tyreB, 'color', cp.c);
    setText(ref.tyreE, ty.age != null ? ty.age : '');
    setText(ref.tyreS, ty.stint ? 'S' + ty.stint : '');
    ref.tyre.title = (ty.compound || '') + (ty.new === false ? ' (used)' : ' (new)');
    if (car.retired) { setText(ref.pit, 'RET'); setClass(ref.pit, 'c-pit badge ret'); }
    else if (car.inPit) { setText(ref.pit, 'IN PIT'); setClass(ref.pit, 'c-pit badge pit'); }
    else if (car.pitOut) { setText(ref.pit, 'OUT'); setClass(ref.pit, 'c-pit badge out'); }
    else { setText(ref.pit, car.pits ? car.pits + 'P' : ''); setClass(ref.pit, 'c-pit'); }
    updateFast(ref, car);
    setClass(ref.e, 'row' + (car.retired ? ' retired' : '') + (car.inPit ? ' inpit' : ''));
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
    const s = store.state; if (!s || !s.cars) return;
    let order = s.order && s.order.length ? s.order : Object.keys(s.cars).sort((a, b) => (s.cars[a].pos || 99) - (s.cars[b].pos || 99));
    empty.hidden = order.length > 0;
    const seen = new Set();
    order.forEach((num, i) => {
      const car = s.cars[num]; if (!car) return;
      seen.add(num);
      updateRow(rows.get(num) || makeRow(num), car, i);
    });
    for (const [num, ref] of rows) if (!seen.has(num)) { ref.e.remove(); rows.delete(num); }
    body.style.height = order.length * ROW_H + 'px';
    applySel();
  }
  function onFast() {
    const s = store.state; if (!s) return;
    for (const [num, ref] of rows) if (ref.car) updateFast(ref, ref.car);
  }

  body.addEventListener('click', (ev) => {
    const r = ev.target.closest('.row'); if (!r) return;
    store.select(r.dataset.num, { additive: ev.shiftKey || ev.ctrlKey || ev.metaKey });
  });
  let fastT = 0;
  store.subscribe((k) => {
    if (k === 'state') onState();
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
