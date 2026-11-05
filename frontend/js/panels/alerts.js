import { h, ls } from '../util.js';
import { sessionOf } from '../session.js';

// Toasts + optional beeps derived from successive state diffs.
export function mount(el, store) {
  const cfg = Object.assign({ toasts: true, sound: false }, ls.get('f1.alerts', {}));
  const save = () => ls.set('f1.alerts', cfg);
  const bT = h('button', { title: 'Toggle toast alerts' }, 'Toasts');
  const bS = h('button', { title: 'Toggle alert sounds' }, 'Sound');
  const sync = () => { bT.classList.toggle('on', cfg.toasts); bS.classList.toggle('on', cfg.sound); };
  bT.onclick = () => { cfg.toasts = !cfg.toasts; save(); sync(); };
  bS.onclick = () => { cfg.sound = !cfg.sound; save(); sync(); if (cfg.sound) beep(660, 0.08); };
  sync(); el.append(bT, bS);
  const box = h('div', { class: 'toasts' }); document.body.append(box);

  let ac = null;
  function beep(freq = 880, dur = 0.15, n = 1) {
    if (!cfg.sound) return;
    try {
      ac = ac || new (window.AudioContext || window.webkitAudioContext)();
      if (ac.state === 'suspended') ac.resume();
      for (let i = 0; i < n; i++) {
        const o = ac.createOscillator(), g = ac.createGain(), t = ac.currentTime + i * (dur + 0.06);
        o.type = 'square'; o.frequency.value = freq; g.gain.setValueAtTime(0.0001, t);
        g.gain.exponentialRampToValueAtTime(0.12, t + 0.01); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
        o.connect(g); g.connect(ac.destination); o.start(t); o.stop(t + dur + 0.02);
      }
    } catch {}
  }
  function toast(text, cls = '', tone) {
    if (tone) beep(...tone);
    if (!cfg.toasts) return;
    const t = h('div', { class: 'toast ' + cls }, text);
    box.prepend(t);
    while (box.children.length > 5) box.lastChild.remove();
    setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 400); }, 6000);
    t.onclick = () => t.remove();
  }

  let prev = null, budget = 0;
  const T = toast; // per-state-diff cap so a big catch-up diff never floods the screen
  const say = (...a) => { if (budget-- > 0) T(...a); };
  function onState() {
    const s = store.state; if (!s || !s.cars) return;
    const cars = s.cars, X = sessionOf(s);
    if (prev && (s.utc == null || prev.utc == null || s.utc < prev.utc - 1 || s.utc - prev.utc > 120)) prev = null; // seek / first state
    budget = 4;
    const cur = {
      utc: s.utc, flag: s.trackStatus && s.trackStatus.flag, fl: s.records && s.records.fastestLap, status: s.session.status,
      part: s.session.part, p1: s.order[0] && cars[s.order[0]] && cars[s.order[0]].best.text ? s.order[0] : null,
      p1text: s.order[0] && cars[s.order[0]] ? cars[s.order[0]].best.text : '', c: {},
    };
    for (const [n, c] of Object.entries(cars)) cur.c[n] = { pos: c.pos, pits: c.pits || 0, inPit: !!c.inPit, ret: !!c.retired, ko: !!c.knockedOut };
    if (prev) {
      if (cur.flag && cur.flag !== prev.flag) {
        const m = {
          red: ['RED FLAG', 'f-red', [440, 0.3, 3]], sc: ['SAFETY CAR deployed', 'f-yellow', [600, 0.2, 2]],
          vsc: ['VIRTUAL SAFETY CAR', 'f-yellow', [600, 0.2, 2]], yellow: ['Yellow flag', 'f-yellow', [760, 0.12, 1]],
          green: ['Track clear - green flag', 'f-green', [980, 0.1, 1]],
        }[cur.flag];
        if (m) say(m[0], m[1], m[2]);
      }
      if (X.timed) {
        if (/abort/i.test(cur.status || '') && !/abort/i.test(prev.status || '')) say('Session suspended (red flag)', 'f-red', [440, 0.3, 2]);
        else if (/abort/i.test(prev.status || '') && /start/i.test(cur.status || '')) say('Session resumed', 'f-green', [980, 0.1, 1]);
        if (X.isQuali && cur.part && prev.part && cur.part !== prev.part) say('Q' + cur.part + ' started', 'f-green', [980, 0.1, 2]);
        // new P1 instead of overtakes
        if (cur.p1 && prev.p1 && cur.p1 !== prev.p1 && cars[cur.p1]) say(`${cars[cur.p1].tla} takes P1 ${cur.p1text}`, 'f-purple', [1400, 0.1, 2]);
        // eliminations
        const ko = Object.keys(cur.c).filter((n) => cur.c[n].ko && prev.c[n] && !prev.c[n].ko);
        if (ko.length) say('Knocked out: ' + ko.slice(0, 6).map((n) => cars[n].tla).join(', ') + (ko.length > 6 ? ' +' + (ko.length - 6) : ''), 'f-red', [300, 0.2, 1]);
      }
      for (const [n, c] of Object.entries(cur.c)) {
        const p = prev.c[n], car = cars[n]; if (!p) continue;
        const tla = car.tla || n;
        const interesting = !X.timed || store.selected.has(n);
        if (interesting && ((c.inPit && !p.inPit) || c.pits > p.pits && !c.inPit && !p.inPit)) say(`${tla} pits`, 'f-pit');
        if (c.ret && !p.ret) say(`${tla} retires`, 'f-red', [300, 0.25, 1]);
        if (!X.timed && store.selected.has(n) && p.pos && c.pos && c.pos !== p.pos && !c.ret) {
          const gained = c.pos < p.pos;
          const other = gained ? Object.entries(cur.c).find(([k, v]) => k !== n && v.pos === c.pos + 1) : Object.entries(cur.c).find(([k, v]) => k !== n && v.pos === c.pos - 1);
          const on = other && cars[other[0]] ? ' ' + (gained ? 'passing ' : 'passed by ') + cars[other[0]].tla : '';
          say(`${tla} P${p.pos} → P${c.pos}${on}`, gained ? 'f-green' : 'f-yellow', [1200, 0.07, 1]);
        }
      }
      const a = prev.fl, b = cur.fl;
      if (!X.timed && b && (!a || a.num !== b.num || a.sec !== b.sec)) {
        const car = cars[b.num];
        say(`Fastest lap ${car ? car.tla : b.num} ${b.text || ''}`, 'f-purple', [1400, 0.1, 2]);
      }
    }
    prev = cur;
  }
  store.subscribe((k) => { if (k === 'state') onState(); else if (k === 'reset') prev = null; });
}
