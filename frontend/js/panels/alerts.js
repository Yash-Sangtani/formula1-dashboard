import { h, ls } from '../util.js';

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

  let prev = null;
  function onState() {
    const s = store.state; if (!s || !s.cars) return;
    const cars = s.cars;
    // rebaseline on seeks / first state
    if (prev && (s.utc < prev.utc - 1 || s.utc - prev.utc > 120)) prev = null;
    const cur = {
      utc: s.utc, flag: s.trackStatus && s.trackStatus.flag, fl: s.records && s.records.fastestLap,
      c: {},
    };
    for (const [n, c] of Object.entries(cars)) cur.c[n] = { pos: c.pos, pits: c.pits || 0, inPit: !!c.inPit, ret: !!c.retired };
    if (prev) {
      if (cur.flag && cur.flag !== prev.flag) {
        const m = {
          red: ['RED FLAG', 'f-red', [440, 0.3, 3]], sc: ['SAFETY CAR deployed', 'f-yellow', [600, 0.2, 2]],
          vsc: ['VIRTUAL SAFETY CAR', 'f-yellow', [600, 0.2, 2]], yellow: ['Yellow flag', 'f-yellow', [760, 0.12, 1]],
          green: ['Track clear - green flag', 'f-green', [980, 0.1, 1]],
        }[cur.flag];
        if (m) toast(m[0], m[1], m[2]);
      }
      for (const [n, c] of Object.entries(cur.c)) {
        const p = prev.c[n], car = cars[n]; if (!p) continue;
        const tla = car.tla || n;
        if ((c.inPit && !p.inPit) || c.pits > p.pits && !c.inPit && !p.inPit) toast(`${tla} pits`, 'f-pit');
        if (c.ret && !p.ret) toast(`${tla} retires`, 'f-red', [300, 0.25, 1]);
        if (store.selected.has(n) && p.pos && c.pos && c.pos !== p.pos && !c.ret) {
          const gained = c.pos < p.pos;
          const other = gained ? Object.entries(cur.c).find(([k, v]) => k !== n && v.pos === c.pos + 1) : Object.entries(cur.c).find(([k, v]) => k !== n && v.pos === c.pos - 1);
          const on = other && cars[other[0]] ? ' ' + (gained ? 'passing ' : 'passed by ') + cars[other[0]].tla : '';
          toast(`${tla} P${p.pos} → P${c.pos}${on}`, gained ? 'f-green' : 'f-yellow', [1200, 0.07, 1]);
        }
      }
      const a = prev.fl, b = cur.fl;
      if (b && (!a || a.num !== b.num || a.sec !== b.sec)) {
        const car = cars[b.num];
        toast(`Fastest lap ${car ? car.tla : b.num} ${b.text || ''}`, 'f-purple', [1400, 0.1, 2]);
      }
    }
    prev = cur;
  }
  store.subscribe((k) => { if (k === 'state') onState(); });
}
