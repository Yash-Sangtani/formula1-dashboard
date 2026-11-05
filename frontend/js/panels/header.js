import { setText, setClass, fmtClock, replay, SPEEDS } from '../util.js';

const FLAGS = {
  green: 'TRACK CLEAR', yellow: 'YELLOW FLAG', sc: 'SAFETY CAR', vsc: 'VIRTUAL SAFETY CAR', red: 'RED FLAG',
};
const MODE = { live: 'LIVE', file: 'REPLAY', sim: 'SIM' };

export function mount(el, store) {
  el.classList.add('hdr-panel');
  el.innerHTML = `
    <div class="hd-brand"><span class="logo">F1</span></div>
    <div class="hd-title"><div class="hd-meeting">Waiting for data</div><div class="hd-session"></div></div>
    <div class="hd-box hd-lap"><label>LAP</label><b><span class="cur">-</span><span class="tot"></span></b></div>
    <div class="hd-box hd-clock"><label class="cl">CLOCK</label><b>--:--:--</b></div>
    <div class="hd-flag flag-green">TRACK CLEAR</div>
    <div class="hd-wx">
      <span class="wx" title="Air temperature"><label>AIR</label><b class="air">-</b></span>
      <span class="wx" title="Track temperature"><label>TRK</label><b class="trk">-</b></span>
      <span class="wx" title="Humidity"><label>HUM</label><b class="hum">-</b></span>
      <span class="wx" title="Wind"><label>WIND</label><b><i class="arrow">&#10148;</i> <span class="wnd">-</span></b></span>
      <span class="rain" hidden>RAIN</span>
    </div>
    <div class="hd-ctl" hidden>
      <button class="pp" title="Play / pause (space)">&#10074;&#10074;</button>
      <span class="spd"></span>
    </div>
    <div class="hd-conn"><span class="dot"></span><span class="mode">-</span><span class="age"></span></div>`;
  const q = (s) => el.querySelector(s);
  const R = {
    meeting: q('.hd-meeting'), session: q('.hd-session'), cur: q('.hd-lap .cur'), tot: q('.hd-lap .tot'),
    clock: q('.hd-clock b'), clabel: q('.hd-clock .cl'), flag: q('.hd-flag'),
    air: q('.air'), trk: q('.trk'), hum: q('.hum'), wnd: q('.wnd'), arrow: q('.arrow'), rain: q('.rain'),
    ctl: q('.hd-ctl'), pp: q('.pp'), spd: q('.spd'),
    conn: q('.hd-conn'), mode: q('.mode'), age: q('.age'),
  };
  for (const s of SPEEDS) {
    const b = document.createElement('button'); b.textContent = s + 'x'; b.dataset.s = s;
    b.onclick = () => replay.setSpeed(s); R.spd.append(b);
  }
  R.pp.onclick = () => replay.toggle();

  let lastMsg = 0, clk = null; // clk: {base, dir, at}

  function onState() {
    const s = store.state; if (!s) return;
    lastMsg = Date.now();
    const se = s.session || {};
    setText(R.meeting, se.meeting || 'Formula 1');
    setText(R.session, [se.name || se.type, se.status && se.status !== 'Started' ? se.status : ''].filter(Boolean).join(' · '));
    setText(R.cur, s.lap && s.lap.current != null ? s.lap.current : '-');
    setText(R.tot, s.lap && s.lap.total ? ' / ' + s.lap.total : '');
    const c = s.clock || {};
    if (c.remainingSec != null) clk = { base: c.remainingSec, dir: -1, at: Date.now(), run: c.extrapolating !== false, label: 'REMAINING' };
    else if (c.elapsedSec != null) clk = { base: c.elapsedSec, dir: 1, at: Date.now(), run: c.extrapolating !== false, label: 'ELAPSED' };
    else clk = null;
    tickClock();
    // flag
    const ts = s.trackStatus || {}; let f = ts.flag || 'green';
    let txt = FLAGS[f] || (ts.label || f).toUpperCase();
    if (f === 'green' && /finish|ends/i.test(se.status || '')) { f = 'finish'; txt = 'SESSION FINISHED'; }
    setClass(R.flag, 'hd-flag flag-' + f); setText(R.flag, txt);
    document.body.dataset.flag = f;
    // weather
    const w = s.weather || {};
    const n = (v, u, d = 1) => (v == null ? '-' : (+v).toFixed(d) + u);
    setText(R.air, n(w.air, '°')); setText(R.trk, n(w.track, '°')); setText(R.hum, n(w.humidity, '%', 0));
    setText(R.wnd, n(w.wind, ' m/s'));
    if (w.windDir != null) R.arrow.style.transform = `rotate(${(+w.windDir + 90) % 360}deg)`; // arrow points downwind
    R.rain.hidden = !w.rain;
  }
  function tickClock() {
    if (!clk) { setText(R.clock, '--:--:--'); return; }
    let v = clk.base;
    if (clk.run && !replay.paused) {
      const sp = replay.available() ? replay.speed : 1;
      v += clk.dir * ((Date.now() - clk.at) / 1000) * sp;
    }
    setText(R.clock, fmtClock(v)); setText(R.clabel, clk.label);
  }
  function onConn() {
    const h = store.hello;
    const mode = h ? (MODE[h.mode] || String(h.mode).toUpperCase()) : '-';
    setText(R.mode, mode);
    const st = store.conn === 'open' ? 'open' : store.conn === 'closed' ? 'closed' : 'connecting';
    setClass(R.conn, 'hd-conn ' + st + ' m-' + (h ? h.mode : 'none'));
    R.conn.title = store.conn === 'closed' ? 'Disconnected - reconnecting' : store.conn;
    if (store.conn === 'closed') setText(R.mode, 'RECONNECTING');
    R.ctl.hidden = !replay.available();
  }
  function onReplay() {
    R.pp.innerHTML = replay.paused ? '&#9654;' : '&#10074;&#10074;';
    for (const b of R.spd.children) b.classList.toggle('on', +b.dataset.s === replay.speed);
  }
  let hello = null;
  store.subscribe((k) => {
    if (k === 'state') onState();
    else if (k === 'fast') lastMsg = Date.now();
    else if (k === 'conn') { if (store.hello !== hello) { hello = store.hello; replay.sync(hello); } onConn(); }
  });
  replay.on(onReplay); onReplay(); onConn(); if (store.state) onState();
  setInterval(() => {
    tickClock();
    if (store.conn !== 'open' || !lastMsg) { setText(R.age, ''); return; }
    const a = (Date.now() - lastMsg) / 1000;
    setText(R.age, a < 1 ? '' : a.toFixed(0) + 's');
    R.age.className = 'age' + (a > 10 ? ' bad' : a > 3 ? ' warn' : '');
  }, 250);
}
