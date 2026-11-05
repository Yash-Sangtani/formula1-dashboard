import { setText, setClass, fmtClock, replay, SPEEDS } from '../util.js';
import { sessionOf, isWaiting, KIND_LABEL } from '../session.js';

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
    <div class="hd-box hd-part" hidden><label>PART</label><b class="pl">-</b></div>
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
    conn: q('.hd-conn'), mode: q('.mode'), age: q('.age'), lapBox: q('.hd-lap'), partBox: q('.hd-part'), pl: q('.hd-part .pl'),
  };
  for (const s of SPEEDS) {
    const b = document.createElement('button'); b.textContent = s + 'x'; b.dataset.s = s;
    b.onclick = () => replay.setSpeed(s); R.spd.append(b);
  }
  R.pp.onclick = () => replay.toggle();

  let clk = null; // clk: {base, dir, at, run, label}
  const FINISHED = /finish|ends|final/i;

  function onState() {
    const s = store.state;
    if (!s) { onNoState(); return; }
    const se = s.session || {}, X = sessionOf(s);
    const waiting = isWaiting(store) && !Object.keys(s.cars || {}).length;
    setText(R.meeting, se.meeting || 'Formula 1');
    const st = se.status && se.status !== 'Started' ? se.status : '';
    const label = X.kind === 'unknown' ? (se.name || se.type) : (se.name || KIND_LABEL[X.kind]);
    setText(R.session, [label, X.timed && X.partLabel ? X.partLabel : '', st === 'Aborted' ? 'SUSPENDED' : st].filter(Boolean).join(' · '));
    // lap counter only in lap-based sessions
    R.lapBox.hidden = X.timed;
    setText(R.cur, s.lap && s.lap.current ? s.lap.current : '-');
    setText(R.tot, s.lap && s.lap.total ? ' / ' + s.lap.total : '');
    R.partBox.hidden = !(X.isQuali && X.partLabel);
    setText(R.pl, X.partLabel || '');
    const c = s.clock || {};
    const stopped = /abort|finish|final|ends|inactive/i.test(se.status || '');
    const run = c.extrapolating !== false && !stopped;
    const part = X.isQuali && X.partLabel ? X.partLabel + ' ' : '';
    if (c.remainingSec != null) clk = { base: c.remainingSec, dir: -1, at: performance.now(), run, label: (part + (X.timed ? 'TIME LEFT' : 'REMAINING')).trim() };
    else if (c.elapsedSec != null) clk = { base: c.elapsedSec, dir: 1, at: performance.now(), run, label: 'ELAPSED' };
    else clk = null;
    tickClock();
    // flag
    const ts = s.trackStatus || {}; let f = ts.flag || 'green';
    let txt = FLAGS[f] || (ts.label || f).toUpperCase();
    if (/abort/i.test(se.status || '') && f !== 'red') { f = 'red'; txt = 'RED FLAG - SESSION SUSPENDED'; }
    else if (f === 'green' && FINISHED.test(se.status || '')) {
      f = 'finish'; txt = X.isQuali && X.part && X.part < 3 ? (X.partLabel || 'PART') + ' ENDED' : 'SESSION FINISHED';
    } else if (f === 'green' && X.timed) txt = 'GREEN - TRACK OPEN';
    if (waiting) { f = 'wait'; txt = 'WAITING FOR SESSION'; }
    setClass(R.flag, 'hd-flag flag-' + f); setText(R.flag, txt);
    document.body.dataset.flag = f;
    document.body.dataset.kind = X.kind;
    // weather
    const w = s.weather || {};
    const n = (v, u, d = 1) => (v == null ? '-' : (+v).toFixed(d) + u);
    setText(R.air, n(w.air, '°')); setText(R.trk, n(w.track, '°')); setText(R.hum, n(w.humidity, '%', 0));
    setText(R.wnd, n(w.wind, ' m/s'));
    if (w.windDir != null) R.arrow.style.transform = `rotate(${(+w.windDir + 90) % 360}deg)`; // arrow points downwind
    R.rain.hidden = !w.rain;
  }
  function onNoState() {
    clk = null; tickClock();
    const waiting = isWaiting(store);
    setText(R.meeting, waiting ? 'No session running' : 'Waiting for data');
    setText(R.session, waiting ? 'The dashboard will start by itself when a session begins' : '');
    R.lapBox.hidden = false; R.partBox.hidden = true; setText(R.cur, '-'); setText(R.tot, '');
    setClass(R.flag, 'hd-flag flag-wait'); setText(R.flag, waiting ? 'WAITING FOR SESSION' : 'WAITING FOR DATA');
    document.body.dataset.flag = 'wait'; document.body.dataset.kind = 'unknown';
  }
  function tickClock() {
    if (!clk) { setText(R.clock, '--:--:--'); setText(R.clabel, 'CLOCK'); return; }
    let v = clk.base;
    if (clk.run && !replay.paused) {
      const sp = replay.available() ? replay.speed : 1;
      v += clk.dir * ((performance.now() - clk.at) / 1000) * sp;
    }
    if (clk.dir < 0) v = Math.max(0, v);
    setText(R.clock, fmtClock(v)); setText(R.clabel, clk.label);
  }
  function onConn() {
    const h = store.hello;
    const mode = h ? (MODE[h.mode] || String(h.mode).toUpperCase()) : '-';
    setText(R.mode, mode);
    const st = store.conn === 'open' ? 'open' : store.conn === 'closed' ? 'closed' : 'connecting';
    setClass(R.conn, 'hd-conn ' + st + ' m-' + (h ? h.mode : 'none'));
    R.conn.title = store.conn === 'closed' ? 'Disconnected - reconnecting' : store.conn;
    if (store.conn === 'closed') setText(R.mode, 'OFFLINE');
    R.ctl.hidden = !replay.available();
    if (!store.state) onNoState();
  }
  function onReplay() {
    R.pp.innerHTML = replay.paused ? '&#9654;' : '&#10074;&#10074;';
    for (const b of R.spd.children) b.classList.toggle('on', +b.dataset.s === replay.speed);
  }
  let hello = null;
  store.subscribe((k) => {
    if (k === 'state') onState();
    else if (k === 'reset') { clk = null; }
    else if (k === 'control') { replay.syncControl(store.control); }
    else if (k === 'conn') { if (store.hello !== hello) { hello = store.hello; replay.sync(hello); } onConn(); }
  });
  replay.on(() => { onReplay(); R.ctl.hidden = !replay.available(); });
  onReplay(); onConn();
  setInterval(() => {
    try {
      tickClock();
      const se = store.state && store.state.session;
      if (store.conn !== 'open' || !store.lastMsgAt) { setText(R.age, ''); R.age.className = 'age'; return; }
      if (replay.available() && replay.paused) { setText(R.age, 'paused'); R.age.className = 'age idle'; return; }
      const a = (performance.now() - store.lastMsgAt) / 1000;
      const idle = se && /final|ends/i.test(se.status || '');
      let t = a < 5 ? '' : 'No data for ' + Math.round(a) + ' s';
      if (!t && store.feedAge != null && store.feedAge > 60 && !idle) t = 'Feed quiet ' + Math.round(store.feedAge) + ' s';
      setText(R.age, t);
      R.age.className = 'age' + (!t ? '' : idle ? ' idle' : a > 30 || (store.feedAge || 0) > 120 ? ' bad' : ' warn');
      R.age.title = 'Time since the last message from the server';
    } catch (e) { /* never break the header ticker */ }
  }, 500);
}
