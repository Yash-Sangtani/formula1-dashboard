import { ls, textOn } from '../util.js';

export function mount(el, store) {
  el.classList.add('map');
  el.innerHTML = `<canvas></canvas>
    <div class="map-opts"><button data-o="labels">Labels</button><button data-o="trails">Trails</button><button data-o="follow">Follow</button></div>
    <div class="map-tip" hidden></div><div class="map-empty">Waiting for track data...</div>`;
  const cv = el.querySelector('canvas'), ctx = cv.getContext('2d');
  const tip = el.querySelector('.map-tip'), emptyEl = el.querySelector('.map-empty');
  const opts = Object.assign({ labels: true, trails: true, follow: false }, ls.get('f1.map', {}));
  const optBtns = [...el.querySelectorAll('.map-opts button')];
  const syncOpts = () => optBtns.forEach((b) => b.classList.toggle('on', !!opts[b.dataset.o]));
  optBtns.forEach((b) => b.onclick = () => { opts[b.dataset.o] = !opts[b.dataset.o]; ls.set('f1.map', opts); syncOpts(); });
  syncOpts();

  let W = 0, H = 0, dpr = 1;
  let geo = null; // rotated, start-ordered, closed polyline
  const anim = {}; // per car interpolation state
  const trails = {}; // num -> [{x,y}]
  let hits = [];
  let mouse = null;
  let yellow = []; let rcRef = null; let rcSig = '';
  let vcx = 0, vcy = 0, zoomF = 1; // follow view state

  // ---------- geometry ----------
  function rot(x, y) {
    const a = geo ? geo.a : 0, c = Math.cos(a), s = Math.sin(a);
    return [x * c - y * s, x * s + y * c];
  }
  function buildGeo() {
    const t = store.track; if (!t || !t.points || t.points.length < 3) { geo = null; return; }
    const a = ((t.rotation || 0) * Math.PI) / 180;
    geo = { a };
    let raw = t.points.slice(); const si = t.startIdx || 0;
    raw = raw.slice(si).concat(raw.slice(0, si));
    const f = raw[0], l = raw[raw.length - 1];
    if (Math.hypot(f[0] - l[0], f[1] - l[1]) > 1e-6) raw.push([f[0], f[1]]);
    const pts = raw.map(([x, y]) => rot(x, y));
    const cum = [0];
    for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    let x0 = 1e18, y0 = 1e18, x1 = -1e18, y1 = -1e18;
    for (const [x, y] of pts) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    Object.assign(geo, { pts, cum, total: cum[cum.length - 1] || 1, x0, x1, y0, y1, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2 });
    vcx = geo.cx; vcy = geo.cy;
    emptyEl.hidden = true;
  }
  function upper(f) { // first index with cum/total > f
    const T = geo.total * f; let lo = 0, hi = geo.cum.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (geo.cum[m] > T) hi = m; else lo = m + 1; }
    return lo;
  }
  function pointAt(f) {
    f = ((f % 1) + 1) % 1; const i = Math.max(1, upper(f));
    const c0 = geo.cum[i - 1], c1 = geo.cum[i], u = c1 > c0 ? (f * geo.total - c0) / (c1 - c0) : 0;
    const p = geo.pts[i - 1], q = geo.pts[i];
    return [p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u];
  }

  // ---------- view transform ----------
  let sc = 1;
  const SX = (x) => (x - vcx) * sc + W / 2, SY = (y) => H / 2 - (y - vcy) * sc;

  function fit() {
    const pad = 30, bw = Math.max(1, geo.x1 - geo.x0), bh = Math.max(1, geo.y1 - geo.y0);
    return Math.min((W - 2 * pad) / bw, (H - 2 * pad) / bh);
  }

  function pathRange(a, b) {
    if (b < a) { pathRange(a, 1); pathRange(0, b); return; }
    a = Math.max(0, a); b = Math.min(1, b);
    const ia = upper(a), ib = upper(b);
    const p0 = pointAt(a); ctx.moveTo(SX(p0[0]), SY(p0[1]));
    for (let i = ia; i < ib; i++) ctx.lineTo(SX(geo.pts[i][0]), SY(geo.pts[i][1]));
    const p1 = pointAt(b >= 1 ? 0.999999 : b); ctx.lineTo(SX(p1[0]), SY(p1[1]));
  }
  function strokeRange(a, b, style, w, alpha = 1) {
    ctx.beginPath(); pathRange(a, b);
    ctx.strokeStyle = style; ctx.lineWidth = w; ctx.globalAlpha = alpha; ctx.stroke(); ctx.globalAlpha = 1;
  }

  // ---------- yellow-flag sectors from race control ----------
  function computeYellow() {
    const rc = store.state && store.state.raceControl;
    if (rc === rcRef) return; rcRef = rc;
    const sig = rc ? rc.length + ':' + (rc[0] && (rc[0].utc + rc[0].message)) : '';
    if (sig === rcSig) return; rcSig = sig;
    yellow = [];
    if (!rc) return;
    let maxS = 0;
    for (const m of rc) if (m.sector) maxS = Math.max(maxS, +m.sector);
    const N = (store.track && store.track.marshalSectors) || (maxS <= 3 ? 3 : Math.max(maxS, 18));
    const state = new Map();
    for (let i = rc.length - 1; i >= 0; i--) { // oldest -> newest
      const m = rc[i]; if (m.category !== 'Flag' || !m.sector) continue;
      const f = String(m.flag || '').toUpperCase();
      if (/YELLOW/.test(f)) state.set(+m.sector, true);
      else if (/CLEAR|GREEN/.test(f)) state.set(+m.sector, false);
    }
    for (const [s, on] of state) if (on) yellow.push([(s - 1) / N, s / N]);
  }

  // ---------- car ingestion ----------
  function ingest(num, d, now) {
    if (!d || (d.x == null && d.progress == null)) return;
    const a = anim[num] || (anim[num] = { t: now, v: 0, vx: 0, vy: 0, rp: null, rx: null });
    if (d.x === a.lx && d.y === a.ly && d.progress === a.lp) return;
    const dt = now - a.t;
    if (d.progress != null) {
      if (a.p != null && dt > 40 && dt < 2500) {
        let dp = d.progress - a.p; dp -= Math.round(dp);
        a.v = Math.max(0, a.v * 0.4 + (dp / dt) * 0.6);
      }
      a.p = d.progress; if (a.rp == null) a.rp = d.progress;
    }
    if (d.x != null) {
      if (a.x != null && dt > 40 && dt < 2500) { a.vx = a.vx * 0.4 + ((d.x - a.x) / dt) * 0.6; a.vy = a.vy * 0.4 + ((d.y - a.y) / dt) * 0.6; }
      a.x = d.x; a.y = d.y; if (a.rx == null) { a.rx = d.x; a.ry = d.y; }
    }
    a.t = now; a.lx = d.x; a.ly = d.y; a.lp = d.progress;
  }
  function ingestAll(useState) {
    const now = performance.now();
    for (const [num, f] of Object.entries(store.fast)) ingest(num, f, now);
    if (useState && store.state) {
      for (const [num, c] of Object.entries(store.state.cars)) {
        if (store.fast[num]) continue;
        ingest(num, { x: c.xy && c.xy.x, y: c.xy && c.xy.y, progress: c.progress }, now);
      }
    }
  }

  // ---------- frame ----------
  let last = performance.now(), trailT = 0;
  function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min(100, now - last); last = now;
    if (!W || !H) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    if (!geo) return;
    const alpha = 1 - Math.exp(-dt / 90);
    const cars = (store.state && store.state.cars) || {};
    const flag = store.state && store.state.trackStatus && store.state.trackStatus.flag;

    // advance cars (rotated world coords)
    const cur = {};
    for (const [num, a] of Object.entries(anim)) {
      const c = cars[num]; if (c && c.retired && c.xy == null && !store.fast[num]) continue;
      const ext = Math.min(now - a.t, 500);
      if (a.p != null && a.rp != null) {
        const tgt = a.p + a.v * ext; let d = tgt - a.rp; d -= Math.round(d);
        if (Math.abs(d) > 0.2) a.rp = tgt; else a.rp += d * alpha;
        a.rp = ((a.rp % 1) + 1) % 1;
        cur[num] = pointAt(a.rp);
      } else if (a.rx != null) {
        a.rx += (a.x + a.vx * ext - a.rx) * alpha; a.ry += (a.y + a.vy * ext - a.ry) * alpha;
        cur[num] = rot(a.rx, a.ry);
      }
    }

    // view
    const base = fit();
    const prim = [...store.selected][0];
    let follow = opts.follow && prim && cur[prim];
    zoomF += ((follow ? 3 : 1) - zoomF) * Math.min(1, alpha * 1.5);
    sc = base * zoomF;
    const tx = follow ? cur[prim][0] : geo.cx, ty = follow ? cur[prim][1] : geo.cy;
    vcx += (tx - vcx) * (follow ? Math.min(1, alpha * 2) : Math.min(1, alpha * 1.5));
    vcy += (ty - vcy) * (follow ? Math.min(1, alpha * 2) : Math.min(1, alpha * 1.5));

    // track
    computeYellow();
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    strokeRange(0, 1, '#0d0e12', 15);
    const asphalt = flag === 'red' ? '#5a1d1d' : flag === 'sc' || flag === 'vsc' ? '#5a4a1a' : '#343842';
    strokeRange(0, 1, asphalt, 10);
    const pulse = 0.55 + 0.45 * Math.sin(now / 250);
    if ((flag === 'sc' || flag === 'vsc') ) strokeRange(0, 1, '#ffb000', 10, 0.25 * pulse);
    if (flag === 'red') strokeRange(0, 1, '#ff0000', 10, 0.4 * pulse);
    if (flag === 'yellow' && !yellow.length) strokeRange(0, 1, '#ffd400', 10, 0.35);
    for (const [a, b] of yellow) strokeRange(a, b, '#ffd400', 10, 0.55 + 0.35 * pulse);
    for (const z of store.track.drsZones || []) strokeRange(z[0], z[1], '#00d96a', 4, 0.75);
    const sec = store.track.sectors || [1 / 3, 2 / 3];
    const SC = ['#e10600', '#3a86ff', '#ffd400'];
    strokeRange(0, sec[0], SC[0], 2, 0.75); strokeRange(sec[0], sec[1], SC[1], 2, 0.75); strokeRange(sec[1], 1, SC[2], 2, 0.75);
    // sector ticks
    for (const f of [sec[0], sec[1]]) {
      const p = pointAt(f), q = pointAt(f + 0.002);
      tick(p, q, 9, '#9aa0ab', 2);
    }
    // start / finish
    tick(geo.pts[0], pointAt(0.002), 11, '#fff', 3);

    // trails
    if (now - trailT > 60) {
      trailT = now;
      for (const num of store.selected) if (cur[num]) {
        const t = trails[num] || (trails[num] = []);
        const lp = t[t.length - 1]; if (lp && Math.hypot(lp[0] - cur[num][0], lp[1] - cur[num][1]) > geo.total * 0.06) t.length = 0;
        t.push([cur[num][0], cur[num][1]]); if (t.length > 90) t.shift();
      }
    }
    if (opts.trails) for (const num of store.selected) {
      const t = trails[num]; if (!t || t.length < 2) continue;
      const col = (cars[num] && cars[num].color) || '#fff';
      ctx.strokeStyle = col; ctx.lineWidth = 3;
      for (let i = 1; i < t.length; i++) {
        ctx.globalAlpha = (i / t.length) * 0.7;
        ctx.beginPath(); ctx.moveTo(SX(t[i - 1][0]), SY(t[i - 1][1])); ctx.lineTo(SX(t[i][0]), SY(t[i][1])); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // cars
    hits = [];
    const nums = Object.keys(cur).sort((a, b) => (store.selected.has(a) ? 1 : 0) - (store.selected.has(b) ? 1 : 0));
    ctx.font = '600 10px ui-monospace, Menlo, Consolas, monospace'; ctx.textBaseline = 'middle';
    for (const num of nums) {
      const c = cars[num] || {}; if (c.retired) continue;
      const x = SX(cur[num][0]), y = SY(cur[num][1]);
      if (x < -20 || y < -20 || x > W + 20 || y > H + 20) continue;
      const sel = store.selected.has(num), col = c.color || '#bbb';
      const r = sel ? 7.5 : 5;
      ctx.globalAlpha = c.inPit ? 0.35 : 1;
      if (sel) {
        ctx.beginPath(); ctx.arc(x, y, 15, 0, 7); ctx.fillStyle = col; ctx.globalAlpha = 0.25 * (c.inPit ? .5 : 1) + 0.1 * pulse; ctx.fill();
        ctx.globalAlpha = c.inPit ? 0.35 : 1;
      }
      ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fillStyle = col; ctx.fill();
      ctx.lineWidth = sel ? 2 : 1; ctx.strokeStyle = sel ? '#fff' : 'rgba(0,0,0,.7)'; ctx.stroke();
      if (opts.labels || sel) {
        const txt = c.tla || num;
        ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(8,9,12,.9)'; ctx.strokeText(txt, x + r + 3, y);
        ctx.fillStyle = sel ? '#fff' : '#d4d7dd'; ctx.fillText(txt, x + r + 3, y);
      }
      ctx.globalAlpha = 1;
      hits.push({ num, x, y });
    }
    if (mouse) updateTip();
  }
  function tick(p, q, len, col, w) {
    let dx = q[0] - p[0], dy = q[1] - p[1]; const d = Math.hypot(dx, dy) || 1; dx /= d; dy /= d;
    const x = SX(p[0]), y = SY(p[1]), nx = dy, ny = dx; // screen normal (y flipped)
    ctx.beginPath(); ctx.moveTo(x - nx * len, y - ny * len); ctx.lineTo(x + nx * len, y + ny * len);
    ctx.strokeStyle = col; ctx.lineWidth = w; ctx.stroke();
  }

  // ---------- interaction ----------
  function nearest(r) {
    if (!mouse) return null; let best = null, bd = r * r;
    for (const h of hits) { const d = (h.x - mouse.x) ** 2 + (h.y - mouse.y) ** 2; if (d < bd) { bd = d; best = h; } }
    return best;
  }
  function updateTip() {
    const h = nearest(16);
    if (!h) { tip.hidden = true; cv.style.cursor = ''; return; }
    const c = store.state.cars[h.num], f = store.fast[h.num] || {};
    const sp = f.speed != null ? f.speed : c.tel && c.tel.speed;
    tip.innerHTML = `<b style="color:${c.color}">${c.tla}</b> ${c.name || ''}<br>P${c.pos} &middot; ${sp != null ? Math.round(sp) + ' km/h' : '-'}<br>gap ${c.gapLeader || '-'} &middot; int ${c.interval || '-'}`;
    tip.hidden = false; cv.style.cursor = 'pointer';
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    tip.style.left = Math.min(W - tw - 4, Math.max(4, h.x + 14)) + 'px';
    tip.style.top = Math.min(H - th - 4, Math.max(4, h.y - th - 10)) + 'px';
  }
  const pt = (e) => { const r = cv.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
  cv.addEventListener('mousemove', (e) => { mouse = pt(e); });
  cv.addEventListener('mouseleave', () => { mouse = null; tip.hidden = true; });
  cv.addEventListener('click', (e) => {
    mouse = pt(e); const h = nearest(20);
    if (h) store.select(h.num, { additive: e.shiftKey || e.ctrlKey || e.metaKey });
  });

  new ResizeObserver(() => {
    const r = el.getBoundingClientRect(); W = Math.floor(r.width); H = Math.floor(r.height);
    dpr = window.devicePixelRatio || 1;
    cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + 'px'; cv.style.height = H + 'px';
  }).observe(el);

  store.subscribe((k) => {
    if (k === 'track') { buildGeo(); }
    else if (k === 'fast') ingestAll(false);
    else if (k === 'state') ingestAll(true);
  });
  if (store.track) buildGeo();
  ingestAll(true);
  requestAnimationFrame(frame);
}
