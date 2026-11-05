// Tiny hand-drawn chart toolkit (canvas) + shared data helpers for the deep-dive / analytics panels.
// No dependencies. Styles live in css/panels.css (.ch-*).

export const COLORS = {
  bg: '#0f0f0f', panel: '#1a1a1a', grid: '#262626', axis: '#4a4a4a', text: '#bdbdbd', muted: '#808080',
  accent: '#e10600', purple: '#b24bf3', green: '#2ecc71', yellow: '#f1c40f', red: '#ff453a',
};
export const TYRE = { SOFT: '#ff3b30', MEDIUM: '#ffd60a', HARD: '#f2f2f2', INTERMEDIATE: '#34c759', WET: '#3b82f6' };
export const tyreColor = (c) => TYRE[String(c || '').toUpperCase()] || '#777';
export const tyreLetter = (c) => ({ SOFT: 'S', MEDIUM: 'M', HARD: 'H', INTERMEDIATE: 'I', WET: 'W' }[String(c || '').toUpperCase()] || '?');
const FONT = '10px -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif';

// ---------------------------------------------------------------- generic helpers
export function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
export function h(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}
export function rafThrottle(fn) {
  let id = 0;
  const run = () => { id = 0; fn(); };
  const t = () => { if (!id) id = requestAnimationFrame(run); };
  t.cancel = () => { if (id) cancelAnimationFrame(id); id = 0; };
  return t;
}
export const isNum = (v) => typeof v === 'number' && isFinite(v);
export function fmtLap(sec) {
  if (!isNum(sec)) return '–';
  const m = Math.floor(sec / 60), s = sec - m * 60;
  return m > 0 ? `${m}:${s.toFixed(3).padStart(6, '0')}` : s.toFixed(3);
}
export function fmtSec(v, d = 3) { return isNum(v) ? v.toFixed(d) : '–'; }
export function fmtDelta(v, d = 3) { return isNum(v) ? (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(d) : '–'; }
export function median(a) {
  const s = a.filter(isNum).sort((x, y) => x - y);
  if (!s.length) return null;
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
export function quantile(a, q) {
  const s = a.filter(isNum).sort((x, y) => x - y);
  if (!s.length) return null;
  const i = (s.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
  return s[lo] + (s[hi] - s[lo]) * (i - lo);
}
export function linreg(xs, ys) {
  const n = xs.length;
  if (n < 2) return null;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; }
  const d = n * sxx - sx * sx;
  if (!d) return null;
  return { slope: (n * sxy - sx * sy) / d, icpt: (sy - (n * sxy - sx * sy) / d * sx) / n };
}
export function niceTicks(min, max, n = 5) {
  if (!(max > min)) return [min];
  const step0 = (max - min) / Math.max(1, n);
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const r = step0 / mag;
  const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

// ---------------------------------------------------------------- race data helpers
const idxCache = new WeakMap();
/** Index history: byLap: Map<lap, Map<num, entry>>, byLapPos: Map<lap, Map<pos, [num, entry]>> */
export function indexHistory(history) {
  if (!history || !history.laps) return { byLap: new Map(), byLapPos: new Map(), laps: [], maxLap: 0 };
  const c = idxCache.get(history);
  if (c) return c;
  const byLap = new Map(), byLapPos = new Map();
  for (const [num, arr] of Object.entries(history.laps)) {
    for (const e of arr || []) {
      if (!byLap.has(e.lap)) { byLap.set(e.lap, new Map()); byLapPos.set(e.lap, new Map()); }
      byLap.get(e.lap).set(num, e);
      if (e.pos != null) byLapPos.get(e.lap).set(e.pos, [num, e]);
    }
  }
  const laps = [...byLap.keys()].sort((a, b) => a - b);
  const out = { byLap, byLapPos, laps, maxLap: laps.length ? laps[laps.length - 1] : 0 };
  idxCache.set(history, out);
  return out;
}
/** Colour + dash per car; teammates sharing a colour get a dashed line. Stable regardless of selection. */
export function teamStyles(cars) {
  const out = {}, seen = {};
  const nums = Object.keys(cars || {}).sort((a, b) => (cars[a].pos || 99) - (cars[b].pos || 99) || (+a) - (+b));
  for (const n of nums) {
    const col = cars[n].color || '#888';
    const k = seen[col] = (seen[col] || 0) + 1;
    out[n] = { color: col, dash: k > 1 ? [6, 4] : null };
  }
  return out;
}
export const tla = (state, n) => (state && state.cars && state.cars[n] && state.cars[n].tla) || n;
/** Safety-car / VSC / red band descriptors (x in lap units) */
export function raceBands(history) {
  const ix = indexHistory(history), out = [];
  const sev = (c) => (c === '5' ? 3 : c === '4' ? 2 : c === '6' || c === '7' ? 1 : 0);
  let cur = null;
  for (const lap of ix.laps) {
    let best = 0;
    for (const e of ix.byLap.get(lap).values()) best = Math.max(best, sev(String(e.track ?? '1')));
    if (best && cur && cur.sev === best && cur.x1 === lap - 0.5) cur.x1 = lap + 0.5;
    else if (best) {
      cur = { sev: best, x0: lap - 0.5, x1: lap + 0.5 };
      cur.color = best === 3 ? 'rgba(225,6,0,.18)' : best === 2 ? 'rgba(255,200,0,.15)' : 'rgba(255,150,0,.10)';
      cur.label = best === 3 ? 'RED' : best === 2 ? 'SC' : 'VSC';
      out.push(cur);
    } else cur = null;
  }
  return out;
}
/** Split a car's lap entries into stints. */
export function stintsFromLaps(entries) {
  const out = [];
  let cur = null, prev = null;
  for (const e of entries || []) {
    const newStint = !cur || (prev && (prev.pit || e.compound !== prev.compound || (isNum(e.tyreAge) && isNum(prev.tyreAge) && e.tyreAge < prev.tyreAge)));
    if (newStint) { cur = { compound: e.compound, laps: [], start: e.lap }; out.push(cur); }
    cur.laps.push(e); cur.end = e.lap; prev = e;
  }
  return out;
}

// ---------------------------------------------------------------- Chart
/**
 * new Chart(host, opts)  opts:
 *  series:[{id,label,color,dash,type:'line'|'scatter'|'bar'|'hbar'|'heat',points,width,fill,step,markers,r,barWidth,barH,cellW,cellH,scale,legend,short}]
 *    line/scatter/bar points [x,y,meta]; hbar [x0,x1,y,{color,text}]; heat [x,y,v]
 *  xDomain/yDomain:[min|null,max|null], yInvert, xTicks/yTicks:[{v,label}], xFmt,yFmt, left,right,top,bottom
 *  bands:[{x0,x1,color,label}], vlines:[{x,color,label,dash}], hlines:[{y,color,label}], endLabels, legend (bool),
 *  tooltipMode:'nearest'|'x', tooltip(s,p,chart)->html, onHover(s,p), onClick(s,p), onLegend(id,hiddenSet), highlight(id), height
 */
export class Chart {
  constructor(host, opts = {}) {
    this.host = host;
    this.o = Object.assign({
      series: [], left: 44, right: 10, top: 8, bottom: 22, legend: false, tooltipMode: 'nearest',
      xFmt: (v) => String(+v.toFixed(2)), yFmt: (v) => String(+v.toFixed(2)), hoverHighlight: true,
    }, opts);
    this.hidden = new Set();
    this.hl = null;
    this.hover = null;
    host.classList.add('ch-host');
    if (this.o.height) host.style.height = this.o.height + 'px';
    this.legendEl = h('div', 'ch-legend');
    this.wrap = h('div', 'ch-wrap');
    this.canvas = h('canvas', 'ch-canvas');
    this.tip = h('div', 'ch-tip');
    this.tip.style.display = 'none';
    this.wrap.append(this.canvas, this.tip);
    host.append(this.legendEl, this.wrap);
    this.ctx = this.canvas.getContext('2d');
    this.size = { w: 0, h: 0 };
    this._draw = rafThrottle(() => this.draw());
    this.ro = new ResizeObserver(() => this._resize());
    this.ro.observe(this.wrap);
    const c = this.canvas;
    c.addEventListener('pointermove', (e) => this._move(e));
    c.addEventListener('pointerdown', (e) => { if (e.pointerType !== 'mouse') this._move(e); });
    c.addEventListener('pointerleave', () => this._leave());
    c.addEventListener('click', () => { if (this.hover && this.o.onClick) this.o.onClick(this.hover.s, this.hover.p); });
    this._legendSig = '';
    this._buildLegend();
    this._resize();
  }
  destroy() { this.ro.disconnect(); this._draw.cancel(); this.host.innerHTML = ''; }
  update(patch) {
    if (patch) Object.assign(this.o, patch);
    if (patch && patch.height) this.host.style.height = patch.height + 'px';
    this._buildLegend();
    this._draw();
  }
  setHidden(ids) { this.hidden = new Set(ids); this._buildLegend(true); this._draw(); }
  redraw() { this._draw(); }

  _vis() { return this.o.series.filter((s) => !this.hidden.has(s.id) && s.points && s.points.length); }
  _resize() {
    const w = this.wrap.clientWidth, hh = this.wrap.clientHeight;
    if (w === this.size.w && hh === this.size.h) return;
    this.size = { w, h: hh };
    const d = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(w * d)); this.canvas.height = Math.max(1, Math.round(hh * d));
    this.canvas.style.width = w + 'px'; this.canvas.style.height = hh + 'px';
    this._draw();
  }
  _buildLegend(force) {
    const el = this.legendEl;
    if (!this.o.legend) { el.style.display = 'none'; return; }
    const items = this.o.series.filter((s) => s.legend !== false && s.label);
    const sig = items.map((s) => s.id + '|' + s.label + '|' + s.color + '|' + (s.dash ? 1 : 0)).join(';');
    if (sig === this._legendSig && !force) { this._legendClasses(); return; }
    this._legendSig = sig;
    el.style.display = '';
    el.innerHTML = '';
    for (const s of items) {
      const b = h('button', 'ch-chip');
      b.type = 'button';
      b.dataset.id = s.id;
      b.innerHTML = `<i style="${s.dash ? `border-top:2px dashed ${s.color}` : `background:${s.color}`}"></i>${esc(s.label)}`;
      b.addEventListener('click', () => {
        if (this.hidden.has(s.id)) this.hidden.delete(s.id); else this.hidden.add(s.id);
        this._legendClasses(); this._draw();
        if (this.o.onLegend) this.o.onLegend(s.id, this.hidden);
      });
      b.addEventListener('dblclick', () => {
        const all = items.map((x) => x.id);
        const only = this.hidden.size === all.length - 1 && !this.hidden.has(s.id);
        this.hidden = new Set(only ? [] : all.filter((x) => x !== s.id));
        this._legendClasses(); this._draw();
        if (this.o.onLegend) this.o.onLegend(s.id, this.hidden);
      });
      b.addEventListener('mouseenter', () => { this.hl = s.id; this._draw(); });
      b.addEventListener('mouseleave', () => { this.hl = null; this._draw(); });
      el.appendChild(b);
    }
    this._legendClasses();
  }
  _legendClasses() { for (const b of this.legendEl.children) b.classList.toggle('off', this.hidden.has(b.dataset.id)); }

  _layout() {
    const o = this.o, { w, h: hh } = this.size;
    const L = o.left, R = o.right, T = o.top, B = o.bottom;
    const pw = Math.max(10, w - L - R), ph = Math.max(10, hh - T - B);
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity, hasBar = false;
    for (const s of this._vis()) {
      for (const p of s.points) {
        if (s.type === 'hbar') { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[1]); y0 = Math.min(y0, p[2] - 0.5); y1 = Math.max(y1, p[2] + 0.5); continue; }
        const x = p[0], y = p[1];
        if (!isNum(x)) continue;
        let xa = x, xb = x;
        if (s.type === 'bar') { const bw = (s.barWidth || 0.7) / 2; xa -= bw; xb += bw; hasBar = true; }
        if (s.type === 'heat') { xa -= (s.cellW || 1) / 2; xb += (s.cellW || 1) / 2; }
        x0 = Math.min(x0, xa); x1 = Math.max(x1, xb);
        if (isNum(y)) {
          const ya = s.type === 'heat' ? y - (s.cellH || 1) / 2 : y, yb = s.type === 'heat' ? y + (s.cellH || 1) / 2 : y;
          y0 = Math.min(y0, ya); y1 = Math.max(y1, yb);
        }
      }
    }
    if (hasBar) { y0 = Math.min(y0, 0); }
    const xd = o.xDomain || [], yd = o.yDomain || [];
    if (!isFinite(x0)) { x0 = 0; x1 = 1; }
    if (!isFinite(y0)) { y0 = 0; y1 = 1; }
    if (x0 === x1) { x0 -= 0.5; x1 += 0.5; }
    if (y0 === y1) { y0 -= 0.5; y1 += 0.5; }
    if (yd[0] == null || yd[1] == null) {
      const pad = (y1 - y0) * 0.06;
      if (!(hasBar && y0 === 0)) y0 -= pad;
      y1 += pad;
    }
    if (xd[0] != null) x0 = xd[0]; if (xd[1] != null) x1 = xd[1];
    if (yd[0] != null) y0 = yd[0]; if (yd[1] != null) y1 = yd[1];
    if (x1 <= x0) x1 = x0 + 1; if (y1 <= y0) y1 = y0 + 1;
    const inv = !!o.yInvert;
    this.L = { L, T, pw, ph, x0, x1, y0, y1 };
    this.sx = (v) => L + ((v - x0) / (x1 - x0)) * pw;
    this.sy = (v) => T + (inv ? (v - y0) / (y1 - y0) : (y1 - v) / (y1 - y0)) * ph;
    this.ix = (px) => x0 + ((px - L) / pw) * (x1 - x0);
    this.iy = (py) => (inv ? y0 + ((py - T) / ph) * (y1 - y0) : y1 - ((py - T) / ph) * (y1 - y0));
  }

  draw() {
    const { w, h: hh } = this.size;
    if (!w || !hh) return;
    const ctx = this.ctx, d = window.devicePixelRatio || 1, o = this.o;
    ctx.setTransform(d, 0, 0, d, 0, 0);
    ctx.clearRect(0, 0, w, hh);
    this._layout();
    const { L, T, pw, ph, x0, x1, y0, y1 } = this.L;
    const sx = this.sx, sy = this.sy;
    ctx.font = FONT; ctx.textBaseline = 'middle';
    // grid + ticks
    const yt = o.yTicks || niceTicks(y0, y1, Math.max(2, Math.floor(ph / 38))).map((v) => ({ v, label: o.yFmt(v) }));
    const xt = o.xTicks || niceTicks(x0, x1, Math.max(2, Math.floor(pw / 62))).map((v) => ({ v, label: o.xFmt(v) }));
    ctx.lineWidth = 1; ctx.strokeStyle = COLORS.grid; ctx.fillStyle = COLORS.muted;
    ctx.textAlign = 'right';
    ctx.beginPath();
    for (const t of yt) {
      if (t.v < Math.min(y0, y1) - 1e-9 || t.v > Math.max(y0, y1) + 1e-9) continue;
      const y = Math.round(sy(t.v)) + 0.5;
      ctx.moveTo(L, y); ctx.lineTo(L + pw, y);
    }
    if (o.gridX) for (const t of xt) { const x = Math.round(sx(t.v)) + 0.5; ctx.moveTo(x, T); ctx.lineTo(x, T + ph); }
    ctx.stroke();
    for (const t of yt) {
      if (t.v < y0 - 1e-9 || t.v > y1 + 1e-9) continue;
      ctx.fillText(t.label, L - 6, sy(t.v));
    }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    let lastX = -1e9;
    for (const t of xt) {
      if (t.v < x0 - 1e-9 || t.v > x1 + 1e-9) continue;
      const x = sx(t.v);
      if (x - lastX < (o.xMinGap || 28)) continue;
      lastX = x;
      ctx.fillText(t.label, x, T + ph + 5);
    }
    if (o.yLabel) { ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.yLabel, 4, 0); }
    if (o.xLabel) { ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(o.xLabel, w - 2, hh); }
    ctx.textBaseline = 'middle';
    // axes
    ctx.strokeStyle = COLORS.axis; ctx.beginPath();
    ctx.moveTo(L + 0.5, T); ctx.lineTo(L + 0.5, T + ph + 0.5); ctx.lineTo(L + pw, T + ph + 0.5); ctx.stroke();

    ctx.save(); ctx.beginPath(); ctx.rect(L, T, pw, ph); ctx.clip();
    for (const b of o.bands || []) {
      ctx.fillStyle = b.color || 'rgba(255,255,255,.06)';
      ctx.fillRect(sx(b.x0), T, sx(b.x1) - sx(b.x0), ph);
      if (b.label) { ctx.fillStyle = COLORS.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(b.label, sx(b.x0) + 3, T + 2); ctx.textBaseline = 'middle'; }
    }
    for (const hl of o.hlines || []) {
      ctx.strokeStyle = hl.color || '#555'; ctx.setLineDash(hl.dash || [3, 3]);
      const y = Math.round(sy(hl.y)) + 0.5; ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + pw, y); ctx.stroke();
    }
    ctx.setLineDash([]);
    const hl = this.hl || o.highlight || null;
    for (const s of this._vis()) this._series(ctx, s, hl);
    for (const v of o.vlines || []) {
      ctx.strokeStyle = v.color || '#888'; ctx.setLineDash(v.dash || [4, 3]); ctx.lineWidth = v.width || 1;
      const x = Math.round(sx(v.x)) + 0.5; ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, T + ph); ctx.stroke();
      if (v.label) { ctx.fillStyle = v.color || '#888'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(v.label, x + 3, T + ph - 2); ctx.textBaseline = 'middle'; }
    }
    ctx.setLineDash([]); ctx.lineWidth = 1;
    // crosshair
    if (this.hover) {
      const { s, p } = this.hover;
      if (s.type !== 'hbar' && s.type !== 'heat' && isNum(p[0])) {
        ctx.strokeStyle = 'rgba(255,255,255,.28)';
        const x = Math.round(sx(p[0])) + 0.5; ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, T + ph); ctx.stroke();
        if (isNum(p[1]) && s.type !== 'bar') {
          ctx.fillStyle = s.color; ctx.strokeStyle = '#fff'; ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), 4, 0, 6.3); ctx.fill(); ctx.stroke();
        }
      }
    }
    ctx.restore();
    if (o.endLabels) this._endLabels(ctx, hl);
  }

  _series(ctx, s, hl) {
    const sx = this.sx, sy = this.sy, { T, ph, L, pw } = this.L;
    const dim = hl && hl !== s.id;
    const color = s.color || '#aaa';
    ctx.globalAlpha = dim ? 0.18 : 1;
    const t = s.type || 'line';
    if (t === 'line') {
      ctx.strokeStyle = color; ctx.lineWidth = (s.width || 1.6) + (hl === s.id ? 1.2 : 0); ctx.lineJoin = 'round';
      ctx.setLineDash(s.dash || []);
      ctx.beginPath();
      let pen = false, px = 0, py = 0;
      for (const p of s.points) {
        if (!isNum(p[0]) || !isNum(p[1])) { pen = false; continue; }
        const x = sx(p[0]), y = sy(p[1]);
        if (!pen) { ctx.moveTo(x, y); pen = true; } else if (s.step) { ctx.lineTo(x, py); ctx.lineTo(x, y); } else ctx.lineTo(x, y);
        px = x; py = y;
      }
      ctx.stroke(); ctx.setLineDash([]);
      if (s.fill) {
        ctx.fillStyle = s.fill; const base = sy(Math.max(this.L.y0, Math.min(this.L.y1, s.base ?? 0)));
        let seg = [];
        const flush = () => {
          if (seg.length > 1) {
            ctx.beginPath(); ctx.moveTo(seg[0][0], base);
            let ly = base;
            for (const q of seg) { if (s.step) { ctx.lineTo(q[0], ly); } ctx.lineTo(q[0], q[1]); ly = q[1]; }
            ctx.lineTo(seg[seg.length - 1][0], base); ctx.closePath(); ctx.fill();
          }
          seg = [];
        };
        for (const p of s.points) { if (!isNum(p[0]) || !isNum(p[1])) flush(); else seg.push([sx(p[0]), sy(p[1])]); }
        flush();
      }
      if (s.markers) {
        ctx.fillStyle = color;
        for (const p of s.points) if (isNum(p[0]) && isNum(p[1])) { ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), s.r || 2.2, 0, 6.3); ctx.fill(); }
      }
    } else if (t === 'scatter') {
      ctx.fillStyle = color; const r = (s.r || 2.6) + (hl === s.id ? 1 : 0);
      for (const p of s.points) {
        if (!isNum(p[0]) || !isNum(p[1])) continue;
        const m = p[2];
        ctx.fillStyle = (m && m.color) || color;
        ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), r, 0, 6.3); ctx.fill();
      }
    } else if (t === 'bar') {
      const bw = Math.max(1, Math.abs(sx(s.barWidth || 0.7) - sx(0)) - 1), base = sy(0);
      for (const p of s.points) {
        if (!isNum(p[1])) continue;
        const m = p[2];
        ctx.fillStyle = (m && m.color) || color;
        const x = sx(p[0]) - bw / 2, y = sy(p[1]);
        ctx.fillRect(x, Math.min(y, base), bw, Math.max(1, Math.abs(base - y)));
      }
    } else if (t === 'hbar') {
      const bh = (s.barH || 0.7) * Math.abs(sy(1) - sy(0));
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      for (const p of s.points) {
        const m = p[3] || {};
        const xa = sx(p[0]), xb = sx(p[1]), y = sy(p[2]) - bh / 2;
        ctx.fillStyle = m.color || color;
        ctx.fillRect(xa + 0.5, y, Math.max(1, xb - xa - 1), bh);
        if (m.text && xb - xa > 18) { ctx.fillStyle = m.textColor || '#111'; ctx.fillText(m.text, (xa + xb) / 2, y + bh / 2 + 0.5); }
      }
    } else if (t === 'heat') {
      const cw = Math.abs(sx(s.cellW || 1) - sx(0)), ch = Math.abs(sy(s.cellH || 1) - sy(0));
      for (const p of s.points) {
        if (!isNum(p[2])) continue;
        ctx.fillStyle = s.scale ? s.scale(p[2]) : heatColor(p[2]);
        ctx.fillRect(sx(p[0]) - cw / 2, sy(p[1]) - ch / 2, cw - 1, ch - 1);
      }
    }
    ctx.globalAlpha = 1;
  }

  _endLabels(ctx, hl) {
    const items = [];
    for (const s of this._vis()) {
      if ((s.type || 'line') !== 'line') continue;
      for (let i = s.points.length - 1; i >= 0; i--) {
        const p = s.points[i];
        if (isNum(p[0]) && isNum(p[1])) { items.push({ y: this.sy(p[1]), x: this.sx(p[0]), t: s.short || s.label || s.id, c: s.color, id: s.id }); break; }
      }
    }
    items.sort((a, b) => a.y - b.y);
    for (let i = 1; i < items.length; i++) if (items[i].y - items[i - 1].y < 9) items[i].y = items[i - 1].y + 9;
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.font = 'bold ' + FONT;
    for (const it of items) { ctx.globalAlpha = hl && hl !== it.id ? 0.3 : 1; ctx.fillStyle = it.c; ctx.fillText(it.t, Math.min(it.x + 4, this.size.w - this.o.right + 3), it.y); }
    ctx.globalAlpha = 1; ctx.font = FONT;
  }

  _pick(mx, my) {
    if (!this.L) return null;
    const xv = this.ix(mx), yv = this.iy(my);
    let best = null, bs = Infinity;
    for (const s of this._vis()) {
      const t = s.type || 'line';
      for (const p of s.points) {
        let sc = Infinity;
        if (t === 'bar') {
          if (!isNum(p[1])) continue;
          const dx = Math.abs(this.sx(p[0]) - mx), bw = Math.abs(this.sx(s.barWidth || 0.7) - this.sx(0)) / 2;
          if (dx <= bw + 1) sc = dx * 0.01;
        } else if (t === 'hbar') {
          if (xv >= p[0] && xv <= p[1] && Math.abs(yv - p[2]) <= (s.barH || 0.7) / 2) sc = 0;
        } else if (t === 'heat') {
          if (Math.abs(xv - p[0]) <= (s.cellW || 1) / 2 && Math.abs(yv - p[1]) <= (s.cellH || 1) / 2) sc = 0;
        } else {
          if (!isNum(p[0]) || !isNum(p[1])) continue;
          const dx = this.sx(p[0]) - mx, dy = this.sy(p[1]) - my;
          sc = (t === 'line' ? dx * dx * 0.35 : dx * dx) + dy * dy;
          if (sc > (t === 'line' ? 2500 : 900)) sc = Infinity;
        }
        if (sc < bs) { bs = sc; best = { s, p }; }
      }
    }
    return best;
  }
  _move(e) {
    const r = this.canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const pk = this._pick(mx, my);
    const prev = this.hover;
    this.hover = pk;
    if (this.o.hoverHighlight) this.hl = pk && (pk.s.type || 'line') !== 'bar' && this.o.series.length > 1 ? pk.s.id : null;
    if (this.o.onHover && (!prev || !pk || prev.s !== pk.s || prev.p !== pk.p)) this.o.onHover(pk && pk.s, pk && pk.p);
    if (!pk) { this.tip.style.display = 'none'; this._draw(); return; }
    this.tip.innerHTML = this._tipHtml(pk);
    this.tip.style.display = 'block';
    const tw = this.tip.offsetWidth, th = this.tip.offsetHeight;
    let tx = mx + 14, ty = my + 12;
    if (tx + tw > this.size.w - 2) tx = mx - tw - 14;
    if (ty + th > this.size.h - 2) ty = this.size.h - th - 2;
    this.tip.style.left = Math.max(0, tx) + 'px'; this.tip.style.top = Math.max(0, ty) + 'px';
    this._draw();
  }
  _leave() {
    this.hover = null; this.hl = null; this.tip.style.display = 'none';
    if (this.o.onHover) this.o.onHover(null, null);
    this._draw();
  }
  _tipHtml({ s, p }) {
    const o = this.o;
    if (o.tooltip) { const r = o.tooltip(s, p, this); if (r) return r; }
    if (o.tooltipMode === 'x' && (s.type || 'line') === 'line') {
      const rows = [];
      for (const q of this._vis()) {
        if ((q.type || 'line') !== 'line') continue;
        const pt = q.points.find((z) => z[0] === p[0] && isNum(z[1]));
        if (pt) rows.push({ s: q, y: pt[1] });
      }
      rows.sort((a, b) => (o.yInvert ? a.y - b.y : b.y - a.y));
      const i = rows.findIndex((r) => r.s === s);
      const lo = Math.max(0, Math.min(i - 4, rows.length - 9)), sl = rows.slice(lo, lo + 9);
      return `<b>${esc(o.xName || '')} ${o.xFmt(p[0])}</b>` + sl.map((r) =>
        `<div class="ch-row${r.s === s ? ' sel' : ''}"><i style="background:${r.s.color}"></i>${esc(r.s.label || r.s.id)}<span>${o.yFmt(r.y)}</span></div>`).join('');
    }
    return `<b>${esc(s.label || s.id)}</b><div>${esc(o.xName || 'x')} ${o.xFmt(p[0])} &nbsp; ${isNum(p[1]) ? o.yFmt(p[1]) : ''}</div>`;
  }
}

export function heatColor(v) { // v 0..1 : cool -> hot
  v = Math.max(0, Math.min(1, v));
  const hue = 220 - v * 220;
  return `hsl(${hue},70%,${35 + v * 12}%)`;
}
