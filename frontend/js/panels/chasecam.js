// 3D chase-cam panel (three.js r160, vendored). Pure ES module, no build step.
// World mapping: data (x, y) metres -> three (X = x, Z = -y), elevation -> Y. Seen from above with "up" = -Z this
// matches the 2D track map (x to the right, y up), i.e. no mirroring.
import * as THREE from '../../vendor/three.module.min.js';
import { ls } from '../util.js';

const MODES = [
  ['chase', 'Chase', 'C', 'Behind and above the followed car'],
  ['onboard', 'Onboard~', 'O', 'Approximate onboard: a camera at the car nose derived from track geometry, not real onboard video'],
  ['tv', 'TV', 'T', 'Trackside TV cameras that hand over as the car passes'],
  ['heli', 'Heli', 'H', 'Helicopter orbiting the followed car'],
  ['free', 'Free', 'R', 'Free orbit: drag to rotate, wheel / pinch to zoom'],
];
const TRACK_W = 13;          // asphalt width, metres
const KERB_W = 1.4;
const DEG = Math.PI / 180;

export function mount(host, store) {
  host.textContent = '';
  const el = document.createElement('div'); host.append(el);
  const prefs = Object.assign({ mode: 'chase', hud: true }, ls.get('f1.chase', {}));
  if (!MODES.some((m) => m[0] === prefs.mode)) prefs.mode = 'chase';
  const savePrefs = () => ls.set('f1.chase', prefs);

  // ------------------------------------------------------------------ DOM
  el.className = 'cc';
  const cv = document.createElement('canvas'); cv.className = 'cc-canvas';
  const msg = document.createElement('div'); msg.className = 'cc-msg'; msg.textContent = 'Waiting for track data...';
  const bar = document.createElement('div'); bar.className = 'cc-bar';
  const note = document.createElement('div'); note.className = 'cc-note';
  const hud = document.createElement('div'); hud.className = 'cc-hud';
  hud.innerHTML = `
    <div class="cc-card">
      <div class="cc-id"><i class="cc-sw"></i><b class="cc-tla">---</b><span class="cc-name"></span><span class="cc-pos"></span></div>
      <div class="cc-main"><div class="cc-speed"><b>0</b><small>km/h</small></div><div class="cc-gear">N</div><div class="cc-drs">DRS</div></div>
      <div class="cc-leds"></div>
      <div class="cc-pedals"><div class="cc-pd"><span>THR</span><div class="cc-trk"><i class="cc-thr"></i></div></div><div class="cc-pd"><span>BRK</span><div class="cc-trk"><i class="cc-brk"></i></div></div></div>
      <div class="cc-kv"><span>Ahead</span><b class="cc-ahead">-</b><span>Behind</span><b class="cc-behind">-</b><span>Tyre</span><b class="cc-tyre">-</b><span>Lap</span><b class="cc-lap">-</b><span>Last</span><b class="cc-last">-</b><span>Gap L</span><b class="cc-gapl">-</b></div>
    </div>
    <div class="cc-list"></div>`;
  el.append(cv, msg, bar, note, hud);
  const q = (s) => hud.querySelector(s);
  const E = {
    sw: q('.cc-sw'), tla: q('.cc-tla'), name: q('.cc-name'), pos: q('.cc-pos'), speed: q('.cc-speed b'), gear: q('.cc-gear'), drs: q('.cc-drs'),
    leds: q('.cc-leds'), thr: q('.cc-thr'), brk: q('.cc-brk'), ahead: q('.cc-ahead'), behind: q('.cc-behind'), tyre: q('.cc-tyre'),
    lap: q('.cc-lap'), last: q('.cc-last'), gapl: q('.cc-gapl'), list: q('.cc-list'),
  };
  const leds = [];
  for (let i = 0; i < 10; i++) { const d = document.createElement('i'); d.className = i < 4 ? 'g' : i < 8 ? 'r' : 'b'; E.leds.append(d); leds.push(d); }

  let snap = true;
  const modeBtns = {};
  for (const [k, t, key, tip] of MODES) {
    const b = document.createElement('button'); b.dataset.mode = k; b.title = `${tip} (${key})`;
    b.innerHTML = `${t}<kbd>${key}</kbd>`;
    b.onclick = () => setMode(k);
    bar.append(b); modeBtns[k] = b;
  }
  const hudBtn = document.createElement('button'); hudBtn.className = 'cc-hudbtn'; hudBtn.title = 'Hide / show HUD (I)'; hudBtn.textContent = 'HUD';
  hudBtn.onclick = () => setHud(!prefs.hud);
  bar.append(hudBtn);

  function setHud(on) { prefs.hud = on; hud.hidden = !on; hudBtn.classList.toggle('on', on); savePrefs(); }
  function setMode(m) {
    prefs.mode = m; savePrefs(); snap = true;
    for (const k in modeBtns) modeBtns[k].classList.toggle('on', k === m);
    note.textContent = 'Approximate onboard view: derived from track geometry, not real onboard video';
    note.hidden = m !== 'onboard';
    cv.style.touchAction = m === 'free' ? 'none' : 'pan-y';
    cv.style.cursor = m === 'free' ? 'grab' : '';
  }

  // ------------------------------------------------------------------ renderer
  let renderer = null, glFail = '';
  try {
    renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true, powerPreference: 'high-performance' });
  } catch (e) {
    glFail = 'WebGL is not available in this browser, so the 3D view cannot be drawn. The HUD and timing list still work.';
    renderer = null;
  }
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, 1, 0.4, 9000);
  const SKY_TOP = new THREE.Color('#2a5d9a'), SKY_HOR = new THREE.Color('#b9d2e6'), SKY_BOT = new THREE.Color('#58704a');
  scene.fog = new THREE.Fog(SKY_HOR.clone().lerp(new THREE.Color('#7f9a8a'), 0.15), 600, 3200);
  scene.add(new THREE.HemisphereLight('#dfeaff', '#4a5a40', 1.15));
  const sun = new THREE.DirectionalLight('#fff3dc', 1.6); sun.position.set(300, 500, 200); scene.add(sun);

  const sky = (() => {
    const g = new THREE.SphereGeometry(4500, 24, 12);
    const p = g.attributes.position, col = new Float32Array(p.count * 3), c = new THREE.Color();
    for (let i = 0; i < p.count; i++) {
      const y = p.getY(i) / 4500;
      if (y >= 0) c.copy(SKY_HOR).lerp(SKY_TOP, Math.pow(y, 0.55)); else c.copy(SKY_HOR).lerp(SKY_BOT, Math.min(1, -y * 3));
      col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
    }
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide, fog: false, depthWrite: false }));
    m.renderOrder = -10; m.frustumCulled = false; scene.add(m); return m;
  })();

  // ------------------------------------------------------------------ track model
  let G = null, trackGroup = null, trackSig = '';
  const P = { x: 0, y: 0, z: 0, tx: 0, tz: 1, pitch: 0 };    // scratch pose (no per-frame allocation)

  function disposeGroup(g) {
    if (!g) return;
    scene.remove(g);
    g.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
  }

  function buildTrack() {
    const t = store.track;
    if (!t || !t.points || t.points.length < 4) return;
    const eSig = t.elevation ? t.elevation.reduce((a, v, i) => a + (v || 0) * ((i % 7) + 1), 0).toFixed(1) : 'none';
    const sig = [t.points.length, t.length, t.startIdx, t.rotation, JSON.stringify(t.drsZones || []), JSON.stringify(t.sectors || []), eSig].join('|');
    if (sig === trackSig) return;
    trackSig = sig;

    const a = (t.rotation || 0) * DEG, ca = Math.cos(a), sa = Math.sin(a);
    const si = t.startIdx || 0, n0 = t.points.length;
    const el0 = t.elevation && t.elevation.length === n0 ? t.elevation : null;
    const rx = [], ry = [], ez = [];
    for (let k = 0; k < n0; k++) {
      const i = (k + si) % n0, [x, y] = t.points[i];
      rx.push(x * ca - y * sa); ry.push(x * sa + y * ca);
      ez.push(el0 && el0[i] != null ? el0[i] : 0);
    }
    rx.push(rx[0]); ry.push(ry[0]); ez.push(ez[0]);
    const cum = [0];
    for (let i = 1; i < rx.length; i++) cum.push(cum[i - 1] + Math.hypot(rx[i] - rx[i - 1], ry[i] - ry[i - 1]));
    const rawLen = cum[cum.length - 1] || 1;
    const scale = t.length > 0 ? t.length / rawLen : 0.1;          // raw units -> metres
    const L = rawLen * scale;
    const N = Math.max(200, Math.min(4000, Math.round(L / 3)));
    const ds = L / N;

    const X = new Float32Array(N), Y = new Float32Array(N), Z = new Float32Array(N);
    let j = 1;
    for (let i = 0; i < N; i++) {
      const d = (i / N) * rawLen;
      while (j < cum.length - 1 && cum[j] < d) j++;
      const c0 = cum[j - 1], c1 = cum[j], u = c1 > c0 ? (d - c0) / (c1 - c0) : 0;
      X[i] = (rx[j - 1] + (rx[j] - rx[j - 1]) * u) * scale;
      Z[i] = -(ry[j - 1] + (ry[j] - ry[j - 1]) * u) * scale;
      Y[i] = ez[j - 1] + (ez[j] - ez[j - 1]) * u;
    }
    const smooth = (arr, w, passes) => {
      const tmp = new Float32Array(N);
      for (let p = 0; p < passes; p++) {
        for (let i = 0; i < N; i++) { let s = 0; for (let k = -w; k <= w; k++) s += arr[(i + k + N) % N]; tmp[i] = s / (2 * w + 1); }
        arr.set(tmp);
      }
    };
    smooth(X, 2, 1); smooth(Z, 2, 1); smooth(Y, Math.max(2, Math.round(20 / ds)), 2);
    let minY = 1e9, maxY = -1e9, minX = 1e9, maxX = -1e9, minZ = 1e9, maxZ = -1e9;
    const TX = new Float32Array(N), TZ = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      minY = Math.min(minY, Y[i]); maxY = Math.max(maxY, Y[i]);
      minX = Math.min(minX, X[i]); maxX = Math.max(maxX, X[i]); minZ = Math.min(minZ, Z[i]); maxZ = Math.max(maxZ, Z[i]);
      const a1 = (i + 1) % N, a0 = (i - 1 + N) % N;
      const dx = X[a1] - X[a0], dz = Z[a1] - Z[a0], d = Math.hypot(dx, dz) || 1;
      TX[i] = dx / d; TZ[i] = dz / d;
    }
    G = { N, L, ds, X, Y, Z, TX, TZ, minY, maxY, groundY: minY - 1.5, minX, maxX, minZ, maxZ, cx: (minX + maxX) / 2, cz: (minZ + maxZ) / 2,
      size: Math.max(maxX - minX, maxZ - minZ), cosA: ca, sinA: sa, rawScale: scale };

    disposeGroup(trackGroup);
    trackGroup = new THREE.Group(); scene.add(trackGroup);
    buildRibbon(t); buildGround(); buildProps(); buildTvCams();
    msg.hidden = true;
    snap = true;
  }

  const mix = (a, b, u) => a + (b - a) * u;
  function buildRibbon(t) {
    const N = G.N, hw = TRACK_W / 2;
    const cols = [-(hw + KERB_W), -hw, -(hw - 0.35), hw - 0.35, hw, hw + KERB_W];
    const NQ = cols.length - 1;
    const pos = new Float32Array(N * NQ * 6 * 3), col = new Float32Array(N * NQ * 6 * 3);
    const asphalt = new THREE.Color('#3b3e46'), asphalt2 = new THREE.Color('#42454d'), line = new THREE.Color('#e8e8e8');
    const kRed = new THREE.Color('#d8201a'), kWhite = new THREE.Color('#f1f1f1'), drs = new THREE.Color('#1f8f58');
    const inDrs = new Uint8Array(N);
    for (const z of t.drsZones || []) {
      const s0 = Math.round(z[0] * N) % N, s1 = Math.round(z[1] * N) % N;
      for (let k = s0, guard = 0; k !== s1 && guard < N; k = (k + 1) % N, guard++) inDrs[k] = 1;
    }
    let o = 0;
    const tmpC = new THREE.Color();
    const vert = (i, ci, c) => {
      const lat = cols[ci], nx = G.TZ[i], nz = -G.TX[i];
      pos[o * 3] = G.X[i] + nx * lat; pos[o * 3 + 1] = G.Y[i]; pos[o * 3 + 2] = G.Z[i] + nz * lat;
      col[o * 3] = c.r; col[o * 3 + 1] = c.g; col[o * 3 + 2] = c.b; o++;
    };
    const stripeLen = Math.max(1, Math.round(5 / G.ds));
    for (let i = 0; i < N; i++) {
      const i2 = (i + 1) % N, stripe = Math.floor(i / stripeLen) % 2;
      for (let qd = 0; qd < NQ; qd++) {
        let c;
        if (qd === 0 || qd === NQ - 1) c = stripe ? kRed : kWhite;
        else if (qd === 1 || qd === NQ - 2) c = line;
        else { c = (Math.floor(i / 20) % 2) ? asphalt : asphalt2; if (inDrs[i]) { tmpC.copy(c).lerp(drs, 0.6); c = tmpC; } }
        vert(i, qd, c); vert(i2, qd, c); vert(i, qd + 1, c);
        vert(i, qd + 1, c); vert(i2, qd, c); vert(i2, qd + 1, c);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.computeVertexNormals();
    const nrm = g.attributes.normal; for (let i = 0; i < nrm.count; i++) { if (nrm.getY(i) < 0) nrm.setXYZ(i, -nrm.getX(i), -nrm.getY(i), -nrm.getZ(i)); }
    const mesh = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }));
    mesh.frustumCulled = false; trackGroup.add(mesh);

    // embankment skirts down to the ground
    const sk = [], skc = [];
    const dark = new THREE.Color('#3f5a35'), darker = new THREE.Color('#2b3d26');
    for (const side of [-1, 1]) {
      const lat = side * (hw + KERB_W), off = side * 8, gy = G.groundY;
      for (let i = 0; i < N; i++) {
        const i2 = (i + 1) % N;
        const ax = G.X[i] + G.TZ[i] * lat, az = G.Z[i] - G.TX[i] * lat, bx = G.X[i2] + G.TZ[i2] * lat, bz = G.Z[i2] - G.TX[i2] * lat;
        const ox = G.TZ[i] * off, oz = -G.TX[i] * off, ox2 = G.TZ[i2] * off, oz2 = -G.TX[i2] * off;
        sk.push(ax, G.Y[i], az, bx, G.Y[i2], bz, ax + ox, gy, az + oz, bx, G.Y[i2], bz, bx + ox2, gy, bz + oz2, ax + ox, gy, az + oz);
        for (let k = 0; k < 6; k++) { const c = k < 2 || k === 3 ? dark : darker; skc.push(c.r, c.g, c.b); }
      }
    }
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.Float32BufferAttribute(sk, 3)); sg.setAttribute('color', new THREE.Float32BufferAttribute(skc, 3));
    sg.computeVertexNormals();
    const skm = new THREE.Mesh(sg, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide })); skm.frustumCulled = false; trackGroup.add(skm);

    // marks: start/finish chequer, sector ticks, DRS zone start lines
    const mp = [], mc = [];
    const quad = (s, w0, w1, len, c, lift) => {
      const i = ((Math.floor(s) % N) + N) % N, nx = G.TZ[i], nz = -G.TX[i];
      const hx = G.TX[i] * len / 2, hz = G.TZ[i] * len / 2, cx = G.X[i], cy = G.Y[i] + lift, cz = G.Z[i];
      const A = [cx + nx * w0 - hx, cy, cz + nz * w0 - hz], B = [cx + nx * w1 - hx, cy, cz + nz * w1 - hz];
      const C = [cx + nx * w1 + hx, cy, cz + nz * w1 + hz], D = [cx + nx * w0 + hx, cy, cz + nz * w0 + hz];
      mp.push(...A, ...D, ...B, ...B, ...D, ...C);
      for (let k = 0; k < 6; k++) mc.push(c.r, c.g, c.b);
    };
    const W = new THREE.Color('#ffffff'), K = new THREE.Color('#101010');
    const cells = 10, cw = (TRACK_W - 0.7) / cells, off2 = Math.max(1, Math.round(1.6 / G.ds));
    for (let c = 0; c < cells; c++) {
      quad(0, -hw + 0.35 + c * cw, -hw + 0.35 + (c + 1) * cw, 1.6, c % 2 ? W : K, 0.05);
      quad(off2, -hw + 0.35 + c * cw, -hw + 0.35 + (c + 1) * cw, 1.6, c % 2 ? K : W, 0.05);
    }
    const sec = t.sectors || [1 / 3, 2 / 3], SC = [new THREE.Color('#3a86ff'), new THREE.Color('#ffd400')];
    sec.slice(0, 2).forEach((f, k) => quad(f * N, -hw + 0.35, hw - 0.35, 1.2, SC[k], 0.05));
    for (const z of t.drsZones || []) quad(z[0] * N, -hw + 0.35, hw - 0.35, 1.0, new THREE.Color('#00d96a'), 0.055);
    const mg = new THREE.BufferGeometry();
    mg.setAttribute('position', new THREE.Float32BufferAttribute(mp, 3)); mg.setAttribute('color', new THREE.Float32BufferAttribute(mc, 3));
    const mm = new THREE.Mesh(mg, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }));
    mm.frustumCulled = false; trackGroup.add(mm);
  }

  function buildGround() {
    const s = Math.max(6000, G.size * 4);
    const gm = new THREE.Mesh(new THREE.PlaneGeometry(s, s), new THREE.MeshLambertMaterial({ color: '#52703f' }));
    gm.rotation.x = -Math.PI / 2; gm.position.set(G.cx, G.groundY - 0.05, G.cz);
    trackGroup.add(gm);
  }

  function buildProps() {
    let seed = 1234567; const rnd = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
    const N = G.N, NT = 260;
    const tg = new THREE.ConeGeometry(5, 15, 6); tg.translate(0, 7.5, 0);
    const trees = new THREE.InstancedMesh(tg, new THREE.MeshLambertMaterial({ color: '#2f5a33' }), NT);
    const m4 = new THREE.Matrix4(), qn = new THREE.Quaternion(), sv = new THREE.Vector3(), pv = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0);
    for (let k = 0; k < NT; k++) {
      const i = Math.floor(rnd() * N), side = rnd() < 0.5 ? -1 : 1, lat = side * (28 + rnd() * 110), s = 0.7 + rnd() * 0.9;
      pv.set(G.X[i] + G.TZ[i] * lat, G.groundY, G.Z[i] - G.TX[i] * lat); sv.set(s, s * (0.8 + rnd() * 0.6), s);
      m4.compose(pv, qn.identity(), sv); trees.setMatrixAt(k, m4);
    }
    trees.frustumCulled = false; trackGroup.add(trees);
    const spots = [0, 0.012, -0.012, 0.025, 0.25, 0.5, 0.75].map((f) => ((Math.round(f * N) % N) + N) % N);
    const bg = new THREE.BoxGeometry(1, 1, 1); bg.translate(0, 0.5, 0);
    const stands = new THREE.InstancedMesh(bg, new THREE.MeshLambertMaterial({ color: '#8d93a0' }), spots.length * 2);
    let k = 0;
    for (const i of spots) for (const side of [-1, 1]) {
      const lat = side * (TRACK_W / 2 + KERB_W + 14);
      qn.setFromAxisAngle(up, Math.atan2(G.TX[i], G.TZ[i]));
      pv.set(G.X[i] + G.TZ[i] * lat, G.groundY, G.Z[i] - G.TX[i] * lat); sv.set(8, (G.Y[i] - G.groundY) + 8, 34);
      m4.compose(pv, qn, sv); stands.setMatrixAt(k++, m4);
    }
    stands.frustumCulled = false; trackGroup.add(stands);
  }

  function pose(f) {
    const N = G.N; f = f - Math.floor(f);
    const u0 = f * N, i = Math.floor(u0) % N, i2 = (i + 1) % N, u = u0 - Math.floor(u0);
    P.x = mix(G.X[i], G.X[i2], u); P.y = mix(G.Y[i], G.Y[i2], u); P.z = mix(G.Z[i], G.Z[i2], u);
    const tx = mix(G.TX[i], G.TX[i2], u), tz = mix(G.TZ[i], G.TZ[i2], u), d = Math.hypot(tx, tz) || 1;
    P.tx = tx / d; P.tz = tz / d;
    P.pitch = Math.atan2(G.Y[i2] - G.Y[i], G.ds);
    return P;
  }
  const idxOf = (f) => Math.floor((f - Math.floor(f)) * G.N) % G.N;
  function nearestFrac(x, z) {
    let best = 0, bd = 1e18;
    for (let i = 0; i < G.N; i += 2) { const d = (G.X[i] - x) ** 2 + (G.Z[i] - z) ** 2; if (d < bd) { bd = d; best = i; } }
    return best / G.N;
  }

  let tvCams = [];
  function buildTvCams() {
    tvCams = [];
    const n = Math.max(6, Math.min(18, Math.round(G.L / 450)));
    for (let k = 0; k < n; k++) {
      const f = k / n, i = Math.floor(f * G.N), side = k % 2 ? 1 : -1, lat = side * (TRACK_W / 2 + KERB_W + 16 + (k % 3) * 6);
      tvCams.push({ f, x: G.X[i] + G.TZ[i] * lat, y: G.Y[i] + 4.5 + (k % 3) * 2, z: G.Z[i] - G.TX[i] * lat });
    }
  }

  // ------------------------------------------------------------------ cars (one InstancedMesh, one draw call)
  const MAXC = 24;
  function carGeometry() {
    const parts = [];
    const add = (w, h, d, x, y, z, color) => {
      const g = new THREE.BoxGeometry(w, h, d).toNonIndexed(); g.translate(x, y, z);
      const n = g.attributes.position.count, c = new Float32Array(n * 3), cc = new THREE.Color(color);
      for (let i = 0; i < n; i++) { c[i * 3] = cc.r; c[i * 3 + 1] = cc.g; c[i * 3 + 2] = cc.b; }
      g.setAttribute('color', new THREE.BufferAttribute(c, 3)); parts.push(g);
    };
    // +Z forward. White parts get tinted by instanceColor (team colour); dark parts stay dark.
    add(0.9, 0.5, 3.4, 0, 0.45, 0.2, '#ffffff');
    add(0.5, 0.28, 1.9, 0, 0.36, 2.35, '#ffffff');
    add(1.9, 0.08, 0.5, 0, 0.18, 3.2, '#222222');
    add(1.0, 0.45, 1.2, 0, 0.55, -1.1, '#ffffff');
    add(1.4, 0.12, 0.7, 0, 0.62, -2.35, '#ffffff');
    add(0.1, 0.55, 0.6, 0.72, 0.4, -2.35, '#222222'); add(0.1, 0.55, 0.6, -0.72, 0.4, -2.35, '#222222');
    add(0.3, 0.2, 0.4, 0, 0.82, 0.3, '#151515');
    add(0.5, 0.4, 0.8, 1.0, 0.33, 1.7, '#111111'); add(0.5, 0.4, 0.8, -1.0, 0.33, 1.7, '#111111');
    add(0.6, 0.45, 0.9, 1.0, 0.36, -1.5, '#111111'); add(0.6, 0.45, 0.9, -1.0, 0.36, -1.5, '#111111');
    let total = 0; for (const g of parts) total += g.attributes.position.count;
    const pos = new Float32Array(total * 3), col = new Float32Array(total * 3), nor = new Float32Array(total * 3);
    let o = 0;
    for (const g of parts) {
      pos.set(g.attributes.position.array, o * 3); col.set(g.attributes.color.array, o * 3); nor.set(g.attributes.normal.array, o * 3);
      o += g.attributes.position.count; g.dispose();
    }
    const out = new THREE.BufferGeometry();
    out.setAttribute('position', new THREE.BufferAttribute(pos, 3)); out.setAttribute('normal', new THREE.BufferAttribute(nor, 3)); out.setAttribute('color', new THREE.BufferAttribute(col, 3));
    return out;
  }
  let carMesh = null;
  const carIdx = new Map(), carNums = [], labels = [];
  if (renderer) {
    carMesh = new THREE.InstancedMesh(carGeometry(), new THREE.MeshLambertMaterial({ vertexColors: true }), MAXC);
    carMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    for (let i = 0; i < MAXC; i++) carMesh.setColorAt(i, new THREE.Color('#bbbbbb'));
    carMesh.frustumCulled = false; carMesh.count = 0; scene.add(carMesh);
  }
  function makeLabel(text, color) {
    const c = document.createElement('canvas'); c.width = 128; c.height = 48;
    const g = c.getContext('2d'); g.font = '700 28px system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillStyle = 'rgba(8,10,14,.78)'; g.fillRect(2, 4, 124, 40);
    g.fillStyle = color; g.fillRect(2, 4, 8, 40);
    g.fillStyle = '#fff'; g.fillText(text, 70, 25);
    const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true, fog: false }));
    sp.renderOrder = 20; sp.frustumCulled = false; sp.visible = false; scene.add(sp);
    return sp;
  }
  const _col = new THREE.Color();
  function syncCars() {
    const cars = (store.state && store.state.cars) || {};
    if (!carMesh) return;
    let added = false;
    for (const num of Object.keys(cars)) {
      if (carIdx.has(num) || carNums.length >= MAXC) continue;
      const slot = carNums.length; carNums.push(num); carIdx.set(num, slot);
      const c = cars[num], color = c.color || '#bbbbbb';
      carMesh.setColorAt(slot, _col.set(color));
      labels[slot] = makeLabel(c.tla || num, color); added = true;
    }
    if (added) { carMesh.instanceColor.needsUpdate = true; carMesh.count = carNums.length; }
  }

  // ------------------------------------------------------------------ car motion (same convention as the 2D map)
  const anim = {};
  function ingest(num, d, now) {
    if (!d || (d.x == null && d.progress == null)) return;
    const a = anim[num] || (anim[num] = { t: now, v: 0, p: null, rp: null, lat: 0, lx: undefined, ly: undefined, lp: undefined, x: null, y: null });
    if (d.x === a.lx && d.y === a.ly && d.progress === a.lp) return;
    const dt = now - a.t;
    if (d.progress != null) {
      if (a.p != null && dt > 40 && dt < 2500) {
        let dp = d.progress - a.p; dp -= Math.round(dp);
        a.v = Math.max(0, a.v * 0.4 + (dp / dt) * 0.6);
      }
      a.p = d.progress; if (a.rp == null) a.rp = d.progress;
    }
    if (d.x != null) { a.x = d.x; a.y = d.y; }
    a.t = now; a.lx = d.x; a.ly = d.y; a.lp = d.progress;
  }
  function ingestAll(useState) {
    const now = performance.now();
    for (const num in store.fast) ingest(num, store.fast[num], now);
    if (useState && store.state) {
      for (const num in store.state.cars) {
        if (store.fast[num]) continue;
        const c = store.state.cars[num];
        ingest(num, { x: c.xy && c.xy.x, y: c.xy && c.xy.y, progress: c.progress }, now);
      }
    }
  }

  function followed() {
    const st = store.state; if (!st) return null;
    const sel = [...store.selected][0];
    if (sel && st.cars[sel]) return sel;
    return (st.order && st.order[0]) || null;
  }

  // ------------------------------------------------------------------ camera
  let W = 0, H = 0;
  const camPos = new THREE.Vector3(), camLook = new THREE.Vector3(), wantPos = new THREE.Vector3(), wantLook = new THREE.Vector3();
  let fov = 62, yaw = 0.6, pitch = 0.45, radius = 45, heliA = 0, tvIdx = -1, lastFollow = null, smoothSpeed = 0;
  const fp = { x: 0, y: 0, z: 0, tx: 0, tz: 1, pitch: 0, lat: 0 };   // followed car pose

  function applyFov() { if (Math.abs(camera.fov - fov) > 0.01) { camera.fov = fov; camera.updateProjectionMatrix(); } }
  function setCamera(dtMs, a) {
    const k = (tau) => (snap ? 1 : 1 - Math.exp(-dtMs / tau));
    let wantFov = 60;
    if (!a || a.rp == null) { // nobody to follow: slow overview orbit
      heliA += dtMs * 0.00008;
      const r = G.size * 0.7;
      wantPos.set(G.cx + Math.sin(heliA) * r, G.maxY + r * 0.55, G.cz + Math.cos(heliA) * r);
      wantLook.set(G.cx, G.minY, G.cz); wantFov = 50;
      camPos.lerp(wantPos, k(300)); camLook.lerp(wantLook, k(300)); fov += (wantFov - fov) * k(300); applyFov();
      return;
    }
    const mode = prefs.mode, sf = Math.min(1, smoothSpeed / 330), Lm = G.L;
    if (mode === 'chase') {
      const dist = 8.5 + sf * 5, hgt = 3.1 + sf * 0.8, fb = a.rp - dist / Lm;
      const back = pose(fb), bi = idxOf(fb);
      wantPos.set(back.x + G.TZ[bi] * fp.lat * 0.7, back.y + hgt, back.z - G.TX[bi] * fp.lat * 0.7);
      const ahead = pose(a.rp + 14 / Lm);
      wantLook.set(ahead.x, ahead.y + 0.8, ahead.z);
      wantFov = 62 + sf * 18;
      camPos.lerp(wantPos, k(90)); camLook.lerp(wantLook, k(60));
    } else if (mode === 'onboard') {
      const fn = a.rp + 2.2 / Lm, nose = pose(fn), bi = idxOf(fn);
      wantPos.set(nose.x + G.TZ[bi] * fp.lat, nose.y + 1.05, nose.z - G.TX[bi] * fp.lat);
      const ahead = pose(a.rp + 40 / Lm);
      wantLook.set(ahead.x + G.TZ[bi] * fp.lat * 0.7, ahead.y + 0.9, ahead.z - G.TX[bi] * fp.lat * 0.7);
      wantFov = 82 + sf * 12;
      camPos.lerp(wantPos, k(45)); camLook.lerp(wantLook, k(60));
    } else if (mode === 'tv') {
      let best = -1, bd = 1e9;
      for (let i = 0; i < tvCams.length; i++) {
        let d = tvCams[i].f - a.rp; d -= Math.round(d); d *= Lm;
        if (d >= -45 && d < bd) { bd = d; best = i; }
      }
      if (best < 0) best = 0;
      if (best !== tvIdx) { tvIdx = best; camLook.set(fp.x, fp.y + 1, fp.z); }
      const c = tvCams[tvIdx];
      camPos.set(c.x, c.y, c.z);
      wantLook.set(fp.x, fp.y + 0.8, fp.z); camLook.lerp(wantLook, k(60));
      const dist = Math.max(8, Math.hypot(c.x - fp.x, c.y - fp.y, c.z - fp.z));
      wantFov = Math.max(7, Math.min(55, 2 * Math.atan(16 / dist) / DEG));
      fov += (wantFov - fov) * k(140); applyFov(); return;
    } else if (mode === 'heli') {
      heliA += dtMs * 0.00018;
      wantPos.set(fp.x + Math.sin(heliA) * 95, fp.y + 75, fp.z + Math.cos(heliA) * 95);
      wantLook.set(fp.x, fp.y, fp.z); wantFov = 48;
      camPos.lerp(wantPos, k(220)); camLook.lerp(wantLook, k(120));
    } else {
      const cp = Math.cos(pitch);
      wantLook.set(fp.x, fp.y + 1, fp.z);
      wantPos.set(fp.x + Math.sin(yaw) * cp * radius, fp.y + 1 + Math.sin(pitch) * radius, fp.z + Math.cos(yaw) * cp * radius);
      wantFov = 55;
      camPos.lerp(wantPos, k(60)); camLook.lerp(wantLook, k(60));
    }
    camPos.y = Math.max(camPos.y, G.groundY + 1.0);
    fov += (wantFov - fov) * k(250);
    applyFov();
  }

  // ---- free orbit input (mouse + touch)
  const ptrs = new Map(); let pinch0 = 0, radius0 = 0;
  cv.addEventListener('pointerdown', (e) => {
    if (prefs.mode !== 'free') return;
    cv.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (ptrs.size === 2) { const [p, r] = [...ptrs.values()]; pinch0 = Math.hypot(p.x - r.x, p.y - r.y); radius0 = radius; }
    cv.style.cursor = 'grabbing';
  });
  cv.addEventListener('pointermove', (e) => {
    const t = ptrs.get(e.pointerId); if (!t || prefs.mode !== 'free') return;
    if (ptrs.size === 1) { yaw -= (e.clientX - t.x) * 0.006; pitch = Math.max(0.03, Math.min(1.5, pitch + (e.clientY - t.y) * 0.005)); }
    t.x = e.clientX; t.y = e.clientY;
    if (ptrs.size === 2) { const [p, r] = [...ptrs.values()]; const d = Math.hypot(p.x - r.x, p.y - r.y); if (pinch0 > 0 && d > 0) radius = Math.max(8, Math.min(1200, radius0 * pinch0 / d)); }
  });
  const endPtr = (e) => { ptrs.delete(e.pointerId); if (!ptrs.size) cv.style.cursor = prefs.mode === 'free' ? 'grab' : ''; };
  cv.addEventListener('pointerup', endPtr); cv.addEventListener('pointercancel', endPtr);
  cv.addEventListener('wheel', (e) => {
    if (prefs.mode !== 'free') return;
    e.preventDefault(); radius = Math.max(8, Math.min(1200, radius * Math.exp(e.deltaY * 0.001)));
  }, { passive: false });

  function visible() { return el.isConnected && el.offsetParent !== null && el.offsetWidth > 0 && el.offsetHeight > 0 && !document.hidden; }
  const onKey = (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || !visible()) return;
    const t = e.target; if (t && /INPUT|TEXTAREA|SELECT/.test(t.tagName)) return;
    const key = e.key.toLowerCase(), m = { c: 'chase', o: 'onboard', t: 'tv', h: 'heli', r: 'free' }[key];
    if (m) setMode(m); else if (key === 'i') setHud(!prefs.hud);
  };
  document.addEventListener('keydown', onKey);

  // ------------------------------------------------------------------ HUD
  let hudT = 0, listSig = '';
  const setT = (node, v) => { if (node._t !== v) { node._t = v; node.textContent = v; } };
  const fmtGap = (s) => (s == null || s === '' ? '-' : String(s));
  function updateHud(num) {
    const st = store.state; if (!st || !num) return;
    const c = st.cars[num]; if (!c) return;
    const f = store.fast[num] || {}, tel = c.tel || {};
    const pick = (k) => (f[k] != null ? f[k] : tel[k]);
    const speed = pick('speed'), rpm = pick('rpm'), gear = pick('gear'), thr = pick('throttle'), brk = pick('brake'), drs = pick('drs');
    E.sw.style.background = c.color || '#888';
    setT(E.tla, c.tla || num); setT(E.name, c.name || ''); setT(E.pos, 'P' + (c.pos || '-'));
    setT(E.speed, speed != null ? String(Math.round(speed)) : '0');
    setT(E.gear, !gear ? 'N' : String(gear));
    E.drs.classList.toggle('on', !!drs);
    E.thr.style.width = Math.max(0, Math.min(100, thr || 0)) + '%';
    E.brk.style.width = Math.max(0, Math.min(100, brk || 0)) + '%';
    const lit = rpm == null ? 0 : Math.round(Math.max(0, Math.min(1, (rpm - 8800) / 3200)) * 10);
    const flash = lit >= 10 && (performance.now() % 200) < 100;
    for (let i = 0; i < 10; i++) leds[i].classList.toggle('on', i < lit && !flash);
    const order = st.order || [], idx = order.indexOf(num);
    const behind = idx >= 0 && order[idx + 1] ? st.cars[order[idx + 1]] : null;
    setT(E.ahead, c.pos === 1 ? 'LEADER' : fmtGap(c.interval));
    setT(E.behind, behind ? fmtGap(behind.interval) : '-');
    setT(E.gapl, c.pos === 1 ? '-' : fmtGap(c.gapLeader));
    setT(E.tyre, c.tyre && c.tyre.compound ? `${c.tyre.compound[0]} ${c.tyre.age != null ? c.tyre.age + 'L' : ''}`.trim() : '-');
    setT(E.lap, c.lap != null ? c.lap + (st.lap && st.lap.total ? '/' + st.lap.total : '') : '-');
    setT(E.last, c.last && c.last.text ? c.last.text : '-');
    const want = new Set(order.slice(0, 10));
    if (idx >= 0) for (let d = -2; d <= 2; d++) if (order[idx + d]) want.add(order[idx + d]);
    const rows = order.filter((n) => want.has(n));
    const sig = rows.map((n) => { const cc = st.cars[n]; return n + (cc.interval || '') + (cc.retired ? 'x' : '') + (cc.inPit ? 'p' : ''); }).join(',') + '|' + num;
    if (sig === listSig) return; listSig = sig;
    E.list.textContent = '';
    let prev = -2;
    for (const n of rows) {
      const cc = st.cars[n], oi = order.indexOf(n);
      if (prev >= 0 && oi > prev + 1) { const gap = document.createElement('div'); gap.className = 'cc-gap'; gap.textContent = '...'; E.list.append(gap); }
      prev = oi;
      const r = document.createElement('div'); r.className = 'cc-row' + (n === num ? ' sel' : '') + (cc.retired ? ' out' : '') + (cc.inPit ? ' pit' : '');
      r.innerHTML = '<span class="p"></span><i></i><b></b><span class="g"></span>';
      r.children[0].textContent = cc.pos || oi + 1; r.children[1].style.background = cc.color || '#888';
      r.children[2].textContent = cc.tla || n; r.children[3].textContent = cc.retired ? 'OUT' : cc.inPit ? 'PIT' : (cc.pos === 1 ? 'Leader' : fmtGap(cc.interval));
      r.onclick = () => { if (n !== [...store.selected][0] || store.selected.size !== 1) { store.selected = new Set(); store.select(n); } };
      E.list.append(r);
    }
  }

  // ------------------------------------------------------------------ frame loop
  const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _e = new THREE.Euler(0, 0, 0, 'YXZ'), _p = new THREE.Vector3(), _s = new THREE.Vector3();
  let last = performance.now(), raf = 0, disposed = false, ctxLost = false;

  function resize() {
    const w = el.clientWidth, h = el.clientHeight;
    if (w === W && h === H) return;
    W = w; H = h;
    if (renderer) { renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2)); renderer.setSize(w, h, false); }
    camera.aspect = w / Math.max(1, h); camera.updateProjectionMatrix();
  }

  function frame(now) {
    if (disposed) return;
    raf = requestAnimationFrame(frame);
    const dt = Math.min(100, now - last); last = now;
    if (!visible()) return;
    resize();
    const num = followed();
    if (num !== lastFollow) { lastFollow = num; tvIdx = -1; listSig = ''; }
    if (!G && store.track) buildTrack();
    if (!G) msg.hidden = false;
    if (store.state) syncCars();

    if (G && carMesh) {
      const cars = (store.state && store.state.cars) || {};
      const alpha = 1 - Math.exp(-dt / 90);
      const tanHalf = Math.tan(camera.fov * DEG / 2);
      for (let s = 0; s < carNums.length; s++) {
        const n = carNums[s], a = anim[n], c = cars[n], lab = labels[s];
        if (!a || (a.rp == null && a.x == null) || (c && c.retired && !store.fast[n])) {
          _m.makeScale(0, 0, 0); carMesh.setMatrixAt(s, _m); if (lab) lab.visible = false; continue;
        }
        const ext = Math.min(now - a.t, 500);
        if (a.p != null) {
          const tgt = a.p + a.v * ext; let d = tgt - a.rp; d -= Math.round(d);
          if (Math.abs(d) > 0.2) a.rp = tgt; else a.rp += d * alpha;
          a.rp -= Math.floor(a.rp);
        } else if (a.x != null) {
          a.rp = nearestFrac((a.x * G.cosA - a.y * G.sinA) * G.rawScale, -(a.x * G.sinA + a.y * G.cosA) * G.rawScale);
        }
        pose(a.rp);
        let latT = 0;
        if (a.x != null) {
          const rxm = (a.x * G.cosA - a.y * G.sinA) * G.rawScale, rzm = -(a.x * G.sinA + a.y * G.cosA) * G.rawScale;
          latT = (rxm - P.x) * P.tz + (rzm - P.z) * -P.tx;
          latT = isFinite(latT) ? Math.max(-5, Math.min(5, latT)) : 0;
        }
        a.lat += (latT - a.lat) * Math.min(1, dt / 400);
        const x = P.x + P.tz * a.lat, z = P.z - P.tx * a.lat, y = P.y;
        const dd = Math.hypot(x - camPos.x, y - camPos.y, z - camPos.z);
        const lenPx = 5.6 / Math.max(0.1, dd * 2 * tanHalf) * Math.max(1, H);
        let sc = lenPx < 14 ? Math.min(14, 14 / lenPx) : 1;
        if (prefs.mode === 'onboard' && n === num) sc = 0.0001;
        _e.set(-P.pitch, Math.atan2(P.tx, P.tz), 0);
        _q.setFromEuler(_e); _p.set(x, y + 0.03, z); _s.set(sc, sc, sc);
        _m.compose(_p, _q, _s); carMesh.setMatrixAt(s, _m);
        if (n === num) { fp.x = x; fp.y = y; fp.z = z; fp.tx = P.tx; fp.tz = P.tz; fp.pitch = P.pitch; fp.lat = a.lat; }
        if (lab) {
          const show = !(n === num && (prefs.mode === 'chase' || prefs.mode === 'onboard')) && dd < 2200;
          lab.visible = show;
          if (show) {
            const hpx = n === num ? 20 : 15, wh = Math.max(1.2, dd * 2 * tanHalf * hpx / Math.max(1, H));
            lab.scale.set(wh * 2.667, wh, 1);
            lab.position.set(x, y + 2.2 + wh * 0.6 + (sc > 1 ? sc * 0.5 : 0), z);
          }
        }
      }
      carMesh.instanceMatrix.needsUpdate = true;
      const fs = num && ((store.fast[num]) || (store.state && store.state.cars[num] && store.state.cars[num].tel));
      smoothSpeed += ((fs && fs.speed != null ? fs.speed : 0) - smoothSpeed) * Math.min(1, dt / 400);
      setCamera(dt, num ? anim[num] : null);
      camera.position.copy(camPos); camera.lookAt(camLook);
      snap = false;
      sky.position.copy(camPos);
    }

    if (now - hudT > 100) { hudT = now; if (prefs.hud) updateHud(num); }
    if (renderer && !ctxLost) { try { renderer.render(scene, camera); } catch (e) { console.error('chasecam render', e); } }
  }

  // ------------------------------------------------------------------ wiring
  cv.addEventListener('webglcontextlost', (e) => { e.preventDefault(); ctxLost = true; msg.hidden = false; msg.textContent = 'WebGL context lost - waiting to restore...'; });
  cv.addEventListener('webglcontextrestored', () => { ctxLost = false; msg.hidden = true; });
  const unsub = store.subscribe((k) => {
    if (k === 'track') { try { buildTrack(); } catch (e) { console.error('chasecam track', e); } }
    else if (k === 'fast') ingestAll(false);
    else if (k === 'state') ingestAll(true);
    else if (k === 'selection') listSig = '';
  });
  if (!renderer) { msg.hidden = false; msg.textContent = glFail; msg.classList.add('err'); cv.hidden = true; }

  setMode(prefs.mode); setHud(prefs.hud);
  ingestAll(true);
  if (store.track) { try { buildTrack(); } catch (e) { console.error('chasecam track', e); } }
  raf = requestAnimationFrame(frame);

  function dispose() {
    disposed = true; cancelAnimationFrame(raf); unsub();
    document.removeEventListener('keydown', onKey);
    disposeGroup(trackGroup);
    if (carMesh) { carMesh.geometry.dispose(); carMesh.material.dispose(); carMesh.dispose(); }
    for (const l of labels) if (l) { l.material.map.dispose(); l.material.dispose(); }
    sky.geometry.dispose(); sky.material.dispose();
    if (renderer) { renderer.dispose(); if (renderer.forceContextLoss) renderer.forceContextLoss(); }
    el.remove();
  }
  host._dispose = dispose;
  return dispose;
}
