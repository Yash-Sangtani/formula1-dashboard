// Fuzz: feed malformed frames to the store while panels are mounted; assert no console errors / page errors / panel-error badges.
const { chromium } = require('playwright');
const BASE = process.argv[2] || 'http://localhost:8150';
const kinds = (process.argv[3] || 'race,fp,quali').split(',');
const ITER = +(process.argv[4] || 250);
(async () => {
  const b = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--no-sandbox'] });
  let bad = 0;
  for (const kind of kinds) for (const [vw, vh] of [[1600, 900], [390, 844]]) {
    const p = await (await b.newContext({ viewport: { width: vw, height: vh } })).newPage();
    const errs = [];
    p.on('console', (m) => { if (m.type() === 'error' && !/GL Driver|Failed to load resource/.test(m.text())) errs.push('console: ' + m.text().slice(0, 300)); });
    p.on('pageerror', (e) => errs.push('pageerror: ' + e.message));
    await p.goto(`${BASE}/index.html?mock=${kind}&warm=60`);
    await p.waitForTimeout(1500);
    const res = await p.evaluate(async (ITER) => {
      const { store } = await import('/js/store.js');
      const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
      const JUNK = [null, undefined, NaN, Infinity, -Infinity, -1, 0, -0, 1e308, 'abc', '', '1:2:3', [], {}, [null], { a: { b: [1] } }, true, false, 'x'.repeat(5000), [[], [[]]], { length: 5 }];
      const rnd = (n) => Math.floor(Math.random() * n);
      const pick = (a) => a[rnd(a.length)];
      const clone = (o) => JSON.parse(JSON.stringify(o));
      function paths(o, pre = [], out = [], d = 0) {
        if (o && typeof o === 'object' && d < 6) for (const k of Object.keys(o)) { out.push(pre.concat(k)); paths(o[k], pre.concat(k), out, d + 1); }
        return out;
      }
      function mutate(o, n) {
        const ps = paths(o); if (!ps.length) return o;
        for (let i = 0; i < n; i++) {
          const pth = pick(ps); let t = o;
          for (let j = 0; j < pth.length - 1; j++) { if (t == null || typeof t !== 'object') { t = null; break; } t = t[pth[j]]; }
          if (t && typeof t === 'object') { const k = pth[pth.length - 1]; if (Math.random() < 0.25) delete t[k]; else t[k] = pick(JUNK); }
        }
        return o;
      }
      const baseState = clone(store.state), baseHist = clone(store.history || { type: 'history', laps: {}, events: [] }), baseTrack = clone(store.track);
      const sendWeird = [
        null, undefined, 5, 'str', [], { type: 'state' }, { type: 'state', cars: null }, { type: 'state', cars: [] }, { type: 'state', cars: { 1: null, 2: 'x', 3: {} } },
        { type: 'state', cars: { 1: { num: 1, last: 'a', best: 5, sectors: 'abc', tyre: null, tel: [], xy: 5, stints: 'x', bestLaps: 7, speedTraps: 3 } }, order: ['1', '9', 1, null] },
        { type: 'state', session: null, clock: 'x', lap: 4, trackStatus: 'red', weather: [], raceControl: { a: 1 }, radio: 'x', records: 5, pitStops: {} },
        { type: 'history' }, { type: 'history', laps: [], events: {} }, { type: 'history', laps: { 1: [null, 'x', { lap: 'a' }, { lap: 3, s: 'z' }] }, events: [null, 4] },
        { type: 'fast' }, { type: 'fast', cars: [] }, { type: 'fast', cars: { 1: null, 2: { x: 'a', speed: NaN } } }, { type: 'track', track: 5 }, { type: 'track', track: { points: [['a', 1], [1], null] } },
        { type: 'track', track: { points: [[0, 0], [1, 1], [2, 0]] } }, { type: 'control', speed: 'x', paused: 5 }, { type: 'mystery' },
      ];
      let sent = 0;
      for (const m of sendWeird) { store.ingest(m); sent++; await sleep(20); }
      // huge lists
      const big = clone(baseState);
      big.raceControl = Array.from({ length: 5000 }, (_, i) => ({ utc: i, lap: i % 60, category: 'Flag', flag: 'YELLOW', message: 'MSG ' + i }));
      big.radio = Array.from({ length: 800 }, (_, i) => ({ utc: i, num: String(1 + (i % 20)), tla: 'X', url: 'about:blank' }));
      for (let i = 0; i < 1000; i++) big.cars['9' + i] = { ...Object.values(big.cars)[0], num: '9' + i, tla: 'Z' + i };
      store.ingest(big); await sleep(100);
      const domAfterBig = document.querySelectorAll('*').length, rcNodes = document.querySelectorAll('.rc-item').length;
      store.ingest(baseState); await sleep(100);
      // random mutations with views cycling
      const views = ['1', '2', '3', '4', '5'];
      for (let i = 0; i < ITER; i++) {
        const r = Math.random();
        if (r < 0.5) store.ingest(mutate(clone(baseState), 1 + rnd(6)));
        else if (r < 0.7) store.ingest(Object.assign(mutate(clone(baseHist), 1 + rnd(8)), { type: 'history' }));
        else if (r < 0.8) store.ingest({ type: 'fast', utc: Date.now() / 1000, cars: mutate(clone(Object.fromEntries(Object.entries(store.fast).slice(0, 5))), 3) });
        else if (r < 0.85) store.ingest(mutate(clone({ type: 'track', track: baseTrack }), 2));
        else if (r < 0.9) store.ingest({ type: 'hello', mode: 'sim', replay: true, phase: pick(['waiting', 'active', null, 5]) });
        else if (r < 0.95) { const nums = Object.keys((store.state && store.state.cars) || {}); if (nums.length) store.select(pick(nums), { additive: Math.random() < 0.5 }); }
        else document.dispatchEvent(new KeyboardEvent('keydown', { key: pick(views) }));
        if (i % 7 === 0) { document.dispatchEvent(new KeyboardEvent('keydown', { key: pick(views) })); }
        await sleep(15);
      }
      // reconnect simulation: hello then fresh state
      store.ingest({ type: 'hello', mode: 'sim', replay: true, phase: 'active' }); await sleep(50);
      store.ingest(baseState); store.ingest(Object.assign(clone(baseHist), { type: 'history' })); await sleep(300);
      const badges = [...document.querySelectorAll('.perr')].map((e) => e.title);
      return { sent, domAfterBig, rcNodes, badges, fastKeys: Object.keys(store.fast).length };
    }, ITER);
    const toasts = await p.evaluate(() => document.querySelectorAll('.toast').length);
    const ok = !errs.length && !res.badges.length;
    if (!ok) bad++;
    console.log(`${ok ? 'PASS' : 'FAIL'} ${kind} ${vw}x${vh}`, JSON.stringify({ ...res, toasts }), errs.slice(0, 6));
    await p.context().close();
  }
  await b.close();
  process.exit(bad ? 1 : 0);
})();
