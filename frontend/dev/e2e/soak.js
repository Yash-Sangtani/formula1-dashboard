// Soak: ~3h worth of frames (state 1Hz = 10800, fast 4Hz = 43200, history/lap) pushed through the store with all panels visible.
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--no-sandbox', '--js-flags=--expose-gc'] });
  const p = await (await b.newContext({ viewport: { width: 1600, height: 900 } })).newPage();
  const errs = []; p.on('console', m => { if (m.type()==='error' && !/GL Driver/.test(m.text())) errs.push(m.text().slice(0,200)); }); p.on('pageerror', e => errs.push(e.message));
  await p.goto('http://localhost:8150/index.html?mock=quali&warm=60'); await p.waitForTimeout(1500);
  await p.keyboard.press('4');
  await p.evaluate(async () => { const { store } = await import('/js/store.js'); const f = document.querySelector('.tw-body .row'); f.click(); store.select(Object.keys(store.state.cars)[3], { additive: true }); });
  const measure = () => p.evaluate(async () => { const { store } = await import('/js/store.js'); if (window.gc) window.gc(); await new Promise(r => setTimeout(r, 300)); if (window.gc) window.gc();
    return { heapMB: +(performance.memory.usedJSHeapSize / 1048576).toFixed(1), nodes: document.querySelectorAll('*').length, fast: Object.keys(store.fast).length, rc: store.state.raceControl.length }; });
  console.log('start', await measure());
  for (let round = 0; round < 6; round++) {
    await p.evaluate(async () => {
      const { store } = await import('/js/store.js');
      const sleep = (ms) => new Promise(r => setTimeout(r, ms));
      for (let i = 0; i < 1800; i++) { // 1800 state frames + 7200 fast per round (x6 = 3h)
        const base = JSON.parse(JSON.stringify(store.state)); base.utc += i + 1;
        base.raceControl = Array.from({ length: 100 }, (_, k) => ({ utc: base.utc - k, lap: k, category: 'Other', message: 'm' + i + '_' + k }));
        store.ingest(base);
        for (let k = 0; k < 4; k++) store.ingest({ type: 'fast', utc: base.utc, cars: Object.fromEntries(Object.keys(base.cars).map(n => [n, { x: Math.random() * 1000, y: Math.random() * 1000, speed: 200, progress: Math.random(), throttle: 50, brake: 0 }])) });
        if (i % 20 === 0) await sleep(5);
      }
    });
    console.log('round', round, await measure());
  }
  console.log('errors', errs.slice(0, 5));
  await b.close();
})();
