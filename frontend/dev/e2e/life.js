const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--no-sandbox'] });
  const p = await (await b.newContext({ viewport: { width: 1600, height: 900 } })).newPage();
  const errs = []; p.on('console', m => { if (m.type()==='error' && !/GL Driver/.test(m.text())) errs.push(m.text().slice(0,200)); }); p.on('pageerror', e => errs.push(e.message));
  await p.goto('http://localhost:8150/index.html?mock=quali&warm=60'); await p.waitForTimeout(1500);
  const res = await p.evaluate(async () => {
    const { store } = await import('/js/store.js'); const sleep = (ms) => new Promise(r => setTimeout(r, ms));
    const out = {}; const base = JSON.parse(JSON.stringify(store.state));
    const rows = () => document.querySelectorAll('.tw-body .row').length;
    const mk = (nums, ren) => { const s = JSON.parse(JSON.stringify(base)); const keep = {}; nums.forEach((n, i) => { const vals = Object.values(base.cars); const src = vals[i % vals.length]; keep[ren ? "9" + n : "" + n] = { ...src, num: ren ? "9" + n : "" + n, tla: ren ? "N" + n : src.tla + (i >= vals.length ? i : "") }; }); s.cars = keep; s.order = Object.keys(keep); return s; };
    store.ingest({ type: 'hello', mode: 'sim', replay: true, phase: 'live' });
    store.ingest(mk([1, 2, 3])); await sleep(100); out.three = rows();
    store.ingest(mk(Array.from({ length: 22 }, (_, i) => i + 1))); await sleep(100); out.full = rows();
    store.ingest(mk(Array.from({ length: 20 }, (_, i) => i + 1), true)); await sleep(100); out.renamed = rows();
    out.oldNumsGone = !document.querySelector('.tw-body .row[data-num="1"]');
    store.ingest(mk([])); await sleep(100); out.empty = rows();
    out.emptyMsgVisible = !document.querySelector('.tw-empty').hidden;
    // switch session kind live: quali -> race shape
    const race = JSON.parse(JSON.stringify(base)); race.session.kind = 'race'; race.session.type = 'Race'; race.session.name = 'Race'; race.lap = { current: 3, total: 50 };
    store.ingest(race); await sleep(100); out.mode = document.querySelector('.tower').dataset.mode; out.lapBoxHidden = document.querySelector('.hd-lap').hidden;
    store.ingest(base); await sleep(100); out.mode2 = document.querySelector('.tower').dataset.mode; out.lapBoxHidden2 = document.querySelector('.hd-lap').hidden;
    out.badges = document.querySelectorAll('.perr').length;
    return out;
  });
  console.log(JSON.stringify(res), errs);
  await b.close();
})();
