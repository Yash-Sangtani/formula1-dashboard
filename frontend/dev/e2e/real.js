const { chromium } = require('playwright');
const SP = process.argv[2];
const specs = [['8151','fp'],['8152','quali'],['8153','sprint_quali'],['8154','sprint'],['8155','race']];
(async () => {
  const b = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--no-sandbox'] });
  for (const [port, kind] of specs) for (const [vw, vh, tag] of [[1600, 900, 'd'], [390, 844, 'm']]) {
    const p = await (await b.newContext({ viewport: { width: vw, height: vh } })).newPage();
    const errs = [];
    p.on('console', m => { if (m.type()==='error' && !/GL Driver/.test(m.text())) errs.push(m.text().slice(0,300)); }); p.on('pageerror', e => errs.push('PAGEERR '+e.message));
    p.on('request', r => { if (!r.url().startsWith('http://localhost:'+port) && !r.url().startsWith('ws://localhost:'+port) && !r.url().startsWith('data:')) errs.push('EXT '+r.url()); });
    await p.goto(`http://localhost:${port}/index.html`); await p.waitForTimeout(6000);
    for (const [v, key] of [['race','1'],['deep','2'],['analytics','3'],['chase','5']]) {
      if (v === 'deep' || v === 'chase') { const f = await p.$('.tw-body .row'); if (f) await f.click(); }
      await p.keyboard.press(key); await p.waitForTimeout(1200);
      if (tag === 'd' || v === 'race') await p.screenshot({ path: `${SP}/real-${kind}-${v}-${tag}.png` });
    }
    const badges = await p.evaluate(() => [...document.querySelectorAll('.perr')].map(e => e.title));
    console.log(kind, tag, errs.length ? errs : 'ok', badges);
    await p.context().close();
  }
  await b.close();
})();
