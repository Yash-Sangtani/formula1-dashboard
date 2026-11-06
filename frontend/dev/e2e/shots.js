const { chromium } = require('playwright');
const SP = process.argv[2], BASE = process.argv[3] || 'http://localhost:8150';
const kinds = (process.argv[4] || 'race,fp,quali,sprint_quali,sprint').split(',');
const views = (process.argv[5] || 'race').split(',');
(async () => {
  const b = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--no-sandbox'] });
  const errs = [];
  for (const [vw, vh, tag] of [[1600, 900, 'd'], [390, 844, 'm']]) {
    for (const kind of kinds) {
      const ctx = await b.newContext({ viewport: { width: vw, height: vh } });
      const p = await ctx.newPage();
      p.on('console', (m) => { if (m.type() === 'error' && !/GL Driver/.test(m.text())) errs.push(`[${kind}/${tag}] ${m.type()}: ${m.text()}`); });
      p.on('pageerror', (e) => errs.push(`[${kind}/${tag}] PAGEERROR ${e.message}`));
      p.on('request', (r) => { if (!r.url().startsWith(BASE) && !r.url().startsWith('data:')) errs.push(`[${kind}] EXTERNAL ${r.url()}`); });
      await p.goto(`${BASE}/index.html?mock=${kind}&warm=${process.env.WARM||8}`);
      await p.waitForTimeout(2500);
      for (const v of views) {
        if (v !== 'race') {
          await p.evaluate(() => { const o = (window.__order = document.querySelector('.tw-body .row')); });
          const first = await p.$('.tw-body .row'); if (first && (v === 'deep' || v === 'chase')) await first.click();
          await p.keyboard.press({ race: '1', deep: '2', analytics: '3', all: '4', chase: '5' }[v]);
          await p.waitForTimeout(1500);
        }
        await p.screenshot({ path: `${SP}/${kind}-${v}-${tag}.png` });
      }
      await ctx.close();
    }
  }
  console.log(errs.length ? errs.join('\n') : 'NO CONSOLE ERRORS');
  await b.close();
})();
