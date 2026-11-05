import { h, fmtUtc, ls } from '../util.js';

const FILTERS = [['all', 'All'], ['flag', 'Flags'], ['sc', 'SC'], ['drs', 'DRS'], ['other', 'Other']];

function kind(m) {
  const c = String(m.category || ''), f = String(m.flag || '').toUpperCase();
  if (/safety/i.test(c) || /SAFETY CAR|VSC|VIRTUAL/i.test(m.message || '')) return 'sc';
  if (c === 'Flag' || f) return 'flag';
  if (c === 'Drs') return 'drs';
  return 'other';
}
function flagCls(m) {
  const f = String(m.flag || '').toUpperCase();
  if (/CHEQ/.test(f)) return 'chq';
  if (/RED/.test(f)) return 'red';
  if (/YELLOW/.test(f)) return 'yellow';
  if (/BLUE/.test(f)) return 'blue';
  if (/BLACK/.test(f)) return 'black';
  if (/CLEAR|GREEN/.test(f)) return 'green';
  if (kind(m) === 'sc') return 'sc';
  if (kind(m) === 'drs') return 'drs';
  return 'other';
}

export function mount(el, store) {
  el.classList.add('rc');
  let filter = ls.get('f1.rcfilter', 'all');
  const bar = h('div', { class: 'chips' }, FILTERS.map(([k, t]) => h('button', { dataset: { f: k }, onclick: () => { filter = k; ls.set('f1.rcfilter', k); sync(); render(true); } }, t)));
  const list = h('div', { class: 'rc-list' });
  el.append(bar, list);
  const sync = () => [...bar.children].forEach((b) => b.classList.toggle('on', b.dataset.f === filter));
  sync();
  let sig = '';
  function render(force) {
    const s = store.state; if (!s) return;
    const msgs = s.raceControl || [];
    const nsig = filter + '|' + msgs.length + '|' + (msgs[0] ? msgs[0].utc + msgs[0].message : '');
    if (!force && nsig === sig) return; sig = nsig;
    list.textContent = '';
    const frag = document.createDocumentFragment();
    for (const m of msgs) {
      if (filter !== 'all' && kind(m) !== filter) continue;
      const car = m.num && s.cars && s.cars[m.num];
      frag.append(h('div', { class: 'rc-item f-' + flagCls(m) + (m.num && store.selected.has(String(m.num)) ? ' sel' : '') },
        h('span', { class: 'flagicon' }),
        h('div', { class: 'rc-main' },
          h('div', { class: 'rc-meta' }, m.lap != null ? h('span', { class: 'lap' }, 'L' + m.lap) : null, h('span', null, fmtUtc(m.utc)),
            m.sector ? h('span', null, 'S' + m.sector) : null,
            car ? h('span', { class: 'tla', style: { background: car.color } }, car.tla) : null),
          h('div', { class: 'rc-msg' }, m.message || ''))));
    }
    if (!frag.childNodes.length) frag.append(h('div', { class: 'empty' }, 'No messages'));
    list.append(frag);
  }
  store.subscribe((k) => { if (k === 'state') render(); else if (k === 'selection') render(true); });
  render();
}
