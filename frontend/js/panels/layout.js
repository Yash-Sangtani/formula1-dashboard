import { h, ls } from '../util.js';
import { sessionOf } from '../session.js';

export const VIEWS = [['race', 'Race', '1'], ['deep', 'Deep dive', '2'], ['analytics', 'Analytics', '3'], ['all', 'All', '4'], ['chase', '3D', '5']];
const KEY = 'f1.layouts';

// ctl: {ui, set(patch), onChange(fn), reset()}
export function mount(el, store, ctl) {
  const tabs = h('div', { class: 'tabs', role: 'tablist' }, VIEWS.map(([k, t, n]) =>
    h('button', { role: 'tab', dataset: { v: k }, title: `${t} (${n})`, onclick: () => ctl.set({ view: k }) }, t)));
  const menuBtn = h('button', { class: 'lay-btn', title: 'Layouts' }, 'Layouts ▾');
  const menu = h('div', { class: 'lay-menu', hidden: true });
  const bigBtn = h('button', { title: 'Big-screen mode: everything at once, fullscreen (F)', onclick: () => ctl.bigScreen() }, '⛶');
  el.append(tabs, h('div', { class: 'lay-wrap' }, menuBtn, menu), bigBtn);

  function renderMenu() {
    const saved = ls.get(KEY, {});
    menu.textContent = '';
    const u = ctl.ui;
    menu.append(
      h('label', null, h('input', { type: 'checkbox', checked: u.tower === 'compact', onchange: (e) => ctl.set({ tower: e.target.checked ? 'compact' : 'auto' }) }), ' Compact tower'),
      h('label', null, h('input', { type: 'checkbox', checked: !u.side, onchange: (e) => ctl.set({ side: !e.target.checked }) }), ' Hide race control / radio'),
      h('hr'));
    const names = Object.keys(saved);
    if (!names.length) menu.append(h('div', { class: 'muted' }, 'No saved layouts'));
    for (const n of names) {
      menu.append(h('div', { class: 'lay-row' },
        h('button', { onclick: () => { ctl.set(saved[n]); menu.hidden = true; } }, n),
        h('button', { class: 'x', title: 'Delete', onclick: () => { delete saved[n]; ls.set(KEY, saved); renderMenu(); } }, '×')));
    }
    menu.append(h('hr'),
      h('button', { onclick: () => {
        const n = (prompt('Save current layout as:', 'Layout ' + (names.length + 1)) || '').trim();
        if (n) { saved[n] = { ...ctl.ui }; ls.set(KEY, saved); renderMenu(); }
      } }, 'Save current layout...'),
      h('button', { onclick: () => { ctl.reset(); menu.hidden = true; } }, 'Reset'));
  }
  menuBtn.onclick = (e) => { e.stopPropagation(); if (menu.hidden) renderMenu(); menu.hidden = !menu.hidden; };
  document.addEventListener('click', (e) => { if (!menu.contains(e.target)) menu.hidden = true; });
  const sync = () => {
    [...tabs.children].forEach((b) => { const on = b.dataset.v === ctl.ui.view; b.classList.toggle('on', on); b.setAttribute('aria-selected', on); });
  };
  store.subscribe((k) => { if (k === 'state') { const t = sessionOf(store.state).timed ? 'Timing' : 'Race'; if (tabs.firstChild.textContent !== t) tabs.firstChild.textContent = t; } });
  ctl.onChange(() => { sync(); if (!menu.hidden) renderMenu(); });
  sync();
}
