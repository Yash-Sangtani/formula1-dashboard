import { h, fmtUtc, ls } from '../util.js';

export function mount(el, store) {
  el.classList.add('radio');
  let onlySel = ls.get('f1.radioSel', false);
  const cb = h('input', { type: 'checkbox', id: 'radio-sel' });
  cb.checked = onlySel;
  cb.onchange = () => { onlySel = cb.checked; ls.set('f1.radioSel', onlySel); render(true); };
  const bar = h('div', { class: 'chips' }, h('label', { for: 'radio-sel' }, cb, ' Selected cars only'));
  const list = h('div', { class: 'radio-list' });
  el.append(bar, list);
  const audio = new Audio(); audio.preload = 'none';
  let playing = null; // url
  audio.addEventListener('ended', stop); audio.addEventListener('error', () => { if (playing) mark(playing, 'err'); stop(); });
  function mark(url, cls) { for (const b of list.querySelectorAll('.play')) if (b.dataset.url === url) b.classList.add(cls); }
  function stop() { playing = null; audio.pause(); for (const b of list.querySelectorAll('.play.on')) { b.classList.remove('on'); b.textContent = '▶'; } }
  function toggle(btn) {
    const url = btn.dataset.url;
    if (playing === url) { stop(); return; }
    stop(); playing = url; audio.src = url;
    audio.play().catch(() => { btn.classList.add('err'); stop(); });
    btn.classList.add('on'); btn.textContent = '■';
  }
  let sig = '';
  function render(force) {
    const s = store.state;
    if (!s) { if (sig !== 'none') { sig = 'none'; stop(); list.textContent = ''; list.append(h('div', { class: 'empty' }, 'No radio clips')); } return; }
    let items = (s.radio || []).slice(0, 60);
    if (onlySel && store.selected.size) items = items.filter((r) => store.selected.has(String(r.num)));
    const nsig = onlySel + '|' + items.length + '|' + (items[0] ? items[0].url : '');
    if (!force && nsig === sig) return; sig = nsig;
    const was = playing;
    list.textContent = '';
    for (const r of items) {
      const car = s.cars && s.cars[r.num];
      const btn = h('button', { class: 'play', dataset: { url: r.url }, title: 'Play clip', onclick: (e) => toggle(e.currentTarget) }, '▶');
      list.append(h('div', { class: 'radio-item' + (store.selected.has(String(r.num)) ? ' sel' : '') },
        btn,
        h('span', { class: 'tla', style: { background: car ? car.color : '#555' }, onclick: () => store.select(String(r.num)) }, r.tla || (car && car.tla) || r.num),
        h('span', { class: 'rtime' }, fmtUtc(r.utc)),
        h('span', { class: 'rname' }, car ? car.name : '')));
      if (was && r.url === was) { btn.classList.add('on'); btn.textContent = '■'; }
    }
    if (!items.length) list.append(h('div', { class: 'empty' }, 'No radio clips'));
  }
  store.subscribe((k) => { if (k === 'state' || k === 'reset') render(); else if (k === 'selection') render(true); });
  render();
}
