import { setText } from '../util.js';

// Connection banner: 'Disconnected — retrying in Ns' / 'Connecting…'. Staleness is shown in the header.
export function mount(el, store) {
  const b = document.createElement('div'); b.id = 'conn-banner'; b.hidden = true; b.setAttribute('role', 'status');
  document.body.append(b);
  let since = performance.now();
  function render() {
    const c = store.conn;
    if (c === 'open') {
      since = performance.now();
      if (store.phase === 'error' && store.statusErr) { b.hidden = false; b.className = ''; setText(b, 'Feed error: ' + store.statusErr + ' (retrying automatically)'); }
      else b.hidden = true;
      return;
    }
    const showAfter = c === 'connecting' && !store.hello ? 1500 : 0;
    if (performance.now() - since < showAfter) { b.hidden = true; return; }
    b.hidden = false;
    b.className = c === 'closed' ? '' : 'connecting';
    if (c === 'closed') {
      const left = Math.max(0, Math.ceil((store.retry.nextAt - performance.now()) / 1000));
      setText(b, 'Disconnected — retrying' + (left ? ' in ' + left + ' s' : ' now…') + (store.retry.attempt > 1 ? ' (attempt ' + store.retry.attempt + ')' : ''));
    } else setText(b, 'Connecting…');
  }
  store.subscribe((k) => { if (k === 'conn') render(); });
  setInterval(() => { try { render(); } catch {} }, 500);
  render();
}
