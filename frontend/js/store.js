// Shared client store: WebSocket connection + latest server messages + selection.
const listeners = new Set();
export const store = {
  state: null, fast: {}, track: null, history: null, hello: null,
  conn: 'connecting',
  selected: new Set(),
  subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); },
  emit(kind) { for (const fn of listeners) { try { fn(kind); } catch (e) { console.error(e); } } },
  select(num, { additive = false } = {}) {
    if (additive) { this.selected.has(num) ? this.selected.delete(num) : this.selected.add(num); }
    else if (this.selected.size === 1 && this.selected.has(num)) { this.selected.clear(); }
    else { this.selected = new Set([num]); }
    try { localStorage.setItem('f1.selected', JSON.stringify([...this.selected])); } catch {}
    this.emit('selection');
  },
  send(obj) { if (this._ws && this._ws.readyState === 1) this._ws.send(JSON.stringify(obj)); },
  _ws: null,
};
try { store.selected = new Set(JSON.parse(localStorage.getItem('f1.selected') || '[]')); } catch {}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  store._ws = ws; store.conn = 'connecting'; store.emit('conn');
  ws.onopen = () => { store.conn = 'open'; store.emit('conn'); };
  ws.onclose = () => { store.conn = 'closed'; store.emit('conn'); setTimeout(connect, 1500); };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    switch (m.type) {
      case 'hello': store.hello = m; store.emit('conn'); break;
      case 'state': store.state = m; store.emit('state'); break;
      case 'fast':
        for (const [k, v] of Object.entries(m.cars)) store.fast[k] = { ...(store.fast[k] || {}), ...v };
        store.fastUtc = m.utc; store.emit('fast'); break;
      case 'track': store.track = m.track; store.emit('track'); break;
      case 'history': store.history = m; store.emit('history'); break;
    }
  };
}
export function start() { connect(); }
