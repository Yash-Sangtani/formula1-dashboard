// Session-kind helpers shared by all panels. Lenient: works with or without the §6 fields.
const str = (v) => (v == null ? '' : String(v));

export function kindOf(state) {
  const se = (state && state.session) || {};
  const k = str(se.kind).toLowerCase();
  if (['practice', 'qualifying', 'sprint_qualifying', 'sprint', 'race'].includes(k)) return k;
  const t = str(se.type).toLowerCase(), n = str(se.name).toLowerCase();
  if (/shootout|sprint qualifying/.test(n)) return 'sprint_qualifying';
  if (/practice|^fp\d/.test(n) || /practice/.test(t)) return 'practice';
  if (/qualifying/.test(n) || /qualifying/.test(t)) return 'qualifying';
  if (/sprint/.test(n)) return 'sprint';
  if (/race/.test(n) || /race/.test(t)) return 'race';
  // unknown: a lap total means a race, otherwise infer from the presence of non-race fields
  if (state && state.lap && state.lap.total) return 'race';
  const cars = state && state.cars ? Object.values(state.cars) : [];
  if (cars.some((c) => c && (c.gapFastest || c.bestLaps || c.timedLaps != null))) return 'practice';
  return se.kind === 'unknown' || !se.type ? 'unknown' : 'race';
}

/** Describe the session for UI decisions. */
export function sessionOf(state) {
  const kind = kindOf(state);
  const se = (state && state.session) || {};
  const isRace = kind === 'race' || kind === 'sprint';
  const isQuali = kind === 'qualifying' || kind === 'sprint_qualifying';
  const isPractice = kind === 'practice';
  // 'unknown' behaves like a race (legacy UI) unless the feed is clearly timed-only
  const timed = isQuali || isPractice;
  let part = se.part == null ? null : +se.part;
  if (!(part >= 1 && part <= 3)) part = null;
  const partLabel = se.partLabel || (isQuali && part ? 'Q' + part : null);
  return { kind, isRace: !timed, isQuali, isPractice, timed, part, partLabel, name: se.name || se.type || '' };
}

export const KIND_LABEL = {
  practice: 'Practice', qualifying: 'Qualifying', sprint_qualifying: 'Sprint Qualifying', sprint: 'Sprint', race: 'Race', unknown: 'Session',
};

export function isWaiting(store) {
  return String(store.phase || '') === 'waiting';
}
