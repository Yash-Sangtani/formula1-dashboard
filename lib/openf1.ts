import type {
  Session,
  Meeting,
  Driver,
  Position,
  CarData,
  Location,
  Stint,
  RaceControlMessage,
} from "./types";

const BASE = "https://api.openf1.org/v1";

async function apiFetch<T>(path: string): Promise<T[]> {
  const res = await fetch(`${BASE}${path}`, { next: { revalidate: 10 } });
  if (!res.ok) throw new Error(`OpenF1 API error ${res.status}: ${path}`);
  return res.json();
}

// ── Sessions ─────────────────────────────────────────────────────────────────

export async function getLatestRaceSession(): Promise<Session | null> {
  const now = new Date().toISOString();
  const sessions = await apiFetch<Session>(
    `/sessions?session_type=Race&date_start%3C=${encodeURIComponent(now)}`
  );
  if (!sessions.length) return null;
  return sessions.sort(
    (a, b) =>
      new Date(b.date_start).getTime() - new Date(a.date_start).getTime()
  )[0];
}

export async function getSessionsByYear(year: number): Promise<Session[]> {
  const sessions = await apiFetch<Session>(`/sessions?year=${year}&session_type=Race`);
  return sessions.sort(
    (a, b) =>
      new Date(b.date_start).getTime() - new Date(a.date_start).getTime()
  );
}

export async function getAllRaceSessions(): Promise<Session[]> {
  // Fetch all race sessions from 2023 onwards (OpenF1 starts from 2023)
  const currentYear = new Date().getFullYear();
  const years = [];
  for (let y = 2023; y <= currentYear; y++) years.push(y);
  const results = await Promise.all(years.map((y) => getSessionsByYear(y)));
  const all = results.flat();
  return all.sort(
    (a, b) =>
      new Date(b.date_start).getTime() - new Date(a.date_start).getTime()
  );
}

// ── Meetings ──────────────────────────────────────────────────────────────────

export async function getMeetings(year: number): Promise<Meeting[]> {
  return apiFetch<Meeting>(`/meetings?year=${year}`);
}

// ── Drivers ──────────────────────────────────────────────────────────────────

export async function getDrivers(sessionKey: number): Promise<Driver[]> {
  return apiFetch<Driver>(`/drivers?session_key=${sessionKey}`);
}

// ── Positions ────────────────────────────────────────────────────────────────

export async function getLatestPositions(
  sessionKey: number
): Promise<Position[]> {
  // Get all positions for the session, then keep latest per driver
  const all = await apiFetch<Position>(`/position?session_key=${sessionKey}`);
  const byDriver = new Map<number, Position>();
  for (const p of all) {
    const existing = byDriver.get(p.driver_number);
    if (!existing || new Date(p.date) > new Date(existing.date)) {
      byDriver.set(p.driver_number, p);
    }
  }
  return Array.from(byDriver.values()).sort((a, b) => a.position - b.position);
}

// ── Car Data (telemetry) ──────────────────────────────────────────────────────

export async function getLatestCarData(
  sessionKey: number,
  driverNumber: number
): Promise<CarData | null> {
  const data = await apiFetch<CarData>(
    `/car_data?session_key=${sessionKey}&driver_number=${driverNumber}`
  );
  if (!data.length) return null;
  return data.sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
  )[0];
}

export async function getAllLatestCarData(
  sessionKey: number
): Promise<Map<number, CarData>> {
  const data = await apiFetch<CarData>(`/car_data?session_key=${sessionKey}`);
  const byDriver = new Map<number, CarData>();
  for (const d of data) {
    const existing = byDriver.get(d.driver_number);
    if (!existing || new Date(d.date) > new Date(existing.date)) {
      byDriver.set(d.driver_number, d);
    }
  }
  return byDriver;
}

// ── Location (GPS) ────────────────────────────────────────────────────────────

export async function getLatestLocations(
  sessionKey: number
): Promise<Map<number, Location>> {
  const data = await apiFetch<Location>(`/location?session_key=${sessionKey}`);
  const byDriver = new Map<number, Location>();
  for (const d of data) {
    const existing = byDriver.get(d.driver_number);
    if (!existing || new Date(d.date) > new Date(existing.date)) {
      byDriver.set(d.driver_number, d);
    }
  }
  return byDriver;
}

// ── Stints (tyre data) ────────────────────────────────────────────────────────

export async function getStints(sessionKey: number): Promise<Stint[]> {
  return apiFetch<Stint>(`/stints?session_key=${sessionKey}`);
}

export function getLatestStintPerDriver(stints: Stint[]): Map<number, Stint> {
  const byDriver = new Map<number, Stint>();
  for (const s of stints) {
    const existing = byDriver.get(s.driver_number);
    if (!existing || s.stint_number > existing.stint_number) {
      byDriver.set(s.driver_number, s);
    }
  }
  return byDriver;
}

// ── Race Control ──────────────────────────────────────────────────────────────

export async function getRaceControlMessages(
  sessionKey: number
): Promise<RaceControlMessage[]> {
  const msgs = await apiFetch<RaceControlMessage>(
    `/race_control?session_key=${sessionKey}`
  );
  return msgs.sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

export function formatGap(seconds: number | null): string {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 0.001) return "LEADER";
  return `+${seconds.toFixed(3)}s`;
}

export function teamColorHex(raw: string | null | undefined): string {
  if (!raw) return "#888888";
  const c = raw.replace(/^#/, "");
  if (/^[0-9a-fA-F]{6}$/.test(c)) return `#${c}`;
  return "#888888";
}

export function drsOpen(drsValue: number): boolean {
  return drsValue >= 10;
}

export function compoundColor(compound: string): string {
  switch (compound?.toUpperCase()) {
    case "SOFT": return "#e10600";
    case "MEDIUM": return "#ffd600";
    case "HARD": return "#f0f0f0";
    case "INTERMEDIATE": return "#39b54a";
    case "WET": return "#0067ff";
    default: return "#888888";
  }
}

export function compoundSymbol(compound: string): string {
  switch (compound?.toUpperCase()) {
    case "SOFT": return "S";
    case "MEDIUM": return "M";
    case "HARD": return "H";
    case "INTERMEDIATE": return "I";
    case "WET": return "W";
    default: return "?";
  }
}
