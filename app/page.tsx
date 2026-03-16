"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import CockpitView from "@/components/CockpitView";
import Dashboard from "@/components/Dashboard";
import type {
  Session,
  Driver,
  Position,
  CarData,
  Stint,
  RaceControlMessage,
} from "@/lib/types";
import {
  getAllRaceSessions,
  getDrivers,
  getLatestPositions,
  getAllLatestCarData,
  getStints,
  getLatestStintPerDriver,
  getRaceControlMessages,
} from "@/lib/openf1";

type Tab = "cockpit" | "dashboard";

export default function Home() {
  const [tab, setTab] = useState<Tab>("cockpit");

  // ── Session data ──
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedSession, setSelectedSession] = useState<Session | null>(null);
  const [sessionsLoading, setSessionsLoading] = useState(true);

  // ── Race data ──
  const [drivers, setDrivers] = useState<Driver[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [allCarData, setAllCarData] = useState<Map<number, CarData>>(new Map());
  const [stints, setStints] = useState<Map<number, Stint>>(new Map());
  const [raceMessages, setRaceMessages] = useState<RaceControlMessage[]>([]);
  const [selectedDriverNum, setSelectedDriverNum] = useState<number | null>(null);
  const [dataLoading, setDataLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Load available sessions on mount ──
  useEffect(() => {
    (async () => {
      setSessionsLoading(true);
      try {
        const all = await getAllRaceSessions();
        setSessions(all);
        // Auto-select the most recent past race
        const now = Date.now();
        const pastRace = all.find(
          (s) => new Date(s.date_start).getTime() <= now
        );
        if (pastRace) setSelectedSession(pastRace);
      } catch (e) {
        setError("Failed to load sessions: " + String(e));
      } finally {
        setSessionsLoading(false);
      }
    })();
  }, []);

  // ── Load session data when a session is selected ──
  const loadSessionData = useCallback(async (session: Session) => {
    setDataLoading(true);
    setError(null);
    try {
      const [drvs, pos, stintData, msgs] = await Promise.all([
        getDrivers(session.session_key),
        getLatestPositions(session.session_key),
        getStints(session.session_key),
        getRaceControlMessages(session.session_key),
      ]);

      const carDataMap = await getAllLatestCarData(session.session_key);
      const stintMap = getLatestStintPerDriver(stintData);

      setDrivers(drvs);
      setPositions(pos);
      setAllCarData(carDataMap);
      setStints(stintMap);
      setRaceMessages(msgs);
      setLastUpdated(new Date());

      // Auto-select P1 driver
      if (pos.length > 0 && selectedDriverNum === null) {
        setSelectedDriverNum(pos[0].driver_number);
      }
    } catch (e) {
      setError("Failed to load race data: " + String(e));
    } finally {
      setDataLoading(false);
    }
  }, [selectedDriverNum]);

  useEffect(() => {
    if (!selectedSession) return;
    loadSessionData(selectedSession);
  }, [selectedSession]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Polling for live updates ──
  const [polling, setPolling] = useState(false);

  useEffect(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    if (polling && selectedSession) {
      pollingRef.current = setInterval(() => {
        loadSessionData(selectedSession);
      }, 10_000);
    }
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, [polling, selectedSession, loadSessionData]);

  // ── Selected driver car data ──
  const selectedCarData = selectedDriverNum
    ? (allCarData.get(selectedDriverNum) ?? null)
    : null;

  // ── Render ──
  return (
    <div className="min-h-screen flex flex-col" style={{ background: "#0f0f0f" }}>
      {/* ══ HEADER ══ */}
      <header
        className="flex items-center justify-between px-4 py-2 border-b"
        style={{ background: "#111", borderColor: "#e10600" }}
      >
        <div className="flex items-center gap-3">
          <span className="text-2xl">🏎️</span>
          <span
            className="font-black text-base tracking-[0.2em] uppercase hidden sm:block"
            style={{ color: "#e10600" }}
          >
            F1 Dashboard
          </span>
        </div>

        {/* Tab switcher */}
        <div className="flex gap-1 bg-black/40 p-1 rounded-lg">
          {(["cockpit", "dashboard"] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className="px-4 py-1.5 rounded-md text-xs font-bold uppercase tracking-wider transition-all"
              style={{
                background: tab === t ? "#e10600" : "transparent",
                color: tab === t ? "#fff" : "#888",
              }}
            >
              {t === "cockpit" ? "🏎️ Cockpit" : "📊 Dashboard"}
            </button>
          ))}
        </div>

        {/* Status */}
        <div className="flex items-center gap-2 text-xs text-gray-500">
          {lastUpdated && (
            <span className="hidden md:block">
              {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={() => setPolling((p) => !p)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded border transition-all"
            style={{
              borderColor: polling ? "#00e676" : "#333",
              color: polling ? "#00e676" : "#666",
              background: polling ? "rgba(0,230,118,0.08)" : "transparent",
            }}
          >
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{ background: polling ? "#00e676" : "#444" }}
            />
            {polling ? "LIVE" : "PAUSED"}
          </button>
        </div>
      </header>

      {/* ══ SESSION SELECTOR BAR ══ */}
      <div
        className="flex flex-wrap items-center gap-3 px-4 py-2 border-b"
        style={{ background: "#0d0d0d", borderColor: "#222" }}
      >
        {sessionsLoading ? (
          <span className="text-gray-500 text-xs animate-pulse">
            Loading races…
          </span>
        ) : (
          <>
            <label className="text-gray-500 text-xs font-bold uppercase tracking-wider">
              Race
            </label>
            <select
              value={selectedSession?.session_key ?? ""}
              onChange={(e) => {
                const s = sessions.find(
                  (s) => s.session_key === Number(e.target.value)
                );
                if (s) setSelectedSession(s);
              }}
              className="bg-gray-900 border border-gray-700 text-white text-xs rounded px-2 py-1.5 cursor-pointer max-w-xs"
            >
              {sessions.map((s) => (
                <option key={s.session_key} value={s.session_key}>
                  {s.year} — {s.country_name} ({s.circuit_short_name}) ·{" "}
                  {new Date(s.date_start).toLocaleDateString()}
                </option>
              ))}
            </select>

            {selectedSession && (
              <div className="flex gap-2 text-xs">
                <span className="bg-gray-800 text-gray-300 px-2 py-1 rounded">
                  {selectedSession.year}
                </span>
                <span className="bg-gray-800 text-gray-300 px-2 py-1 rounded">
                  {selectedSession.circuit_short_name}
                </span>
              </div>
            )}

            {dataLoading && (
              <span className="text-gray-500 text-xs animate-pulse">
                Loading data…
              </span>
            )}

            {selectedSession && !dataLoading && (
              <button
                onClick={() => loadSessionData(selectedSession)}
                className="text-xs text-gray-500 hover:text-white border border-gray-700 hover:border-gray-500 px-2 py-1 rounded transition-colors"
              >
                ↺ Refresh
              </button>
            )}
          </>
        )}
      </div>

      {/* ══ ERROR BANNER ══ */}
      {error && (
        <div className="mx-4 mt-3 px-4 py-2 rounded border border-red-800 bg-red-900/20 text-red-400 text-xs">
          ⚠️ {error}
        </div>
      )}

      {/* ══ MAIN CONTENT ══ */}
      <main className="flex-1 p-4">
        {!selectedSession && !sessionsLoading && (
          <div className="flex items-center justify-center h-64 text-gray-500">
            No sessions available
          </div>
        )}

        {selectedSession && (
          <>
            {tab === "cockpit" && (
              <div className="max-w-5xl mx-auto">
                {drivers.length === 0 && !dataLoading ? (
                  <div className="text-center text-gray-500 py-16">
                    No driver data — select a session above
                  </div>
                ) : (
                  <CockpitView
                    drivers={drivers}
                    positions={positions}
                    selectedDriverNumber={
                      selectedDriverNum ?? positions[0]?.driver_number ?? 1
                    }
                    onSelectDriver={setSelectedDriverNum}
                    carData={selectedCarData}
                    stints={stints}
                    sessionName={`${selectedSession.year} ${selectedSession.session_name}`}
                    circuitName={selectedSession.circuit_short_name}
                    countryName={selectedSession.country_name}
                  />
                )}
              </div>
            )}

            {tab === "dashboard" && (
              <div className="max-w-4xl mx-auto">
                {/* Session header */}
                <div className="mb-4 p-4 rounded-xl border border-gray-800 bg-gray-900/50">
                  <div className="text-f1-red font-black text-lg tracking-widest uppercase">
                    {selectedSession.country_name}
                  </div>
                  <div className="text-gray-400 text-sm mt-0.5">
                    {selectedSession.circuit_short_name} ·{" "}
                    {selectedSession.year} {selectedSession.session_name}
                    {" · "}
                    {new Date(selectedSession.date_start).toLocaleDateString(
                      undefined,
                      { weekday: "long", year: "numeric", month: "long", day: "numeric" }
                    )}
                  </div>
                </div>

                <Dashboard
                  drivers={drivers}
                  positions={positions}
                  allCarData={allCarData}
                  stints={stints}
                  raceMessages={raceMessages}
                  selectedDriverNumber={
                    selectedDriverNum ?? positions[0]?.driver_number ?? 1
                  }
                  onSelectDriver={setSelectedDriverNum}
                />
              </div>
            )}
          </>
        )}
      </main>

      {/* ══ FOOTER ══ */}
      <footer className="text-center text-gray-700 text-xs py-3 border-t border-gray-800">
        Powered by{" "}
        <a
          href="https://openf1.org"
          target="_blank"
          rel="noopener noreferrer"
          className="text-gray-500 hover:text-white transition-colors"
        >
          OpenF1 API
        </a>{" "}
        · Not affiliated with Formula 1
      </footer>
    </div>
  );
}
