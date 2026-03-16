"use client";

import type { Driver, Position, CarData, Stint, RaceControlMessage } from "@/lib/types";
import {
  teamColorHex,
  compoundColor,
  compoundSymbol,
  formatGap,
  drsOpen,
} from "@/lib/openf1";

interface DashboardProps {
  drivers: Driver[];
  positions: Position[];
  allCarData: Map<number, CarData>;
  stints: Map<number, Stint>;
  raceMessages: RaceControlMessage[];
  selectedDriverNumber: number;
  onSelectDriver: (n: number) => void;
}

export default function Dashboard({
  drivers,
  positions,
  allCarData,
  stints,
  raceMessages,
  selectedDriverNumber,
  onSelectDriver,
}: DashboardProps) {
  return (
    <div className="flex flex-col gap-4">
      {/* ── Standings table ── */}
      <div
        className="rounded-xl overflow-hidden border border-gray-800"
        style={{ background: "#111" }}
      >
        <div className="px-4 py-2 border-b border-gray-800 flex items-center gap-2">
          <span className="text-f1-red text-sm font-bold tracking-widest uppercase">
            🏆 Standings
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-500 text-xs uppercase tracking-wider border-b border-gray-800">
                <th className="px-3 py-2 text-left w-8">P</th>
                <th className="px-3 py-2 text-left">Driver</th>
                <th className="px-3 py-2 text-left hidden md:table-cell">Team</th>
                <th className="px-3 py-2 text-center">Tyre</th>
                <th className="px-3 py-2 text-right hidden sm:table-cell">Speed</th>
                <th className="px-3 py-2 text-center hidden sm:table-cell">DRS</th>
                <th className="px-3 py-2 text-right">Gap</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((pos, idx) => {
                const drv = drivers.find(
                  (d) => d.driver_number === pos.driver_number
                );
                const cd = allCarData.get(pos.driver_number);
                const stint = stints.get(pos.driver_number);
                const isSelected = pos.driver_number === selectedDriverNumber;
                const isLeader = idx === 0;

                return (
                  <tr
                    key={pos.driver_number}
                    onClick={() => onSelectDriver(pos.driver_number)}
                    className="border-b border-gray-800/50 cursor-pointer transition-colors hover:bg-gray-800/40"
                    style={{
                      background: isSelected
                        ? "rgba(225,6,0,0.12)"
                        : "transparent",
                    }}
                  >
                    {/* Position */}
                    <td className="px-3 py-2.5">
                      <span
                        className="font-mono font-bold text-sm"
                        style={{
                          color: isLeader ? "#ffd600" : isSelected ? "#e10600" : "#aaa",
                        }}
                      >
                        {pos.position}
                      </span>
                    </td>

                    {/* Driver */}
                    <td className="px-3 py-2.5">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-0.5 h-5 rounded-full flex-shrink-0"
                          style={{
                            background: teamColorHex(drv?.team_colour),
                          }}
                        />
                        <span className="font-bold text-white text-sm tracking-wide">
                          {drv?.name_acronym ?? pos.driver_number}
                        </span>
                        {isSelected && (
                          <span className="text-[9px] text-f1-red font-bold border border-f1-red/50 px-1 rounded">
                            YOU
                          </span>
                        )}
                      </div>
                    </td>

                    {/* Team */}
                    <td className="px-3 py-2.5 hidden md:table-cell">
                      <span className="text-gray-400 text-xs">
                        {drv?.team_name ?? "—"}
                      </span>
                    </td>

                    {/* Tyre */}
                    <td className="px-3 py-2.5 text-center">
                      {stint?.compound ? (
                        <span
                          className="inline-flex items-center justify-center w-6 h-6 rounded-full font-bold text-xs border"
                          style={{
                            background: compoundColor(stint.compound),
                            borderColor: compoundColor(stint.compound),
                            color:
                              stint.compound === "HARD" ? "#111" : "#fff",
                          }}
                        >
                          {compoundSymbol(stint.compound)}
                        </span>
                      ) : (
                        <span className="text-gray-600">—</span>
                      )}
                    </td>

                    {/* Speed */}
                    <td className="px-3 py-2.5 text-right hidden sm:table-cell">
                      <span className="font-mono text-xs text-gray-300">
                        {cd ? `${cd.speed} km/h` : "—"}
                      </span>
                    </td>

                    {/* DRS */}
                    <td className="px-3 py-2.5 text-center hidden sm:table-cell">
                      {cd ? (
                        <span
                          className="text-xs font-bold"
                          style={{
                            color: drsOpen(cd.drs) ? "#00e676" : "#444",
                          }}
                        >
                          {drsOpen(cd.drs) ? "●" : "○"}
                        </span>
                      ) : (
                        <span className="text-gray-600">—</span>
                      )}
                    </td>

                    {/* Gap */}
                    <td className="px-3 py-2.5 text-right">
                      <span className="font-mono text-xs text-gray-300">
                        {isLeader ? (
                          <span className="text-yellow-400 font-bold">LEAD</span>
                        ) : (
                          "—"
                        )}
                      </span>
                    </td>
                  </tr>
                );
              })}
              {positions.length === 0 && (
                <tr>
                  <td colSpan={7} className="text-center text-gray-500 py-8">
                    No position data available
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Race Control Messages ── */}
      {raceMessages.length > 0 && (
        <div
          className="rounded-xl border border-gray-800 overflow-hidden"
          style={{ background: "#111" }}
        >
          <div className="px-4 py-2 border-b border-gray-800">
            <span className="text-f1-red text-sm font-bold tracking-widest uppercase">
              📻 Race Control
            </span>
          </div>
          <div className="divide-y divide-gray-800/50">
            {raceMessages.slice(0, 6).map((msg, i) => (
              <div key={i} className="px-4 py-2.5">
                <div className="flex items-start gap-2">
                  <span
                    className="text-[10px] font-bold px-1.5 py-0.5 rounded mt-0.5 flex-shrink-0"
                    style={{
                      background:
                        msg.flag === "RED"
                          ? "#e10600"
                          : msg.flag === "YELLOW"
                          ? "#ffd600"
                          : msg.flag === "GREEN"
                          ? "#00e676"
                          : msg.flag === "CHEQUERED"
                          ? "#fff"
                          : "#333",
                      color:
                        msg.flag === "YELLOW" || msg.flag === "CHEQUERED"
                          ? "#111"
                          : "#fff",
                    }}
                  >
                    {msg.flag ?? msg.category ?? "MSG"}
                  </span>
                  <span className="text-gray-300 text-xs leading-relaxed">
                    {msg.message}
                  </span>
                </div>
                <div className="text-gray-600 text-[10px] mt-1">
                  {new Date(msg.date).toLocaleTimeString()}
                  {msg.lap_number ? ` · Lap ${msg.lap_number}` : ""}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
