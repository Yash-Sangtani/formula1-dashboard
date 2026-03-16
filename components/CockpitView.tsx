"use client";

import { useEffect, useRef, useCallback } from "react";
import type { Driver, Position, CarData, Stint } from "@/lib/types";
import { teamColorHex, drsOpen, compoundColor, compoundSymbol } from "@/lib/openf1";

interface CockpitViewProps {
  drivers: Driver[];
  positions: Position[];
  selectedDriverNumber: number;
  onSelectDriver: (n: number) => void;
  carData: CarData | null;
  stints: Map<number, Stint>;
  sessionName: string;
  circuitName: string;
  countryName: string;
}

export default function CockpitView({
  drivers,
  positions,
  selectedDriverNumber,
  onSelectDriver,
  carData,
  stints,
  sessionName,
  circuitName,
  countryName,
}: CockpitViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const selectedDriver = drivers.find(
    (d) => d.driver_number === selectedDriverNumber
  );
  const selectedPosition = positions.find(
    (p) => p.driver_number === selectedDriverNumber
  );
  const myPos = selectedPosition?.position ?? 20;

  const speed = carData?.speed ?? 0;
  const gear = carData?.n_gear ?? 0;
  const rpm = carData?.rpm ?? 0;
  const throttle = carData?.throttle ?? 0;
  const brake = carData?.brake ?? 0;
  const isDrsOpen = drsOpen(carData?.drs ?? 0);

  const driverStint = stints.get(selectedDriverNumber);
  const compound = driverStint?.compound ?? "";

  // Cars directly ahead (lower position number means further ahead)
  const carsAhead = positions
    .filter((p) => p.driver_number !== selectedDriverNumber && p.position < myPos)
    .sort((a, b) => b.position - a.position) // closest first
    .slice(0, 3);

  const carBehind = positions.find((p) => p.position === myPos + 1);

  // ── Canvas draw ──────────────────────────────────────────────────────────────
  const drawScene = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    const VPX = W / 2; // vanishing point X (center)
    const VPY = H * 0.36; // vanishing point Y (horizon)

    ctx.clearRect(0, 0, W, H);

    // ── Sky gradient ──
    const skyGrad = ctx.createLinearGradient(0, 0, 0, VPY);
    skyGrad.addColorStop(0, "#4a7a9b");
    skyGrad.addColorStop(0.6, "#7aadca");
    skyGrad.addColorStop(1, "#a8cbda");
    ctx.fillStyle = skyGrad;
    ctx.fillRect(0, 0, W, VPY + 4);

    // Subtle clouds
    ctx.fillStyle = "rgba(255,255,255,0.18)";
    for (let i = 0; i < 5; i++) {
      const cx = (W * (i * 0.22 + 0.05));
      const cy = VPY * (0.2 + i * 0.08);
      ctx.beginPath();
      ctx.ellipse(cx, cy, 70 + i * 20, 18 + i * 5, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    // ── Ground beyond track (runoff / grass) ──
    const groundGrad = ctx.createLinearGradient(0, VPY, 0, H);
    groundGrad.addColorStop(0, "#3a4a35");
    groundGrad.addColorStop(0.15, "#4a5f42");
    groundGrad.addColorStop(1, "#2e3d2c");
    ctx.fillStyle = groundGrad;
    ctx.fillRect(0, VPY, W, H - VPY);

    // ── Track dimensions at various depths ──
    const TW_BOT = W * 0.54;
    const TW_TOP = W * 0.042;
    const KERB_BOT = W * 0.055;
    const KERB_TOP = W * 0.007;
    const WALL_BOT = W * 0.018;
    const WALL_TOP = W * 0.003;
    const WALL_H_BOT = 58;
    const WALL_H_TOP = 4;

    // Interpolate track coords at depth t (0=bottom, 1=horizon)
    const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
    const trackX = (side: number, t: number) =>
      VPX + side * lerp(TW_BOT / 2, TW_TOP / 2, t);
    const kerbX = (side: number, t: number) =>
      VPX + side * lerp(TW_BOT / 2 + KERB_BOT, TW_TOP / 2 + KERB_TOP, t);
    const wallX = (side: number, t: number) =>
      VPX +
      side *
        lerp(TW_BOT / 2 + KERB_BOT + WALL_BOT, TW_TOP / 2 + KERB_TOP + WALL_TOP, t);
    const trackY = (t: number) => lerp(H, VPY, t);

    // ── Kerb stripes (red/white) ──
    const N_KERBS = 14;
    for (let i = 0; i < N_KERBS; i++) {
      const t1 = i / N_KERBS;
      const t2 = (i + 1) / N_KERBS;
      const y1 = trackY(t1);
      const y2 = trackY(t2);
      ctx.fillStyle = i % 2 === 0 ? "#cc0000" : "#f0f0f0";

      // Left kerb
      ctx.beginPath();
      ctx.moveTo(trackX(-1, t1), y1);
      ctx.lineTo(kerbX(-1, t1), y1);
      ctx.lineTo(kerbX(-1, t2), y2);
      ctx.lineTo(trackX(-1, t2), y2);
      ctx.closePath();
      ctx.fill();

      // Right kerb
      ctx.beginPath();
      ctx.moveTo(trackX(1, t1), y1);
      ctx.lineTo(kerbX(1, t1), y1);
      ctx.lineTo(kerbX(1, t2), y2);
      ctx.lineTo(trackX(1, t2), y2);
      ctx.closePath();
      ctx.fill();
    }

    // ── Asphalt track surface ──
    const asphGrad = ctx.createLinearGradient(0, VPY, 0, H);
    asphGrad.addColorStop(0, "#4e4e4e");
    asphGrad.addColorStop(0.5, "#3a3a3a");
    asphGrad.addColorStop(1, "#2c2c2c");
    ctx.fillStyle = asphGrad;
    ctx.beginPath();
    ctx.moveTo(trackX(-1, 0), trackY(0));
    ctx.lineTo(trackX(-1, 1), trackY(1));
    ctx.lineTo(trackX(1, 1), trackY(1));
    ctx.lineTo(trackX(1, 0), trackY(0));
    ctx.closePath();
    ctx.fill();

    // ── Concrete barriers ──
    for (const side of [-1, 1]) {
      // Barrier face (perspective quad)
      const barGrad = ctx.createLinearGradient(0, 0, side * W * 0.3, 0);
      barGrad.addColorStop(0, "#c8c8c8");
      barGrad.addColorStop(1, "#989898");
      ctx.fillStyle = barGrad;

      ctx.beginPath();
      ctx.moveTo(kerbX(side, 0), trackY(0));
      ctx.lineTo(kerbX(side, 0), trackY(0) - WALL_H_BOT);
      ctx.lineTo(kerbX(side, 1), trackY(1) - WALL_H_TOP);
      ctx.lineTo(kerbX(side, 1), trackY(1));
      ctx.closePath();
      ctx.fill();

      // Barrier top face
      ctx.fillStyle = "#aaaaaa";
      ctx.beginPath();
      ctx.moveTo(kerbX(side, 0), trackY(0) - WALL_H_BOT);
      ctx.lineTo(wallX(side, 0), trackY(0) - WALL_H_BOT * 0.9);
      ctx.lineTo(wallX(side, 1), trackY(1) - WALL_H_TOP * 0.9);
      ctx.lineTo(kerbX(side, 1), trackY(1) - WALL_H_TOP);
      ctx.closePath();
      ctx.fill();

      // Barrier joint lines
      ctx.strokeStyle = "#888";
      ctx.lineWidth = 1;
      for (let j = 1; j < 8; j++) {
        const t = j / 8;
        const x0 = kerbX(side, t);
        const y0 = trackY(t);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x0, y0 - lerp(WALL_H_BOT, WALL_H_TOP, t));
        ctx.stroke();
      }
    }

    // ── Grandstands (behind barriers) ──
    for (const side of [-1, 1]) {
      const STAND_BOT_X = wallX(side, 0);
      const STAND_TOP_X = wallX(side, 0.8);
      const STAND_OUTER_BOT = STAND_BOT_X + side * W * 0.28;
      const STAND_OUTER_TOP = STAND_TOP_X + side * W * 0.1;

      const standGrad = ctx.createLinearGradient(0, VPY, 0, H * 0.8);
      standGrad.addColorStop(0, "#3a4550");
      standGrad.addColorStop(1, "#4a5560");
      ctx.fillStyle = standGrad;

      ctx.beginPath();
      ctx.moveTo(STAND_BOT_X, trackY(0) - WALL_H_BOT);
      ctx.lineTo(STAND_OUTER_BOT, trackY(0) - WALL_H_BOT - 200);
      ctx.lineTo(STAND_OUTER_TOP, trackY(0.8) - 60);
      ctx.lineTo(STAND_TOP_X, trackY(0.8) - WALL_H_TOP);
      ctx.closePath();
      ctx.fill();

      // Crowd dots simulation
      ctx.fillStyle = "rgba(200,200,200,0.12)";
      for (let row = 0; row < 6; row++) {
        for (let col = 0; col < 20; col++) {
          const tx = STAND_BOT_X + side * (col / 20) * W * 0.25;
          const ty = trackY(0) - WALL_H_BOT - row * 28 - 20;
          if (ty > VPY + 10) {
            ctx.beginPath();
            ctx.arc(tx, ty, 4, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }

      // F1 advertising board on barrier
      const adY = trackY(0.3) - lerp(WALL_H_BOT, WALL_H_TOP, 0.3) * 0.6;
      const adX = kerbX(side, 0.3);
      const adW = side * 60;
      const adH = -22;
      ctx.fillStyle = "#e10600";
      ctx.fillRect(adX, adY, adW, adH);
      ctx.fillStyle = "#fff";
      ctx.font = "bold 10px Arial";
      ctx.textAlign = "center";
      ctx.fillText("F1", adX + adW / 2, adY + adH * 0.4);
    }

    // ── Track edge white lines ──
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(trackX(-1, 0), trackY(0));
    ctx.lineTo(trackX(-1, 1), trackY(1));
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(trackX(1, 0), trackY(0));
    ctx.lineTo(trackX(1, 1), trackY(1));
    ctx.stroke();

    // ── Dashed center line ──
    const DASH_COUNT = 22;
    for (let i = 0; i < DASH_COUNT; i++) {
      if (i % 2 === 0) continue;
      const t1 = i / DASH_COUNT;
      const t2 = (i + 1) / DASH_COUNT;
      const lw = Math.max(0.5, 2.5 * (1 - t1));
      ctx.lineWidth = lw;
      ctx.strokeStyle = "rgba(255,255,255,0.55)";
      ctx.beginPath();
      ctx.moveTo(VPX, trackY(t1));
      ctx.lineTo(VPX, trackY(t2));
      ctx.stroke();
    }

    // ── Other cars (simple perspective boxes) ──
    carsAhead.forEach((pos, idx) => {
      const drv = drivers.find((d) => d.driver_number === pos.driver_number);
      if (!drv) return;
      const distFrac = Math.min(1, (idx + 1) / 4); // 0=close, 1=far
      const t = 0.22 + distFrac * 0.48;
      const carY2 = trackY(t);
      const carW2 = lerp(TW_BOT, TW_TOP, t) * 0.35;
      const carH2 = carW2 * 0.42;
      const color = teamColorHex(drv.team_colour);

      // Car body
      ctx.fillStyle = color;
      ctx.fillRect(VPX - carW2 / 2, carY2 - carH2, carW2, carH2 * 0.65);

      // Cockpit/halo
      ctx.fillStyle = "#111";
      ctx.fillRect(
        VPX - carW2 * 0.18,
        carY2 - carH2 * 0.95,
        carW2 * 0.36,
        carH2 * 0.45
      );

      // Rear wing
      ctx.fillStyle = color;
      ctx.fillRect(VPX - carW2 * 0.48, carY2 - carH2 * 0.85, carW2 * 0.96, 3);

      // Driver tag (only for nearby cars)
      if (t < 0.6) {
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        ctx.fillRect(VPX - carW2 / 2, carY2 - carH2 - 18, carW2, 16);
        ctx.fillStyle = "#fff";
        ctx.font = `bold ${Math.max(7, carW2 * 0.22)}px monospace`;
        ctx.textAlign = "center";
        ctx.fillText(drv.name_acronym, VPX, carY2 - carH2 - 6);
      }
    });

    // ── Cockpit / Car elements at bottom ──
    const CAR_TOP = H * 0.56;

    // Car body (nose going away, widening toward viewer)
    // Main body left side
    ctx.fillStyle = teamColorHex(selectedDriver?.team_colour);
    ctx.beginPath();
    ctx.moveTo(VPX - 12, CAR_TOP + 20);
    ctx.lineTo(VPX - 110, H + 20);
    ctx.lineTo(VPX - W * 0.46, H + 20);
    ctx.lineTo(VPX - W * 0.4, CAR_TOP + 30);
    ctx.closePath();
    ctx.fill();

    // Car body right side
    ctx.beginPath();
    ctx.moveTo(VPX + 12, CAR_TOP + 20);
    ctx.lineTo(VPX + 110, H + 20);
    ctx.lineTo(VPX + W * 0.46, H + 20);
    ctx.lineTo(VPX + W * 0.4, CAR_TOP + 30);
    ctx.closePath();
    ctx.fill();

    // Engine cover / center spine
    ctx.fillStyle = teamColorHex(selectedDriver?.team_colour);
    ctx.beginPath();
    ctx.moveTo(VPX - 10, CAR_TOP + 10);
    ctx.lineTo(VPX + 10, CAR_TOP + 10);
    ctx.lineTo(VPX + 60, H + 20);
    ctx.lineTo(VPX - 60, H + 20);
    ctx.closePath();
    ctx.fill();

    // Halo (dark arch)
    ctx.strokeStyle = "#222";
    ctx.lineWidth = 14;
    ctx.beginPath();
    ctx.ellipse(VPX, CAR_TOP + 12, 95, 38, 0, Math.PI, 0, true);
    ctx.stroke();
    ctx.lineWidth = 10;
    ctx.strokeStyle = "#333";
    ctx.beginPath();
    ctx.ellipse(VPX, CAR_TOP + 12, 95, 38, 0, Math.PI, 0, true);
    ctx.stroke();

    // Center halo pillar
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(VPX - 7, CAR_TOP - 18, 14, 36);

    // ── Steering wheel ──
    const SWX = VPX;
    const SWY = H * 0.82;
    const SWR = H * 0.11;

    // Steering wheel outer rim (thick ring)
    ctx.strokeStyle = "#333";
    ctx.lineWidth = SWR * 0.22;
    ctx.beginPath();
    ctx.arc(SWX, SWY, SWR, 0, Math.PI * 2);
    ctx.stroke();

    ctx.strokeStyle = "#555";
    ctx.lineWidth = SWR * 0.18;
    ctx.beginPath();
    ctx.arc(SWX, SWY, SWR, 0, Math.PI * 2);
    ctx.stroke();

    // Spokes
    ctx.lineWidth = SWR * 0.1;
    ctx.strokeStyle = "#444";
    // Left spoke
    ctx.beginPath();
    ctx.moveTo(SWX - SWR * 0.12, SWY);
    ctx.lineTo(SWX - SWR * 0.88, SWY + SWR * 0.35);
    ctx.stroke();
    // Right spoke
    ctx.beginPath();
    ctx.moveTo(SWX + SWR * 0.12, SWY);
    ctx.lineTo(SWX + SWR * 0.88, SWY + SWR * 0.35);
    ctx.stroke();
    // Top spoke
    ctx.beginPath();
    ctx.moveTo(SWX, SWY - SWR * 0.18);
    ctx.lineTo(SWX, SWY - SWR * 0.9);
    ctx.stroke();

    // Red crossbar
    ctx.lineWidth = SWR * 0.11;
    ctx.strokeStyle = "#e10600";
    ctx.beginPath();
    ctx.moveTo(SWX - SWR * 0.6, SWY);
    ctx.lineTo(SWX + SWR * 0.6, SWY);
    ctx.stroke();

    // Hub center
    ctx.fillStyle = "#111";
    ctx.beginPath();
    ctx.arc(SWX, SWY, SWR * 0.22, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#555";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(SWX, SWY, SWR * 0.22, 0, Math.PI * 2);
    ctx.stroke();

    // Mini dashboard on wheel (speed & gear display)
    ctx.fillStyle = "#0a0a0a";
    const dashW = SWR * 0.9;
    const dashH = SWR * 0.38;
    ctx.fillRect(SWX - dashW / 2, SWY - dashH / 2, dashW, dashH);
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(SWX - dashW / 2, SWY - dashH / 2, dashW, dashH);

    ctx.fillStyle = "#00d4ff";
    ctx.font = `bold ${dashH * 0.48}px monospace`;
    ctx.textAlign = "center";
    ctx.fillText(
      String(speed).padStart(3, "0"),
      SWX - dashW * 0.22,
      SWY + dashH * 0.16
    );
    ctx.fillStyle = "#666";
    ctx.font = `${dashH * 0.24}px monospace`;
    ctx.fillText("MPH", SWX - dashW * 0.22, SWY + dashH * 0.45);

    ctx.fillStyle = "#e10600";
    ctx.font = `bold ${dashH * 0.38}px monospace`;
    ctx.fillText(
      String(Math.round(rpm / 100)).padStart(2, "0") + "00",
      SWX + dashW * 0.28,
      SWY + dashH * 0.12
    );
    ctx.fillStyle = "#666";
    ctx.font = `${dashH * 0.24}px monospace`;
    ctx.fillText("RPM", SWX + dashW * 0.28, SWY + dashH * 0.45);

    // Driver hands on steering wheel
    for (const side of [-1, 1]) {
      ctx.fillStyle = "#e8e0d0"; // glove color
      ctx.beginPath();
      ctx.ellipse(
        SWX + side * SWR * 0.82,
        SWY + SWR * 0.28,
        SWR * 0.16,
        SWR * 0.22,
        side * 0.4,
        0,
        Math.PI * 2
      );
      ctx.fill();
      ctx.strokeStyle = "#c0b8a8";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }, [
    carData,
    selectedDriver,
    positions,
    drivers,
    carsAhead,
    speed,
    rpm,
    myPos,
  ]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resize = () => {
      const rect = canvas.parentElement!.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = Math.min(rect.height, rect.width * 0.6);
      drawScene();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas.parentElement!);
    return () => ro.disconnect();
  }, [drawScene]);

  useEffect(() => {
    drawScene();
  }, [drawScene]);

  // ── RPM bar color ──
  const rpmFrac = Math.min(1, rpm / 15000);
  const rpmBarColor =
    rpmFrac < 0.6 ? "#00e676" : rpmFrac < 0.85 ? "#ffd600" : "#e10600";

  const thrFrac = Math.min(100, throttle);
  const brkFrac = Math.min(100, brake);

  return (
    <div className="relative w-full bg-black rounded-xl overflow-hidden select-none">
      {/* ── 3D perspective canvas ── */}
      <canvas ref={canvasRef} className="w-full block" style={{ minHeight: 320 }} />

      {/* ══════════════════════ HUD OVERLAY ══════════════════════ */}

      {/* ── TOP LEFT: Race info & standings ── */}
      <div className="absolute top-3 left-3 flex flex-col gap-1">
        {/* Session label */}
        <div className="bg-black/80 backdrop-blur px-2 py-0.5 rounded text-xs text-gray-400 uppercase tracking-widest">
          {sessionName}
        </div>
        {/* Position header */}
        <div
          className="flex items-center gap-1 px-2 py-1 rounded"
          style={{ background: "rgba(0,0,0,0.85)" }}
        >
          <span className="text-white font-mono font-bold text-lg w-7 text-right">
            P{myPos}
          </span>
          <span
            className="w-1 h-6 rounded-sm"
            style={{
              background: teamColorHex(selectedDriver?.team_colour),
            }}
          />
          <span className="text-white font-bold text-sm tracking-widest">
            {selectedDriver?.name_acronym ?? "???"}
          </span>
        </div>
        {/* Cars around */}
        {carsAhead.slice(0, 2).map((pos) => {
          const drv = drivers.find((d) => d.driver_number === pos.driver_number);
          return (
            <div
              key={pos.driver_number}
              className="flex items-center gap-1 px-2 py-0.5 rounded opacity-90"
              style={{ background: "rgba(0,0,0,0.75)" }}
            >
              <span className="text-gray-400 font-mono text-xs w-7 text-right">
                P{pos.position}
              </span>
              <span
                className="w-1 h-4 rounded-sm"
                style={{ background: teamColorHex(drv?.team_colour) }}
              />
              <span className="text-gray-200 font-bold text-xs tracking-widest">
                {drv?.name_acronym ?? "???"}
              </span>
            </div>
          );
        })}
        {carBehind && (
          <div className="flex items-center gap-1 px-2 py-0.5 rounded opacity-75 mt-0.5"
            style={{ background: "rgba(0,0,0,0.6)" }}>
            <span className="text-gray-500 font-mono text-xs w-7 text-right">
              P{carBehind.position}
            </span>
            <span
              className="w-1 h-4 rounded-sm opacity-60"
              style={{
                background: teamColorHex(
                  drivers.find((d) => d.driver_number === carBehind.driver_number)?.team_colour
                ),
              }}
            />
            <span className="text-gray-400 font-bold text-xs tracking-widest opacity-75">
              {drivers.find((d) => d.driver_number === carBehind.driver_number)?.name_acronym ?? "???"}
            </span>
          </div>
        )}
      </div>

      {/* ── TOP CENTER: Circuit name ── */}
      <div className="absolute top-3 left-1/2 -translate-x-1/2 bg-black/75 px-3 py-1 rounded text-xs text-gray-300 font-bold tracking-widest uppercase whitespace-nowrap">
        {circuitName}
      </div>

      {/* ── TOP RIGHT: DRS + Sector indicators ── */}
      <div className="absolute top-3 right-3 flex flex-col items-end gap-2">
        <div
          className="px-3 py-1.5 rounded font-bold text-xs tracking-widest border"
          style={{
            background: isDrsOpen
              ? "rgba(0,230,118,0.2)"
              : "rgba(0,0,0,0.8)",
            borderColor: isDrsOpen ? "#00e676" : "#333",
            color: isDrsOpen ? "#00e676" : "#555",
          }}
        >
          DRS {isDrsOpen ? "▶ OPEN" : "● CLOSED"}
        </div>
        {/* S1 / S2 sector boxes */}
        <div className="flex gap-1">
          {["S1", "S2", "S3"].map((s) => (
            <div
              key={s}
              className="bg-black/80 border border-gray-700 text-gray-400 text-xs font-bold px-2 py-1 rounded font-mono"
            >
              {s}
              <div className="text-gray-600 text-[10px]">--:--.---</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── BOTTOM LEFT: Throttle / Brake bars + Tyre ── */}
      <div className="absolute bottom-10 left-3 flex flex-col gap-2">
        {/* Tyre compound */}
        {compound && (
          <div
            className="w-9 h-9 rounded-full flex items-center justify-center font-bold text-sm border-2 shadow-lg"
            style={{
              background: compoundColor(compound),
              borderColor: compoundColor(compound),
              color: compound === "HARD" ? "#111" : "#fff",
            }}
          >
            {compoundSymbol(compound)}
          </div>
        )}

        {/* Vertical bars */}
        <div className="flex gap-1.5 items-end">
          {/* Throttle bar */}
          <div className="flex flex-col items-center gap-0.5">
            <span className="text-[9px] text-green-400 font-bold">T</span>
            <div className="w-4 h-20 bg-gray-900 border border-gray-700 rounded-sm overflow-hidden flex flex-col-reverse">
              <div
                className="w-full transition-all duration-200 rounded-sm"
                style={{
                  height: `${thrFrac}%`,
                  background: "linear-gradient(to top, #00c853, #69f0ae)",
                }}
              />
            </div>
          </div>
          {/* Brake bar */}
          <div className="flex flex-col items-center gap-0.5">
            <span className="text-[9px] text-red-500 font-bold">B</span>
            <div className="w-4 h-20 bg-gray-900 border border-gray-700 rounded-sm overflow-hidden flex flex-col-reverse">
              <div
                className="w-full transition-all duration-200 rounded-sm"
                style={{
                  height: `${brkFrac}%`,
                  background: "linear-gradient(to top, #c62828, #ef9a9a)",
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* ── BOTTOM RIGHT: Speed & Gear display ── */}
      <div
        className="absolute bottom-10 right-3 rounded-xl px-4 py-3 flex flex-col items-center"
        style={{ background: "rgba(0,0,0,0.88)", border: "1px solid #333" }}
      >
        {/* Gear */}
        <div
          className="font-mono font-black leading-none"
          style={{
            fontSize: "clamp(36px, 6vw, 64px)",
            color: gear === 0 ? "#ffd600" : "#e10600",
            textShadow: `0 0 20px ${gear === 0 ? "#ffd600" : "#e10600"}80`,
          }}
        >
          {gear === 0 ? "N" : gear}
        </div>
        <div className="text-gray-500 text-[9px] font-bold tracking-widest mt-0.5">
          GEAR
        </div>

        <div className="w-full h-px bg-gray-700 my-2" />

        {/* Speed */}
        <div className="font-mono font-bold text-white text-center">
          <span style={{ fontSize: "clamp(20px, 3.5vw, 36px)" }}>
            {String(Math.round(speed)).padStart(3, "0")}
          </span>
          <div className="text-gray-500 text-[9px] font-bold tracking-widest">
            KM/H
          </div>
        </div>
      </div>

      {/* ── BOTTOM: RPM bar (full width) ── */}
      <div className="absolute bottom-0 left-0 right-0 h-3 bg-gray-900/80">
        {/* RPM segments (like shift lights) */}
        <div className="flex h-full px-1 gap-px">
          {Array.from({ length: 20 }).map((_, i) => {
            const seg = (i + 1) / 20;
            const active = seg <= rpmFrac;
            let segColor = "#00e676";
            if (i >= 16) segColor = "#e10600";
            else if (i >= 12) segColor = "#ffd600";
            return (
              <div
                key={i}
                className="flex-1 rounded-sm transition-all duration-75"
                style={{ background: active ? segColor : "#1a1a1a" }}
              />
            );
          })}
        </div>
      </div>

      {/* ── Driver selector overlay (bottom center) ── */}
      <div className="absolute bottom-5 left-1/2 -translate-x-1/2">
        <select
          value={selectedDriverNumber}
          onChange={(e) => onSelectDriver(Number(e.target.value))}
          className="bg-black/85 border border-gray-600 text-white text-xs font-bold rounded px-2 py-1 cursor-pointer appearance-none text-center"
          style={{ minWidth: 120 }}
        >
          {positions.map((pos) => {
            const drv = drivers.find((d) => d.driver_number === pos.driver_number);
            return (
              <option key={pos.driver_number} value={pos.driver_number}>
                P{pos.position} {drv?.name_acronym ?? pos.driver_number}
              </option>
            );
          })}
        </select>
      </div>
    </div>
  );
}
