# app.py
# Streamlit F1 Live Dashboard — dark theme, live track map, cockpit racing view
# Requirements: pip install streamlit fastf1 plotly pandas numpy

import os
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import fastf1
from fastf1.core import Laps

# ------------------ App/Cache Config ------------------
st.set_page_config(page_title="F1 Live Dashboard", layout="wide", page_icon="🏎️")

# Ensure FastF1 cache dir exists
cache_dir = os.path.join(os.path.dirname(__file__), ".fastf1_cache")
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# ------------------ Dark F1 Theme CSS ------------------
st.markdown("""
<style>
/* Base dark theme */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f0f0f !important;
    color: #f0f0f0 !important;
}
[data-testid="stSidebar"] {
    background-color: #1a1a1a !important;
    border-right: 2px solid #e10600;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #e10600 !important;
    letter-spacing: 2px;
    text-transform: uppercase;
}
/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #1a1a1a !important;
    border-bottom: 2px solid #e10600;
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #1a1a1a !important;
    color: #aaa !important;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 8px 20px;
    border-radius: 4px 4px 0 0;
}
.stTabs [aria-selected="true"] {
    background-color: #e10600 !important;
    color: #fff !important;
}
/* Metrics */
[data-testid="metric-container"] {
    background-color: #1a1a1a;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 10px !important;
}
[data-testid="metric-container"] label {
    color: #888 !important;
    font-size: 11px !important;
    letter-spacing: 1px;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f0f0f0 !important;
    font-size: 24px !important;
    font-weight: 700;
}
/* Dataframe */
.stDataFrame { border: 1px solid #333; border-radius: 6px; }
/* Title */
h1 { color: #e10600 !important; letter-spacing: 3px; text-transform: uppercase; }
h2, h3 { color: #f0f0f0 !important; letter-spacing: 1px; }
/* Info/status bar */
[data-testid="stAlert"] {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: #aaa !important;
}
/* Selectbox / inputs */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background-color: #1a1a1a !important;
    border-color: #444 !important;
    color: #f0f0f0 !important;
}
/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #1a1a1a; }
::-webkit-scrollbar-thumb { background: #e10600; border-radius: 3px; }
/* Remove top padding */
.block-container { padding-top: 1rem !important; }
/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ------------------ Sidebar Controls ------------------
st.sidebar.markdown("# 🏎️ F1 Dashboard")
st.sidebar.markdown("---")
year = st.sidebar.number_input("Season", min_value=2018, max_value=2025, value=2024, step=1)
event = st.sidebar.text_input("Event (name or round)", value="Bahrain")
session_code = st.sidebar.selectbox("Session", ["R", "Q", "SQ", "FP1", "FP2", "FP3"], index=0)
st.sidebar.markdown("---")
update_interval = st.sidebar.slider("Update interval (s)", 1, 15, 5, 1)
show_labels = st.sidebar.checkbox("Show driver codes", value=True)
max_radio_per_driver = st.sidebar.slider("Max radio messages per driver", 1, 5, 2, 1)

# ------------------ Team Color Palette ------------------
TEAM_PALETTE = [
    "#e10600", "#3671C6", "#00D2BE", "#F58020", "#0600EF",
    "#52E252", "#2B4562", "#B6BABD", "#FF8700", "#900000",
    "#00FFFF", "#FF00FF"
]

# ------------------ Helper Utilities ------------------
def safe_get(df: pd.DataFrame, cols):
    return [c for c in cols if c in df.columns]

def ensure_columns(df: pd.DataFrame, columns_with_defaults: dict):
    for col, default in columns_with_defaults.items():
        if col not in df.columns:
            df[col] = default
    return df

def get_team_cmap(teams):
    teams_sorted = sorted([t for t in teams if pd.notna(t)])
    return {team: TEAM_PALETTE[i % len(TEAM_PALETTE)] for i, team in enumerate(teams_sorted)}

# ------------------ Data Loading ------------------
@st.cache_data(show_spinner=False)
def load_session_once(year: int, event: str, session_code: str):
    ses = fastf1.get_session(year, event, session_code)
    ses.load(laps=True, telemetry=False, weather=False, messages=True)
    return ses

@st.cache_data(show_spinner=False)
def load_driver_telemetry(year: int, event: str, session_code: str, driver: str):
    """Load last lap telemetry for a specific driver (Speed, nGear, RPM, Throttle, Brake, DRS, X, Y)."""
    try:
        ses = fastf1.get_session(year, event, session_code)
        ses.load(laps=True, telemetry=True, weather=False, messages=False)
        laps = ses.laps.pick_drivers(driver)
        if laps is None or laps.empty:
            return None
        last_lap = laps.sort_values("LapNumber").iloc[-1]
        tel = last_lap.get_telemetry()
        if tel is None or tel.empty:
            return None
        return tel
    except Exception:
        return None

# ------------------ Track Utilities ------------------
@st.cache_data(show_spinner=False)
def get_track_xy_cached(year: int, event: str, session_code: str) -> np.ndarray:
    """Get track XY from cached session data."""
    try:
        ses = fastf1.get_session(year, event, session_code)
        ses.load(laps=True, telemetry=True, weather=False, messages=False)
        laps = ses.laps
        if laps is None or laps.empty:
            return np.empty((0, 2))
        try:
            fast_lap = laps.pick_fastest()
            tel = fast_lap.get_telemetry()
            if tel is not None and not tel.empty and "X" in tel and "Y" in tel:
                xy = tel[["X", "Y"]].dropna().to_numpy()
                if len(xy) > 3000:
                    xy = xy[::3]
                return xy
        except Exception:
            pass
        try:
            for _, lap in laps.sort_values("LapTime").head(10).iterrows():
                tel = lap.get_telemetry()
                if tel is not None and not tel.empty and "X" in tel and "Y" in tel:
                    xy = tel[["X", "Y"]].dropna().to_numpy()
                    if len(xy) > 3000:
                        xy = xy[::3]
                    return xy
        except Exception:
            pass
    except Exception:
        pass
    return np.empty((0, 2))

def get_track_xy(session: fastf1.core.Session) -> np.ndarray:
    laps = session.laps
    if laps is None or laps.empty:
        return np.empty((0, 2))
    try:
        fast_lap = laps.pick_fastest()
        tel = fast_lap.get_telemetry()
        if tel is not None and not tel.empty and "X" in tel and "Y" in tel:
            xy = tel[["X", "Y"]].dropna().to_numpy()
            if len(xy) > 3000:
                xy = xy[::3]
            return xy
    except Exception:
        pass
    try:
        for _, lap in laps.sort_values("LapTime").head(10).iterrows():
            tel = lap.get_telemetry()
            if tel is not None and not tel.empty and "X" in tel and "Y" in tel:
                xy = tel[["X", "Y"]].dropna().to_numpy()
                if len(xy) > 3000:
                    xy = xy[::3]
                return xy
    except Exception:
        pass
    return np.empty((0, 2))

# ------------------ Dashboard Track Figure ------------------
def build_track_figure(track_xy: np.ndarray) -> go.Figure:
    fig = go.Figure()
    if len(track_xy) > 0:
        fig.add_trace(go.Scatter(
            x=track_xy[:, 0],
            y=track_xy[:, 1],
            mode="lines",
            line=dict(color="#e10600", width=4),
            name="Track",
            hoverinfo="skip"
        ))
    fig.update_layout(
        height=540,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="#0f0f0f",
        plot_bgcolor="#161616",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color="#aaa"), bgcolor="rgba(0,0,0,0)"
        )
    )
    return fig

def add_cars_to_fig(fig: go.Figure, df: pd.DataFrame, show_labels: bool) -> go.Figure:
    valid = df.dropna(subset=["X", "Y"])
    if valid.empty:
        return fig
    teams = valid["Team"].dropna().unique().tolist() if "Team" in valid else []
    cmap = get_team_cmap(teams)
    colors = valid["Team"].map(cmap).fillna("#e10600") if "Team" in valid else "#e10600"
    fig.add_trace(go.Scatter(
        x=valid["X"], y=valid["Y"],
        mode="markers+text" if show_labels else "markers",
        text=valid["Driver"] if show_labels else None,
        textposition="top center",
        textfont=dict(color="white", size=10),
        marker=dict(size=14, color=list(colors), line=dict(color="white", width=2)),
        name="Cars",
        hovertemplate=(
            "<b>%{text}</b><br>X:%{x:.1f} Y:%{y:.1f}<extra></extra>"
            if show_labels else "X:%{x:.1f} Y:%{y:.1f}<extra></extra>"
        )
    ))
    return fig

# ------------------ Tyre / Gap ------------------
def compute_tyre_life_and_compound(laps: Laps):
    tyre_life, compounds = [], []
    for drv, drv_laps in laps.groupby("Driver"):
        drv_laps = drv_laps.sort_values("LapNumber")
        if "Compound" in drv_laps and drv_laps["Compound"].notna().any():
            last_comp = drv_laps.iloc[-1]["Compound"]
            compounds.append((drv, last_comp))
            cnt = 0
            for _, row in drv_laps[::-1].iterrows():
                if row.get("Compound", None) == last_comp:
                    cnt += 1
                else:
                    break
            tyre_life.append((drv, cnt))
        else:
            compounds.append((drv, np.nan))
            tyre_life.append((drv, np.nan))
    return (
        pd.DataFrame(compounds, columns=["Driver", "Compound"]),
        pd.DataFrame(tyre_life, columns=["Driver", "TyreLifeLaps"])
    )

def compute_gap_to_leader(laps: Laps) -> pd.DataFrame:
    if laps is None or laps.empty:
        return pd.DataFrame(columns=["Driver", "GapToLeaderSec"])
    latest = laps.sort_values(["Driver", "LapNumber"]).groupby("Driver").tail(1).copy()
    if "Time" in latest and latest["Time"].notna().any():
        leader_row = None
        if "Position" in latest and latest["Position"].notna().any():
            cand = latest[latest["Position"] == 1]
            if not cand.empty:
                leader_row = cand.iloc[0]
        if leader_row is None:
            leader_row = latest.loc[latest["Time"].idxmin()]
        leader_time = leader_row["Time"]
        gaps = []
        for _, row in latest.iterrows():
            t = row["Time"]
            if pd.notna(t) and pd.notna(leader_time):
                gap = max(0.0, (t - leader_time).total_seconds())
                gaps.append((row["Driver"], gap))
            else:
                gaps.append((row["Driver"], np.nan))
        return pd.DataFrame(gaps, columns=["Driver", "GapToLeaderSec"])
    if "LapTime" in latest and latest["LapTime"].notna().any():
        leader_row = None
        if "Position" in latest and latest["Position"].notna().any():
            cand = latest[latest["Position"] == 1]
            if not cand.empty:
                leader_row = cand.iloc[0]
        if leader_row is None:
            leader_row = latest.loc[latest["LapTime"].idxmin()]
        leader_laptime = leader_row["LapTime"]
        gaps = []
        for _, row in latest.iterrows():
            lt = row["LapTime"]
            if pd.notna(lt) and pd.notna(leader_laptime):
                gap = max(0.0, (lt - leader_laptime).total_seconds())
                gaps.append((row["Driver"], gap))
            else:
                gaps.append((row["Driver"], np.nan))
        return pd.DataFrame(gaps, columns=["Driver", "GapToLeaderSec"])
    return pd.DataFrame(columns=["Driver", "GapToLeaderSec"])

def latest_xy_from_last_lap(drv_laps: pd.DataFrame):
    try:
        last_lap = drv_laps.sort_values("LapNumber").iloc[-1]
        tel = last_lap.get_telemetry()
        if tel is not None and not tel.empty and "X" in tel and "Y" in tel:
            return tel["X"].iloc[-1], tel["Y"].iloc[-1]
    except Exception:
        pass
    return np.nan, np.nan

def get_current_snapshot(session: fastf1.core.Session) -> pd.DataFrame:
    laps: Laps = session.laps
    out = pd.DataFrame(columns=["Driver", "Position", "Compound", "TyreLifeLaps", "X", "Y", "Team", "GapToLeaderSec"])
    if laps is None or laps.empty:
        return out
    latest = laps.sort_values(["Driver", "LapNumber"]).groupby("Driver").tail(1)
    pos_df = latest[["Driver", "Position", "Team"]].copy() if "Position" in latest else pd.DataFrame(columns=["Driver", "Position", "Team"])
    comp_df, life_df = compute_tyre_life_and_compound(laps)
    xy_rows = []
    for drv, drv_laps in laps.groupby("Driver"):
        x, y = latest_xy_from_last_lap(drv_laps)
        xy_rows.append((drv, x, y))
    xy_df = pd.DataFrame(xy_rows, columns=["Driver", "X", "Y"])
    gap_df = compute_gap_to_leader(laps)
    df = pos_df.merge(comp_df, on="Driver", how="outer")
    df = df.merge(life_df, on="Driver", how="outer")
    df = df.merge(xy_df, on="Driver", how="outer")
    df = df.merge(gap_df, on="Driver", how="outer")
    df = ensure_columns(df, {
        "Position": np.nan, "Compound": "", "TyreLifeLaps": np.nan,
        "X": np.nan, "Y": np.nan, "Team": "", "GapToLeaderSec": np.nan
    })
    if df["Position"].notna().any():
        df = df.sort_values("Position", na_position="last")
    return df.reset_index(drop=True)

def get_radio_messages(session: fastf1.core.Session, max_per_driver: int = 2) -> pd.DataFrame:
    try:
        msgs = session.messages
    except Exception:
        msgs = None
    if msgs is None or msgs.empty:
        return pd.DataFrame(columns=["Driver", "Message", "Category", "Time", "LapNumber"])
    cols = msgs.columns.tolist()
    drv_col = "Driver" if "Driver" in cols else ("Recipient" if "Recipient" in cols else None)
    txt_col = "Message" if "Message" in cols else ("Text" if "Text" in cols else None)
    cat_col = "Category" if "Category" in cols else None
    time_col = "Time" if "Time" in cols else None
    lap_col = "LapNumber" if "LapNumber" in cols else ("Lap" if "Lap" in cols else None)
    if drv_col is None or txt_col is None:
        return pd.DataFrame(columns=["Driver", "Message", "Category", "Time", "LapNumber"])
    df = msgs.copy()
    if time_col and df[time_col].notna().any():
        df = df.sort_values(time_col, ascending=False)
    rename_map = {drv_col: "Driver", txt_col: "Message"}
    if cat_col:
        rename_map[cat_col] = "Category"
    if time_col:
        rename_map[time_col] = "Time"
    if lap_col:
        rename_map[lap_col] = "LapNumber"
    out = df.rename(columns=rename_map)
    if "Driver" in out:
        out = out.groupby("Driver", as_index=False).head(max_per_driver)
    keep = ["Driver", "Message", "Category", "Time", "LapNumber"]
    out = out[[c for c in keep if c in out.columns]].copy()
    if "Time" in out and pd.api.types.is_timedelta64_dtype(out["Time"]):
        out["Time"] = out["Time"].apply(
            lambda td: f"{int(td.total_seconds()//60):02d}:{td.total_seconds()%60:06.3f}"
        )
    return out.reset_index(drop=True)

# ==================== COCKPIT VIEW FUNCTIONS ====================

def compute_track_edges(track_xy: np.ndarray, half_width: float = 6.0):
    """Compute left/right track edges by offsetting centerline perpendicular to travel direction."""
    if len(track_xy) < 2:
        return track_xy, track_xy
    n = len(track_xy)
    tangents = np.zeros_like(track_xy)
    tangents[1:-1] = track_xy[2:] - track_xy[:-2]
    tangents[0] = track_xy[1] - track_xy[0]
    tangents[-1] = track_xy[-1] - track_xy[-2]
    # Normalize
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms < 1e-6, 1.0, norms)
    tangents = tangents / norms
    # Perpendicular normals (rotate 90°)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    left_edge = track_xy + half_width * normals
    right_edge = track_xy - half_width * normals
    return left_edge, right_edge

def compute_driver_heading(track_xy: np.ndarray, driver_xy: np.ndarray) -> float:
    """Find heading angle (radians) of driver based on nearest track point direction."""
    if len(track_xy) < 2:
        return 0.0
    diffs = track_xy - driver_xy
    dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(dists))
    next_idx = min(idx + 3, len(track_xy) - 1)
    if next_idx == idx:
        next_idx = max(idx - 3, 0)
    if next_idx == idx:
        return 0.0
    dx = track_xy[next_idx, 0] - track_xy[idx, 0]
    dy = track_xy[next_idx, 1] - track_xy[idx, 1]
    return math.atan2(dy, dx)

def perspective_project(pts_xy: np.ndarray, driver_xy: np.ndarray, heading: float,
                        focal: float = 180.0, horizon: float = 0.60,
                        max_dist: float = 500.0, ground_scale: float = 90.0):
    """
    Project 2D track points into a simulated 3D perspective view.
    Returns (screen_x, screen_y) arrays. screen_y in [0,1], screen_x in approx [-350,350].
    """
    if len(pts_xy) == 0:
        return np.array([]), np.array([])
    # Translate to driver frame
    pts = pts_xy - driver_xy
    # Rotate so driver faces +y (forward)
    angle = -(heading - math.pi / 2)
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    pts = (rot @ pts.T).T
    # Filter: in front (positive local_y), within max_dist, not too wide
    fwd = pts[:, 1]
    side = pts[:, 0]
    mask = (fwd > 1.5) & (fwd < max_dist) & (np.abs(side) < max_dist * 0.8)
    pts = pts[mask]
    if len(pts) == 0:
        return np.array([]), np.array([])
    # Perspective projection
    screen_x = focal * pts[:, 0] / pts[:, 1]
    # screen_y: close objects at bottom (0), far at horizon
    sy = horizon - ground_scale / pts[:, 1]
    sy = np.clip(sy, 0.0, horizon)
    return screen_x, sy

def build_cockpit_figure(
    track_xy: np.ndarray,
    driver_xy: np.ndarray,
    heading: float,
    other_cars: pd.DataFrame,
    selected_driver: str,
    speed: float = 0,
    gear: int = 0,
    rpm: float = 0,
    throttle: float = 0,
    brake: float = 0,
    drs: bool = False,
) -> go.Figure:
    """Build the simulated 3D cockpit perspective figure."""

    fig = go.Figure()
    fig.update_layout(
        height=540,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(range=[-350, 350], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[0, 1.0], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        paper_bgcolor="#0f0f0f",
        plot_bgcolor="#0f0f0f",
        showlegend=False,
        title=dict(
            text=f"<b>🏎️ COCKPIT VIEW — {selected_driver}</b>",
            font=dict(color="#e10600", size=16),
            x=0.5, xanchor="center"
        )
    )

    shapes = []
    annotations = []

    # ── Sky background ──
    shapes.append(dict(
        type="rect", xref="x", yref="y",
        x0=-350, x1=350, y0=0.60, y1=1.0,
        fillcolor="#0d1b2a", line_width=0, layer="below"
    ))
    # ── Asphalt ground ──
    shapes.append(dict(
        type="rect", xref="x", yref="y",
        x0=-350, x1=350, y0=0.0, y1=0.60,
        fillcolor="#1a1a1a", line_width=0, layer="below"
    ))
    # ── Horizon line ──
    shapes.append(dict(
        type="line", xref="x", yref="y",
        x0=-350, x1=350, y0=0.60, y1=0.60,
        line=dict(color="#333", width=1)
    ))

    # ── Perspective track ──
    if len(track_xy) > 0 and not np.all(np.isnan(driver_xy)):
        left_edge, right_edge = compute_track_edges(track_xy, half_width=7.0)

        sx_l, sy_l = perspective_project(left_edge, driver_xy, heading)
        sx_r, sy_r = perspective_project(right_edge, driver_xy, heading)
        sx_c, sy_c = perspective_project(track_xy, driver_xy, heading)

        # Track ribbon (filled polygon: left edge forward, right edge backward)
        if len(sx_l) > 1 and len(sx_r) > 1:
            poly_x = list(sx_l) + list(sx_r[::-1]) + [sx_l[0]]
            poly_y = list(sy_l) + list(sy_r[::-1]) + [sy_l[0]]
            fig.add_trace(go.Scatter(
                x=poly_x, y=poly_y,
                fill="toself",
                fillcolor="#2d2d2d",
                line=dict(color="#555", width=1),
                hoverinfo="skip", mode="lines"
            ))

        # Track edge kerbs (white lines)
        if len(sx_l) > 1:
            fig.add_trace(go.Scatter(
                x=sx_l, y=sy_l, mode="lines",
                line=dict(color="#ffffff", width=2), hoverinfo="skip"
            ))
        if len(sx_r) > 1:
            fig.add_trace(go.Scatter(
                x=sx_r, y=sy_r, mode="lines",
                line=dict(color="#ffffff", width=2), hoverinfo="skip"
            ))

        # Center dashed line (white)
        if len(sx_c) > 2:
            # Downsample for dashed effect
            step = max(1, len(sx_c) // 20)
            for i in range(0, len(sx_c) - step, step * 2):
                fig.add_trace(go.Scatter(
                    x=[sx_c[i], sx_c[min(i + step, len(sx_c)-1)]],
                    y=[sy_c[i], sy_c[min(i + step, len(sy_c)-1)]],
                    mode="lines", line=dict(color="#fff", width=2, dash="solid"),
                    hoverinfo="skip"
                ))

        # ── Other cars on track ──
        valid_cars = other_cars.dropna(subset=["X", "Y"])
        teams = valid_cars["Team"].dropna().unique().tolist() if "Team" in valid_cars else []
        cmap = get_team_cmap(teams)

        for _, car in valid_cars.iterrows():
            if car["Driver"] == selected_driver:
                continue
            car_xy = np.array([[car["X"], car["Y"]]])
            cx, cy = perspective_project(car_xy, driver_xy, heading)
            if len(cx) > 0:
                team_color = cmap.get(car.get("Team", ""), "#888")
                fig.add_trace(go.Scatter(
                    x=cx, y=cy,
                    mode="markers+text",
                    text=[car["Driver"]],
                    textposition="top center",
                    textfont=dict(color="white", size=9),
                    marker=dict(size=14, color=team_color,
                                line=dict(color="white", width=2),
                                symbol="circle"),
                    hovertemplate=f"<b>{car['Driver']}</b><extra></extra>"
                ))

    # ── Driver position indicator (triangle at bottom center) ──
    shapes.append(dict(
        type="path", xref="x", yref="y",
        path="M 0 0.12 L -15 0.03 L 15 0.03 Z",
        fillcolor="#e10600", line_color="#fff", line_width=2
    ))

    # ── Steering wheel (paper coordinates) ──
    # Outer rim
    shapes.append(dict(
        type="circle", xref="paper", yref="paper",
        x0=0.36, y0=-0.02, x1=0.64, y1=0.20,
        fillcolor="#1c1c1c", line=dict(color="#666", width=6)
    ))
    # Hub
    shapes.append(dict(
        type="circle", xref="paper", yref="paper",
        x0=0.475, y0=0.06, x1=0.525, y1=0.10,
        fillcolor="#111", line=dict(color="#888", width=2)
    ))
    # Top spoke
    shapes.append(dict(
        type="line", xref="paper", yref="paper",
        x0=0.50, y0=0.09, x1=0.50, y1=0.19,
        line=dict(color="#555", width=5)
    ))
    # Bottom-left spoke
    shapes.append(dict(
        type="line", xref="paper", yref="paper",
        x0=0.50, y0=0.07, x1=0.39, y1=0.02,
        line=dict(color="#555", width=5)
    ))
    # Bottom-right spoke
    shapes.append(dict(
        type="line", xref="paper", yref="paper",
        x0=0.50, y0=0.07, x1=0.61, y1=0.02,
        line=dict(color="#555", width=5)
    ))
    # Horizontal cross-bar
    shapes.append(dict(
        type="line", xref="paper", yref="paper",
        x0=0.40, y0=0.09, x1=0.60, y1=0.09,
        line=dict(color="#e10600", width=4)
    ))

    # ── RPM bar (bottom, spanning full width) ──
    rpm_frac = min(max(rpm / 15000.0, 0.0), 1.0)
    # Background
    shapes.append(dict(
        type="rect", xref="paper", yref="paper",
        x0=0.0, y0=-0.04, x1=1.0, y1=-0.01,
        fillcolor="#2a2a2a", line_width=0
    ))
    # Fill (color transitions: green → yellow → red)
    bar_color = "#00e676" if rpm_frac < 0.6 else ("#ffd600" if rpm_frac < 0.85 else "#e10600")
    shapes.append(dict(
        type="rect", xref="paper", yref="paper",
        x0=0.0, y0=-0.04, x1=rpm_frac, y1=-0.01,
        fillcolor=bar_color, line_width=0
    ))

    # ── Speed annotation (left) ──
    annotations.append(dict(
        x=0.08, y=0.12, xref="paper", yref="paper",
        text=f"<b>{speed:.0f}</b>",
        font=dict(color="#ffffff", size=36, family="monospace"),
        showarrow=False, align="center"
    ))
    annotations.append(dict(
        x=0.08, y=0.05, xref="paper", yref="paper",
        text="km/h",
        font=dict(color="#888", size=12),
        showarrow=False, align="center"
    ))

    # ── Gear annotation (right) ──
    annotations.append(dict(
        x=0.92, y=0.12, xref="paper", yref="paper",
        text=f"<b>{gear}</b>",
        font=dict(color="#e10600", size=48, family="monospace"),
        showarrow=False, align="center"
    ))
    annotations.append(dict(
        x=0.92, y=0.04, xref="paper", yref="paper",
        text="GEAR",
        font=dict(color="#888", size=12),
        showarrow=False, align="center"
    ))

    # ── DRS indicator (top right) ──
    drs_color = "#00e676" if drs else "#555"
    drs_text = "DRS ▶" if drs else "DRS ●"
    annotations.append(dict(
        x=0.92, y=0.96, xref="paper", yref="paper",
        text=f"<b>{drs_text}</b>",
        font=dict(color=drs_color, size=14),
        showarrow=False, bgcolor="#111", borderpad=4
    ))

    # ── Throttle / Brake bars (left side) ──
    thr_frac = min(max(throttle / 100.0, 0.0), 1.0)
    brk_frac = min(max(brake / 100.0, 0.0), 1.0)
    bar_h = 0.25
    bar_top = 0.55
    bar_bot = bar_top - bar_h
    # Throttle background + fill
    shapes.append(dict(
        type="rect", xref="paper", yref="paper",
        x0=0.02, y0=bar_bot, x1=0.04, y1=bar_top,
        fillcolor="#222", line=dict(color="#444", width=1)
    ))
    shapes.append(dict(
        type="rect", xref="paper", yref="paper",
        x0=0.02, y0=bar_bot, x1=0.04, y1=bar_bot + bar_h * thr_frac,
        fillcolor="#00e676", line_width=0
    ))
    annotations.append(dict(
        x=0.03, y=bar_top + 0.03, xref="paper", yref="paper",
        text="T", font=dict(color="#00e676", size=11), showarrow=False
    ))
    # Brake background + fill
    shapes.append(dict(
        type="rect", xref="paper", yref="paper",
        x0=0.05, y0=bar_bot, x1=0.07, y1=bar_top,
        fillcolor="#222", line=dict(color="#444", width=1)
    ))
    shapes.append(dict(
        type="rect", xref="paper", yref="paper",
        x0=0.05, y0=bar_bot, x1=0.07, y1=bar_bot + bar_h * brk_frac,
        fillcolor="#e10600", line_width=0
    ))
    annotations.append(dict(
        x=0.06, y=bar_top + 0.03, xref="paper", yref="paper",
        text="B", font=dict(color="#e10600", size=11), showarrow=False
    ))

    fig.update_layout(shapes=shapes, annotations=annotations)
    return fig

# ==================== MAIN UI ====================
st.title("🏎️ F1 Live Dashboard")

with st.spinner("Loading session data..."):
    try:
        session = load_session_once(year, event, session_code)
    except Exception as e:
        st.error(f"Failed to load session: {e}")
        st.stop()

track_xy = get_track_xy(session)
base_fig = build_track_figure(track_xy)

tab1, tab2 = st.tabs(["📊 Live Dashboard", "🏎️ Cockpit View"])

# ── Placeholders ──
with tab1:
    left, right = st.columns([3, 2])
    track_ph = left.empty()
    table_ph = right.empty()
    radio_header_ph = right.empty()
    radio_ph = right.empty()
    status_ph = st.empty()
    run = st.checkbox("▶ Start live updates", value=False)

with tab2:
    cockpit_driver_ph = st.empty()
    cockpit_fig_ph = st.empty()
    cockpit_metrics_ph = st.empty()
    cockpit_gap_ph = st.empty()
    cockpit_status_ph = st.empty()


def render_dashboard(session_obj):
    """Render the Live Dashboard tab (Tab 1)."""
    try:
        session_obj.load(laps=True, telemetry=False, weather=False, messages=True)
    except Exception:
        pass

    df = get_current_snapshot(session_obj)

    global track_xy, base_fig
    fig = go.Figure(base_fig.to_dict())
    if track_xy.size == 0:
        retry_xy = get_track_xy(session_obj)
        if retry_xy.size != 0:
            track_xy = retry_xy
            base_fig = build_track_figure(track_xy)
            fig = go.Figure(base_fig.to_dict())
    fig = add_cars_to_fig(fig, df, show_labels)

    if track_xy.size == 0:
        left.info("Track polyline not available yet — will appear once telemetry loads.", icon="ℹ️")

    track_ph.plotly_chart(fig, use_container_width=True)

    # Standings table
    tbl = df.copy()
    def fmt_gap(x):
        if pd.isna(x): return ""
        return "LEADER" if x <= 0.0005 else f"+{x:.3f}s"

    tbl["Gap to Leader"] = tbl["GapToLeaderSec"].apply(fmt_gap) if "GapToLeaderSec" in tbl.columns else ""
    if "TyreLifeLaps" in tbl.columns:
        tbl["Tyre Age"] = tbl["TyreLifeLaps"]
        shown = safe_get(tbl, ["Position", "Driver", "Team", "Gap to Leader", "Compound", "Tyre Age"])
    else:
        shown = safe_get(tbl, ["Position", "Driver", "Team", "Gap to Leader", "Compound"])

    with table_ph.container():
        st.markdown("#### 🏆 Driver Standings")
        st.dataframe(tbl[shown], use_container_width=True, hide_index=True)

    radio_header_ph.markdown("#### 📻 Radio Messages")
    radio_df = get_radio_messages(session_obj, max_per_driver=max_radio_per_driver)
    if radio_df.empty:
        radio_ph.info("No radio messages available yet.")
    else:
        radio_cols = safe_get(radio_df, ["Driver", "Time", "LapNumber", "Message"])
        radio_ph.dataframe(radio_df[radio_cols], use_container_width=True, hide_index=True)

    return df


def render_cockpit(session_obj, df: pd.DataFrame):
    """Render the Cockpit View tab (Tab 2)."""
    drivers = []
    if not df.empty and "Driver" in df.columns:
        drivers = sorted(df["Driver"].dropna().tolist())

    if not drivers:
        cockpit_driver_ph.warning("No driver data available yet. Load a session with lap data.")
        return

    with cockpit_driver_ph.container():
        col_sel, col_info = st.columns([1, 2])
        with col_sel:
            selected_driver = st.selectbox(
                "🏎️ You are driving as:",
                drivers,
                key="cockpit_driver_select"
            )
        with col_info:
            drv_row = df[df["Driver"] == selected_driver]
            if not drv_row.empty:
                row = drv_row.iloc[0]
                pos = int(row["Position"]) if pd.notna(row.get("Position")) else "—"
                team = row.get("Team", "—") or "—"
                gap = row.get("GapToLeaderSec", np.nan)
                gap_str = "LEADER" if pd.isna(gap) or gap <= 0.0005 else f"+{gap:.3f}s"
                st.markdown(f"""
<div style="background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:12px;margin-top:8px">
  <span style="color:#e10600;font-size:22px;font-weight:700">P{pos}</span>
  &nbsp;&nbsp;
  <span style="color:#f0f0f0;font-size:18px;font-weight:600">{selected_driver}</span>
  &nbsp;&nbsp;
  <span style="color:#888;font-size:14px">{team}</span>
  &nbsp;&nbsp;
  <span style="color:#aaa;font-size:14px">{gap_str}</span>
</div>
""", unsafe_allow_html=True)

    # Get driver XY from snapshot
    drv_row = df[df["Driver"] == selected_driver]
    driver_xy = np.array([np.nan, np.nan])
    if not drv_row.empty:
        x, y = drv_row.iloc[0].get("X", np.nan), drv_row.iloc[0].get("Y", np.nan)
        if pd.notna(x) and pd.notna(y):
            driver_xy = np.array([float(x), float(y)])

    # Load telemetry for HUD
    tel = load_driver_telemetry(year, event, session_code, selected_driver)

    speed, gear, rpm, throttle, brake, drs_open = 0.0, 1, 0.0, 0.0, 0.0, False
    if tel is not None and not tel.empty:
        last = tel.iloc[-1]
        speed = float(last.get("Speed", 0) or 0)
        gear = int(last.get("nGear", 1) or 1)
        rpm = float(last.get("RPM", 0) or 0)
        throttle = float(last.get("Throttle", 0) or 0)
        brake = float(last.get("Brake", 0) or 0) * 100  # FastF1 brake is 0-1
        drs_val = last.get("DRS", 0)
        drs_open = int(drs_val or 0) in (10, 12, 14)

    # Compute heading
    heading = 0.0
    if len(track_xy) > 0 and not np.all(np.isnan(driver_xy)):
        heading = compute_driver_heading(track_xy, driver_xy)

    # Build cockpit figure
    cockpit_fig = build_cockpit_figure(
        track_xy=track_xy,
        driver_xy=driver_xy,
        heading=heading,
        other_cars=df,
        selected_driver=selected_driver,
        speed=speed, gear=gear, rpm=rpm,
        throttle=throttle, brake=brake, drs=drs_open
    )
    cockpit_fig_ph.plotly_chart(cockpit_fig, use_container_width=True)

    # ── Metrics row ──
    with cockpit_metrics_ph.container():
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("SPEED", f"{speed:.0f}", help="km/h")
        c2.metric("GEAR", f"{gear}")
        c3.metric("RPM", f"{rpm:.0f}")
        c4.metric("THROTTLE", f"{throttle:.0f}%")
        c5.metric("BRAKE", f"{brake:.0f}%")
        c6.metric("DRS", "OPEN ✓" if drs_open else "CLOSED")

    # ── Gap to ahead / behind ──
    with cockpit_gap_ph.container():
        sorted_df = df.dropna(subset=["Position"]).sort_values("Position")
        drv_positions = sorted_df["Driver"].tolist()
        if selected_driver in drv_positions:
            idx = drv_positions.index(selected_driver)
            g1, g2 = st.columns(2)
            if idx > 0:
                car_ahead = drv_positions[idx - 1]
                ahead_row = df[df["Driver"] == car_ahead]
                if not ahead_row.empty:
                    my_gap = df[df["Driver"] == selected_driver]["GapToLeaderSec"].values
                    their_gap = ahead_row["GapToLeaderSec"].values
                    if len(my_gap) > 0 and len(their_gap) > 0 and pd.notna(my_gap[0]) and pd.notna(their_gap[0]):
                        diff = abs(float(my_gap[0]) - float(their_gap[0]))
                        g1.metric(f"GAP AHEAD ({car_ahead})", f"+{diff:.3f}s")
            if idx < len(drv_positions) - 1:
                car_behind = drv_positions[idx + 1]
                behind_row = df[df["Driver"] == car_behind]
                if not behind_row.empty:
                    my_gap = df[df["Driver"] == selected_driver]["GapToLeaderSec"].values
                    their_gap = behind_row["GapToLeaderSec"].values
                    if len(my_gap) > 0 and len(their_gap) > 0 and pd.notna(my_gap[0]) and pd.notna(their_gap[0]):
                        diff = abs(float(their_gap[0]) - float(my_gap[0]))
                        g2.metric(f"GAP BEHIND ({car_behind})", f"+{diff:.3f}s")


# ── Main render loop ──
with tab1:
    if run:
        while True:
            df_now = render_dashboard(session)
            now_utc = pd.Timestamp.utcnow().strftime("%H:%M:%S UTC")
            status_ph.info(
                f"⏱ Last update: **{now_utc}** &nbsp;|&nbsp; "
                f"Drivers: **{len(df_now)}** &nbsp;|&nbsp; "
                f"Interval: **{update_interval}s**"
            )
            with tab2:
                render_cockpit(session, df_now)
            time.sleep(update_interval)
    else:
        df_now = render_dashboard(session)
        status_ph.info("⏸ Live updates paused — tick ▶ to start the loop.")
        with tab2:
            render_cockpit(session, df_now)
