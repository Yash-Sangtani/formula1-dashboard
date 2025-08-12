# app.py
# Streamlit F1 Live Dashboard: robust track + moving markers, bigger table with gap to leader, and radio messages
# Requirements:
#   pip install streamlit fastf1 plotly pandas numpy

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import fastf1
from fastf1.core import Laps

# ------------------ App/Cache Config ------------------
st.set_page_config(page_title="F1 Live Dashboard", layout="wide")

# Ensure FastF1 cache dir exists before enabling cache (prevents NotADirectoryError)
cache_dir = os.path.join(os.path.dirname(__file__), ".fastf1_cache")
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# ------------------ Sidebar Controls ------------------
st.sidebar.title("F1 Live Dashboard")
year = st.sidebar.number_input("Season", min_value=2018, max_value=2025, value=2024, step=1)
event = st.sidebar.text_input("Event (name or round)", value="Bahrain")
session_code = st.sidebar.selectbox("Session", ["R", "Q", "SQ", "FP1", "FP2", "FP3"], index=0)
update_interval = st.sidebar.slider("Update interval (seconds)", 1, 15, 5, 1)
show_labels = st.sidebar.checkbox("Show driver codes on markers", value=True)
max_radio_per_driver = st.sidebar.slider("Max radio messages per driver", 1, 5, 2, 1)

# ------------------ Helper Utilities ------------------
def safe_get(df: pd.DataFrame, cols):
    """Return a list of existing columns in df among cols, preserving order."""
    return [c for c in cols if c in df.columns]

def ensure_columns(df: pd.DataFrame, columns_with_defaults: dict):
    """Ensure each key exists in df; if missing, create with default value."""
    for col, default in columns_with_defaults.items():
        if col not in df.columns:
            df[col] = default
    return df

@st.cache_data(show_spinner=False)
def load_session_once(year: int, event: str, session_code: str):
    """
    Load a session once (laps + messages); cache to reduce load time.
    """
    ses = fastf1.get_session(year, event, session_code)
    ses.load(laps=True, telemetry=False, weather=False, messages=True)
    return ses

def build_track_figure(track_xy: np.ndarray) -> go.Figure:
    """
    Create a Plotly figure with the circuit polyline and fixed equal aspect axes.
    """
    fig = go.Figure()
    if len(track_xy) > 0:
        fig.add_trace(go.Scatter(
            x=track_xy[:, 0],
            y=track_xy[:, 1],
            mode="lines",
            line=dict(color="#888", width=3),
            name="Track"
        ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def get_track_xy(session: fastf1.core.Session) -> np.ndarray:
    """
    Try multiple strategies to obtain a circuit polyline; return empty if not possible.
    """
    laps = session.laps
    if laps is None or laps.empty:
        return np.empty((0, 2))
    # Strategy 1: fastest lap telemetry (X,Y)
    try:
        fast_lap = laps.pick_fastest()
        tel = fast_lap.get_telemetry()
        if tel is not None and not tel.empty and "X" in tel and "Y" in tel:
            xy = tel[["X","Y"]].dropna().to_numpy()
            if len(xy) > 3000:
                xy = xy[::3]
            return xy
    except Exception:
        pass
    # Strategy 2: any fast clean lap telemetry
    try:
        for _, lap in laps.sort_values("LapTime").head(10).iterrows():
            tel = lap.get_telemetry()
            if tel is not None and not tel.empty and "X" in tel and "Y" in tel:
                xy = tel[["X","Y"]].dropna().to_numpy()
                if len(xy) > 3000:
                    xy = xy[::3]
                return xy
    except Exception:
        pass
    # Still unavailable
    return np.empty((0, 2))

def latest_xy_from_last_lap(drv_laps: pd.DataFrame):
    """
    Last known (X,Y) from the latest lap telemetry; safe and tolerant.
    """
    try:
        last_lap = drv_laps.sort_values("LapNumber").iloc[-1]
        tel = last_lap.get_telemetry()
        if tel is not None and not tel.empty and "X" in tel and "Y" in tel:
            return tel["X"].iloc[-1], tel["Y"].iloc[-1]
    except Exception:
        pass
    return np.nan, np.nan

def add_cars_to_fig(fig: go.Figure, df: pd.DataFrame, show_labels: bool) -> go.Figure:
    """
    Add a scatter trace with car markers (colored by team) and optional driver code labels.
    """
    valid = df.dropna(subset=["X", "Y"])
    if valid.empty:
        return fig

    # Deterministic team colors
    palette = [
        "#e10600", "#3671C6", "#00D2BE", "#F58020", "#0600EF",
        "#52E252", "#2B4562", "#B6BABD", "#FF8700", "#900000",
        "#00FFFF", "#FF00FF"
    ]
    teams = sorted(valid["Team"].dropna().unique().tolist()) if "Team" in valid else []
    cmap = {team: palette[i % len(palette)] for i, team in enumerate(teams)}
    colors = valid["Team"].map(cmap) if "Team" in valid else "#e10600"

    fig.add_trace(go.Scatter(
        x=valid["X"], y=valid["Y"],
        mode="markers+text" if show_labels else "markers",
        text=valid["Driver"] if show_labels else None,
        textposition="top center",
        marker=dict(size=10, color=colors, line=dict(color="white", width=1)),
        name="Cars",
        hovertemplate=(
            "<b>%{text}</b><br>X:%{x:.1f} Y:%{y:.1f}<extra></extra>"
            if show_labels else "X:%{x:.1f} Y:%{y:.1f}<extra></extra>"
        )
    ))
    return fig

def compute_tyre_life_and_compound(laps: Laps):
    """
    Compute current compound and approximate tyre life (laps on current compound).
    """
    tyre_life = []
    compounds = []
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
        pd.DataFrame(compounds, columns=["Driver","Compound"]),
        pd.DataFrame(tyre_life, columns=["Driver","TyreLifeLaps"])
    )

def compute_gap_to_leader(laps: Laps) -> pd.DataFrame:
    """
    Compute gap to leader using latest available 'Time' (total race time) per driver as a proxy.
    Fallback to last 'LapTime' differences if necessary.
    Output: Driver, GapToLeaderSec (float).
    """
    if laps is None or laps.empty:
        return pd.DataFrame(columns=["Driver","GapToLeaderSec"])

    latest = laps.sort_values(["Driver","LapNumber"]).groupby("Driver").tail(1).copy()

    # Preferred: 'Time' is Timedelta of race time
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
                gap = (t - leader_time).total_seconds()
                gap = max(0.0, gap)
                gaps.append((row["Driver"], gap))
            else:
                gaps.append((row["Driver"], np.nan))
        return pd.DataFrame(gaps, columns=["Driver","GapToLeaderSec"])

    # Fallback: LapTime difference
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
                gap = (lt - leader_laptime).total_seconds()
                gap = max(0.0, gap)
                gaps.append((row["Driver"], gap))
            else:
                gaps.append((row["Driver"], np.nan))
        return pd.DataFrame(gaps, columns=["Driver","GapToLeaderSec"])

    return pd.DataFrame(columns=["Driver","GapToLeaderSec"])

def get_current_snapshot(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Return DataFrame with guaranteed columns:
    Driver, Position, Compound, TyreLifeLaps, X, Y, Team, GapToLeaderSec
    Missing data are filled with defaults where needed so the table never KeyErrors.
    """
    laps: Laps = session.laps
    out = pd.DataFrame(columns=["Driver","Position","Compound","TyreLifeLaps","X","Y","Team","GapToLeaderSec"])
    if laps is None or laps.empty:
        return out

    latest = laps.sort_values(["Driver","LapNumber"]).groupby("Driver").tail(1)
    pos_df = latest[["Driver","Position","Team"]].copy() if "Position" in latest else pd.DataFrame(columns=["Driver","Position","Team"])

    comp_df, life_df = compute_tyre_life_and_compound(laps)

    # X/Y coordinates for each driver (safe)
    xy_rows = []
    for drv, drv_laps in laps.groupby("Driver"):
        x, y = latest_xy_from_last_lap(drv_laps)
        xy_rows.append((drv, x, y))
    xy_df = pd.DataFrame(xy_rows, columns=["Driver","X","Y"])

    gap_df = compute_gap_to_leader(laps)

    df = pos_df.merge(comp_df, on="Driver", how="outer")
    df = df.merge(life_df, on="Driver", how="outer")
    df = df.merge(xy_df, on="Driver", how="outer")
    df = df.merge(gap_df, on="Driver", how="outer")

    df = ensure_columns(df, {
        "Position": np.nan,
        "Compound": "",
        "TyreLifeLaps": np.nan,
        "X": np.nan,
        "Y": np.nan,
        "Team": "",
        "GapToLeaderSec": np.nan
    })

    if df["Position"].notna().any():
        df = df.sort_values("Position", na_position="last")
    return df.reset_index(drop=True)

def get_radio_messages(session: fastf1.core.Session, max_per_driver: int = 2) -> pd.DataFrame:
    """
    Return most recent radio messages per driver (up to max_per_driver).
    Columns: Driver, Message, Category, Time, LapNumber (if available).
    """
    try:
        msgs = session.messages
    except Exception:
        msgs = None

    if msgs is None or msgs.empty:
        return pd.DataFrame(columns=["Driver","Message","Category","Time","LapNumber"])

    cols = msgs.columns.tolist()
    drv_col = "Driver" if "Driver" in cols else ("Recipient" if "Recipient" in cols else None)
    txt_col = "Message" if "Message" in cols else ("Text" if "Text" in cols else None)
    cat_col = "Category" if "Category" in cols else None
    time_col = "Time" if "Time" in cols else None
    lap_col = "LapNumber" if "LapNumber" in cols else ("Lap" if "Lap" in cols else None)

    if drv_col is None or txt_col is None:
        return pd.DataFrame(columns=["Driver","Message","Category","Time","LapNumber"])

    df = msgs.copy()
    if time_col and df[time_col].notna().any():
        df = df.sort_values(time_col, ascending=False)

    out = df.rename(columns={
        drv_col: "Driver",
        txt_col: "Message",
        cat_col if cat_col else "Category": "Category",
        time_col if time_col else "Time": "Time",
        lap_col if lap_col else "LapNumber": "LapNumber"
    })

    if "Driver" in out:
        out = out.groupby("Driver", as_index=False).head(max_per_driver)

    keep = ["Driver","Message","Category","Time","LapNumber"]
    out = out[[c for c in keep if c in out.columns]].copy()

    # Format Time if Timedelta
    if "Time" in out and pd.api.types.is_timedelta64_dtype(out["Time"]):
        out["Time"] = out["Time"].apply(lambda td: f"{int(td.total_seconds()//60):02d}:{td.total_seconds()%60:06.3f}")
    return out.reset_index(drop=True)

# ------------------ Main UI ------------------
st.title("F1 Live Dashboard")

with st.spinner("Loading session data..."):
    try:
        session = load_session_once(year, event, session_code)
    except Exception as e:
        st.error(f"Failed to load session: {e}")
        st.stop()

# Build base track polyline (may be empty if session lacks telemetry yet)
track_xy = get_track_xy(session)
base_fig = build_track_figure(track_xy)

# Make table bigger: allocate more width to the right column (track:table = 3:2)
left, right = st.columns([3, 2])

# Placeholders
track_ph = left.empty()
table_ph = right.empty()
radio_header_ph = right.empty()
radio_ph = right.empty()
status_ph = st.empty()

run = st.checkbox("Start live updates", value=False)

def render_once(session_obj):
    # Refresh laps and messages to capture new data
    try:
        session_obj.load(laps=True, telemetry=False, weather=False, messages=True)
    except Exception:
        pass

    df = get_current_snapshot(session_obj)

    # Build figure
    fig = go.Figure(base_fig.to_dict())
    global track_xy
    if track_xy.size == 0:
        retry_xy = get_track_xy(session_obj)
        if retry_xy.size != 0:
            track_xy = retry_xy
            tmp = build_track_figure(track_xy)
            fig = go.Figure(tmp.to_dict())
    fig = add_cars_to_fig(fig, df, show_labels)

    if track_xy.size == 0:
        left.info("Track polyline not available yet; will display markers once position telemetry appears.", icon="ℹ️")

    track_ph.plotly_chart(fig, use_container_width=True)

    # Enhanced table safely
    tbl = df.copy()

    # Gap formatting
    def fmt_gap(x):
        if pd.isna(x):
            return ""
        if x <= 0.0005:
            return "0.000s"
        return f"+{x:.3f}s"

    if "GapToLeaderSec" in tbl.columns:
        tbl["Gap to Leader"] = tbl["GapToLeaderSec"].apply(fmt_gap)
    else:
        tbl["Gap to Leader"] = ""

    # Ensure columns exist
    display_cols = ["Position", "Driver", "Gap to Leader", "Compound", "TyreLifeLaps", "Team", "X", "Y"]
    tbl = ensure_columns(tbl, {c: "" for c in display_cols})

    # Rename TyreLifeLaps for display if present
    if "TyreLifeLaps" in tbl.columns:
        tbl["Tyre Age (laps)"] = tbl["TyreLifeLaps"]
        shown = safe_get(tbl, ["Position","Driver","Gap to Leader","Compound","Tyre Age (laps)","Team","X","Y"])
    else:
        shown = safe_get(tbl, ["Position","Driver","Gap to Leader","Compound","Team","X","Y"])

    table_ph.dataframe(tbl[shown], use_container_width=True)

    # Radio messages beneath table
    radio_header_ph.subheader("Radio Messages (latest)")
    radio_df = get_radio_messages(session_obj, max_per_driver=max_radio_per_driver)
    if radio_df.empty:
        radio_ph.info("No radio messages available yet.")
    else:
        radio_cols = safe_get(radio_df, ["Driver","Time","LapNumber","Message","Category"])
        radio_ph.dataframe(radio_df[radio_cols], use_container_width=True)

    return df, radio_df

if run:
    while True:
        df_now, radio_now = render_once(session)
        now_utc = pd.Timestamp.utcnow().strftime("%H:%M:%S UTC")
        status_ph.info(f"Last update: {now_utc} • Drivers: {len(df_now)} • Interval: {update_interval}s")
        time.sleep(update_interval)
else:
    df_now, radio_now = render_once(session)
    status_ph.info("Live updates are paused. Tick the checkbox to start the loop.")
