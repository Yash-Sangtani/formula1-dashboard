# app.py
# Streamlit F1 Live Dashboard: track with moving car markers + tyre/position table
# Requirements:
#   pip install streamlit fastf1 plotly pandas numpy

import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import fastf1
from fastf1.core import Laps

# ------------------ App/Cache Config ------------------
st.set_page_config(page_title="F1 Live Dashboard", layout="wide")
# Enable FastF1 on-disk cache to speed up repeated queries
fastf1.Cache.enable_cache(".fastf1_cache")

# ------------------ Sidebar Controls ------------------
st.sidebar.title("F1 Live Dashboard")
year = st.sidebar.number_input("Season", min_value=2018, max_value=2025, value=2024, step=1)
# Event can be round number (int/str) or event name like "Bahrain", "Monza", etc.
event = st.sidebar.text_input("Event (name or round)", value="Bahrain")
session_code = st.sidebar.selectbox("Session", ["R", "Q", "SQ", "FP1", "FP2", "FP3"], index=0)
update_interval = st.sidebar.slider("Update interval (seconds)", 1, 15, 5, 1)
show_labels = st.sidebar.checkbox("Show driver codes on markers", value=True)

# ------------------ Helper Functions ------------------
@st.cache_data(show_spinner=False)
def load_session_once(year: int, event: str, session_code: str):
    """
    Load a session once (laps only by default). Using cache to reduce load time.
    """
    ses = fastf1.get_session(year, event, session_code)
    # laps=True to get Laps DataFrame with compounds/positions; disable heavy extras initially
    ses.load(laps=True, telemetry=False, weather=False, messages=False)
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

def get_current_positions_df(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Build a snapshot DataFrame with:
      Driver, Position, Compound, TyreLifeLaps (approx), X, Y, Team
    Uses last completed laps and latest telemetry point for X/Y.
    """
    laps: Laps = session.laps
    if laps is None or laps.empty:
        return pd.DataFrame(columns=["Driver","Position","Compound","TyreLifeLaps","X","Y","Team"])

    # latest records per driver
    latest = laps.sort_values(["Driver", "LapNumber"]).groupby("Driver").tail(1)

    # Estimate tyre life in laps as number of consecutive laps on current compound
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

    comp_df = pd.DataFrame(compounds, columns=["Driver","Compound"])
    life_df = pd.DataFrame(tyre_life, columns=["Driver","TyreLifeLaps"])

    # Team/Position if present
    pos_df = latest[["Driver","Position","Team"]].copy() if "Position" in latest else pd.DataFrame(columns=["Driver","Position","Team"])

    # For each driver, get latest telemetry point (X,Y) from their latest lap
    xy_rows = []
    for drv, drv_laps in laps.groupby("Driver"):
        drv_laps = drv_laps.sort_values("LapNumber")
        try:
            last_lap = drv_laps.iloc[-1]
            tel = last_lap.get_telemetry()  # contains 'X' and 'Y' columns
            if tel is not None and not tel.empty and "X" in tel and "Y" in tel:
                x = tel["X"].iloc[-1]
                y = tel["Y"].iloc[-1]
                xy_rows.append((drv, x, y))
            else:
                xy_rows.append((drv, np.nan, np.nan))
        except Exception:
            xy_rows.append((drv, np.nan, np.nan))

    xy_df = pd.DataFrame(xy_rows, columns=["Driver","X","Y"])

    df = pos_df.merge(comp_df, on="Driver", how="outer")
    df = df.merge(life_df, on="Driver", how="outer")
    df = df.merge(xy_df, on="Driver", how="outer")

    if "Position" in df and df["Position"].notna().any():
        df = df.sort_values("Position", na_position="last")

    # Keep consistent columns/order
    return df[["Driver","Position","Compound","TyreLifeLaps","X","Y","Team"]].reset_index(drop=True)

def get_track_xy(session: fastf1.core.Session) -> np.ndarray:
    """
    Produce a simple circuit polyline using the fastest lap telemetry X/Y.
    """
    laps = session.laps
    if laps is None or laps.empty:
        return np.empty((0, 2))
    try:
        fast_lap = laps.pick_fastest()
        tel = fast_lap.get_telemetry()  # includes 'X','Y'
        if tel is not None and not tel.empty and "X" in tel and "Y" in tel:
            xy = tel[["X","Y"]].dropna().to_numpy()
            # downsample for performance if needed
            if len(xy) > 3000:
                xy = xy[::3]
            return xy
    except Exception:
        pass
    return np.empty((0, 2))

def add_cars_to_fig(fig: go.Figure, df: pd.DataFrame, show_labels: bool) -> go.Figure:
    """
    Add a scatter trace with car markers (colored by team) and optional driver code labels.
    """
    valid = df.dropna(subset=["X", "Y"])
    if valid.empty:
        return fig

    # Deterministic colors per team
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

# ------------------ Main UI ------------------
st.title("F1 Live Dashboard")

# Load session snapshot
with st.spinner("Loading session data..."):
    try:
        session = load_session_once(year, event, session_code)
    except Exception as e:
        st.error(f"Failed to load session: {e}")
        st.stop()

# Precompute track polyline once
track_xy = get_track_xy(session)
base_fig = build_track_figure(track_xy)

# Two-column layout: left track, right table
left, right = st.columns([2, 1])

# Placeholders for live updates
track_ph = left.empty()
table_ph = right.empty()
status_ph = st.empty()

run = st.checkbox("Start live updates", value=False)

def render_once(session_obj):
    # Update session laps to catch new data points (FastF1 leverages cache to pull deltas)
    try:
        session_obj.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception:
        pass

    df = get_current_positions_df(session_obj)

    # Clone base figure and add car markers
    fig = go.Figure(base_fig.to_dict())
    fig = add_cars_to_fig(fig, df, show_labels)

    # Render chart and table
    track_ph.plotly_chart(fig, use_container_width=True)
    table_view = df[["Position", "Driver", "Compound", "TyreLifeLaps", "Team"]].copy()
    table_view.rename(columns={"TyreLifeLaps": "Tyre Age (laps)"}, inplace=True)
    table_ph.dataframe(table_view, use_container_width=True)

    return df

if run:
    while True:
        df_now = render_once(session)
        now_utc = pd.Timestamp.utcnow().strftime("%H:%M:%S UTC")
        status_ph.info(f"Last update: {now_utc} • Drivers: {len(df_now)} • Interval: {update_interval}s")
        time.sleep(update_interval)
else:
    df_now = render_once(session)
    
    status_ph.info("Live updates are paused. Tick the checkbox to start the loop.")
