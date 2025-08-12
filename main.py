import fastf1
session = fastf1.get_session(2024, 'Spa', 'Race')
session.load()
laps = session.laps

from fastf1.plotting import plot_track
track = session.get_circuit_info().coordinates
# Use plotly or matplotlib to plot the track

import plotly.express as px
df = get_current_positions_df()  # DataFrame with driver, x, y
fig = px.scatter(df, x="x", y="y", color="driver")

import streamlit as st

col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.dataframe(table_data)

import time

placeholder = st.empty()
while True:
    # Fetch latest positions/telemetry
    # Update your figure and data table
    with placeholder.container():
        # Render updated UI
    time.sleep(1)  # update interval[49][46][53]pip install streamlit fastf1 plotly pan



