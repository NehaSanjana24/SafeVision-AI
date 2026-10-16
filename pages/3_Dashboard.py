"""Interactive safety command center with operational trends and filters."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.report import dataframe_to_csv
from core.ui import date_range_filter, load_scan_data, require_page_auth

require_page_auth()
st.title("Safety Command Center")
st.caption("Explore site compliance, violation patterns, and recent detections.")
start_date, end_date = date_range_filter()
scans, violations = load_scan_data(start_date, end_date)

if scans.empty:
    demo = st.sidebar.toggle("Show sample demo data", value=False)
    if demo:
        rng = np.random.default_rng(22)
        now = pd.Timestamp.now().floor("h")
        scan_rows = []
        violation_rows = []
        types = ["Missing helmet", "Missing vest", "No mask"]
        sources = ["image", "video", "webcam"]
        for index in range(96):
            timestamp = now - pd.Timedelta(hours=int(rng.integers(0, 24 * 30)))
            scan_id = index + 1
            persons = int(rng.integers(1, 18))
            count = int(rng.integers(0, min(4, persons) + 1))
            scan_rows.append({"id": scan_id, "timestamp": timestamp, "source_type": sources[index % 3], "filename": f"site_{index:02}.mp4", "total_persons": persons, "total_violations": count, "compliance_pct": 100 * (persons - count) / persons, "username": ["A. Singh", "R. Chen", "M. Lopez"][index % 3]})
            for _ in range(count):
                violation_rows.append({"id": len(violation_rows) + 1, "scan_id": scan_id, "violation_type": types[int(rng.integers(0, len(types)))], "confidence": float(rng.uniform(.52, .99)), "frame_number": int(rng.integers(0, 600)), "timestamp": timestamp, "scan_timestamp": timestamp, "source_type": sources[index % 3], "filename": f"site_{index:02}.mp4", "username": ["A. Singh", "R. Chen", "M. Lopez"][index % 3]})
        scans = pd.DataFrame(scan_rows)
        violations = pd.DataFrame(violation_rows)
    else:
        st.info("No scans are recorded in this date range. Turn on sample demo data in the sidebar to explore the charts, or run a detection first.")
        st.stop()

source_options = sorted(scans["source_type"].dropna().unique().tolist())
source_filter = st.sidebar.multiselect("Source type", source_options, default=source_options)
if not violations.empty:
    violation_options = sorted(violations["violation_type"].dropna().unique().tolist())
    violation_filter = st.sidebar.multiselect("Violation type", violation_options, default=violation_options)
else:
    violation_filter = []
scans = scans[scans["source_type"].isin(source_filter)]
filtered_violations = violations[
    violations["source_type"].isin(source_filter) & violations["violation_type"].isin(violation_filter)
] if not violations.empty else violations

total_scans = len(scans)
total_violations = len(filtered_violations)
compliance = float(scans["compliance_pct"].mean()) if total_scans else 100.0
common = filtered_violations["violation_type"].mode().iloc[0] if total_violations else "None"
kpis = st.columns(4)
kpis[0].metric("Total scans", f"{total_scans:,}")
kpis[1].metric("Total violations", f"{total_violations:,}")
kpis[2].metric("Overall compliance", f"{compliance:.1f}%")
kpis[3].metric("Most common violation", str(common))

st.markdown("#### Trend lines")
resolution = st.radio("Time resolution", ["Daily", "Hourly"], horizontal=True)
frequency = "D" if resolution == "Daily" else "h"
if not filtered_violations.empty:
    filtered_violations = filtered_violations.copy()
    filtered_violations["period"] = pd.to_datetime(filtered_violations["scan_timestamp"]).dt.floor(frequency)
    violation_trend = filtered_violations.groupby("period").size().rename("Violations").reset_index()
else:
    violation_trend = pd.DataFrame(columns=["period", "Violations"])
scan_trend = scans.copy()
scan_trend["period"] = pd.to_datetime(scan_trend["timestamp"]).dt.floor(frequency)
compliance_trend = scan_trend.groupby("period")["compliance_pct"].mean().reset_index()
left, right = st.columns(2)
left.plotly_chart(px.line(violation_trend, x="period", y="Violations", markers=True, title="Violations over time", color_discrete_sequence=["#ed665d"]), width="stretch")
right.plotly_chart(px.line(compliance_trend, x="period", y="compliance_pct", markers=True, title="Compliance trend", color_discrete_sequence=["#5ec18b"]).update_yaxes(range=[0, 100], title="Compliance (%)"), width="stretch")

st.markdown("#### Distribution and site patterns")
left, right = st.columns(2)
if total_violations:
    type_counts = filtered_violations.groupby("violation_type").size().reset_index(name="count")
    left.plotly_chart(px.pie(type_counts, names="violation_type", values="count", hole=.58, title="Violation type distribution", color_discrete_sequence=["#f4a340", "#ed665d", "#5ec18b", "#5b9bd5", "#d8dca2"]), width="stretch")
else:
    left.info("No violations match the current filters.")
grouping = st.radio("Break down by", ["Source", "User"], horizontal=True, key="breakdown")
if total_violations:
    breakdown_column = "source_type" if grouping == "Source" else "username"
    counts = filtered_violations.groupby(breakdown_column).size().reset_index(name="Violations")
    right.plotly_chart(px.bar(counts, x=breakdown_column, y="Violations", title=f"Violations by {grouping.lower()}", color="Violations", color_continuous_scale=["#5ec18b", "#f4a340", "#ed665d"]), width="stretch")
else:
    right.info("No violations match the current filters.")

st.markdown("#### Confidence and timing")
left, right = st.columns(2)
if total_violations:
    left.plotly_chart(px.histogram(filtered_violations, x="confidence", nbins=18, title="Confidence score distribution", color_discrete_sequence=["#f4a340"]).update_xaxes(tickformat=".0%"), width="stretch")
    heat = filtered_violations.copy()
    heat["weekday"] = pd.to_datetime(heat["scan_timestamp"]).dt.day_name()
    heat["hour"] = pd.to_datetime(heat["scan_timestamp"]).dt.hour
    heat_data = heat.pivot_table(index="weekday", columns="hour", values="id", aggfunc="count", fill_value=0)
    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat_data = heat_data.reindex(ordered_days, fill_value=0).reindex(columns=range(24), fill_value=0)
    right.plotly_chart(px.imshow(heat_data, labels={"x": "Hour", "y": "Day of week", "color": "Violations"}, title="Violations by day and hour", color_continuous_scale="YlOrRd", aspect="auto"), width="stretch")
else:
    left.info("No confidence scores available.")
    right.info("No violation timing available.")

gauge = go.Figure(go.Indicator(mode="gauge+number", value=compliance, number={"suffix": "%"}, title={"text": "Current compliance"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#5ec18b"}, "steps": [{"range": [0, 65], "color": "#422725"}, {"range": [65, 85], "color": "#45381f"}, {"range": [85, 100], "color": "#1d3d30"}], "threshold": {"line": {"color": "#f4a340", "width": 3}, "value": 90}}))
gauge.update_layout(height=250, margin=dict(l=25, r=25, t=50, b=15))
st.plotly_chart(gauge, width="stretch")

st.markdown("#### Latest detections")
latest = filtered_violations.sort_values("scan_timestamp", ascending=False).head(100) if not filtered_violations.empty else filtered_violations
if latest.empty:
    st.info("No detections to show.")
else:
    display_columns = [column for column in ("scan_timestamp", "violation_type", "confidence", "source_type", "filename", "username", "frame_number") if column in latest.columns]
    st.dataframe(latest[display_columns], width="stretch", hide_index=True)
    st.download_button("Download latest detections CSV", dataframe_to_csv(latest[display_columns]), "latest_detections.csv", "text/csv")