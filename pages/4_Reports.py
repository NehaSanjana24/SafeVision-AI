"""Build and export date-bounded violation reports."""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from core.report import build_pdf_report, dataframe_to_csv
from core.ui import date_range_filter, load_scan_data, require_page_auth

require_page_auth()
st.title("Violation Reports")
st.caption("Create an evidence-backed summary for a selected reporting period.")
start_date, end_date = date_range_filter("Report date range")
scans, violations = load_scan_data(start_date, end_date)
st.markdown(f"**Reporting period:** {start_date:%b %d, %Y} to {end_date:%b %d, %Y}")

if scans.empty:
    st.info("No scans were recorded in this date range. Run a detection to build a report.")
    st.stop()

total_violations = len(violations)
average_compliance = float(scans["compliance_pct"].mean())
most_common = violations["violation_type"].mode().iloc[0] if total_violations else "None"
kpis = st.columns(4)
kpis[0].metric("Scans", f"{len(scans):,}")
kpis[1].metric("Violations", f"{total_violations:,}")
kpis[2].metric("Mean compliance", f"{average_compliance:.1f}%")
kpis[3].metric("Most common", str(most_common))

if total_violations:
    breakdown = violations.groupby("violation_type").size().reset_index(name="Count").sort_values("Count", ascending=False)
    left, right = st.columns([1, 1.5])
    left.dataframe(breakdown, width="stretch", hide_index=True)
    right.plotly_chart(px.bar(breakdown, x="violation_type", y="Count", color="Count", color_continuous_scale=["#f4a340", "#ed665d"], title="Violation breakdown"), width="stretch")
else:
    st.success("No violations were found for this period.")

st.markdown("#### Annotated evidence")
evidence = violations[violations["evidence_path"].notna()] if "evidence_path" in violations.columns else pd.DataFrame()
if evidence.empty:
    st.caption("No evidence images are available for this period.")
else:
    for _, row in evidence.drop_duplicates("evidence_path").head(12).iterrows():
        path = str(row["evidence_path"])
        if os.path.isfile(path):
            st.image(path, caption=f"{row['violation_type']} · {row['filename']} · {row['scan_timestamp']}", width=480)

pdf_bytes = build_pdf_report(violations, scans)
csv_bytes = dataframe_to_csv(violations)
left, right = st.columns(2)
left.download_button("Download PDF report", pdf_bytes, file_name=f"safety_report_{start_date}_{end_date}.pdf", mime="application/pdf", type="primary", width="stretch")
right.download_button("Download violations CSV", csv_bytes, file_name=f"safety_violations_{start_date}_{end_date}.csv", mime="text/csv", width="stretch")