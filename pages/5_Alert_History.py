"""Searchable violation ledger with evidence thumbnails and admin deletion."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from core.ui import date_range_filter, load_scan_data, require_page_auth
from db.database import delete_violation

require_page_auth()
st.title("Alert History")
st.caption("Review, search, and resolve recorded safety events.")
start_date, end_date = date_range_filter()
scans, violations = load_scan_data(start_date, end_date)

if violations.empty:
    st.info("No violations were recorded in this date range.")
    st.stop()

search = st.text_input("Search alerts", placeholder="Violation, filename, or user")
source_types = sorted(violations["source_type"].dropna().unique().tolist())
violation_types = sorted(violations["violation_type"].dropna().unique().tolist())
filters = st.columns(2)
with filters[0]:
    selected_sources = st.multiselect("Source type", source_types, default=source_types)
with filters[1]:
    selected_types = st.multiselect("Violation type", violation_types, default=violation_types)

filtered = violations[
    violations["source_type"].isin(selected_sources)
    & violations["violation_type"].isin(selected_types)
].copy()
if search.strip():
    searchable = filtered[["violation_type", "filename", "username"]].fillna("").astype(str).agg(" ".join, axis=1)
    filtered = filtered[searchable.str.contains(search.strip(), case=False, regex=False)]

st.markdown(f"**{len(filtered):,} alert(s)**")
if filtered.empty:
    st.info("No alerts match the current search and filters.")
else:
    st.dataframe(
        filtered[["scan_timestamp", "violation_type", "confidence", "source_type", "filename", "username", "frame_number"]],
        width="stretch",
        hide_index=True,
    )
    st.markdown("#### Evidence review")
    for _, row in filtered.head(30).iterrows():
        with st.container(border=True):
            image_col, detail_col, action_col = st.columns([1, 2, 1])
            path = row.get("evidence_path")
            if path and os.path.isfile(str(path)):
                image_col.image(str(path), width="stretch")
            else:
                image_col.caption("No saved image")
            detail_col.markdown(f"**{row['violation_type']}** · {float(row['confidence']):.1%}")
            detail_col.caption(f"{row['scan_timestamp']} · {row['source_type']} · {row['filename']} · {row['username']}")
            if pd.notna(row.get("frame_number")):
                detail_col.caption(f"Frame {int(row['frame_number']):,}")
            if st.session_state.get("role") == "admin":
                if action_col.button("Delete", key=f"delete_violation_{int(row['id'])}"):
                    delete_violation(int(row["id"]))
                    st.rerun()
