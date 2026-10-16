"""Home overview and entry point into image/video screening."""

import streamlit as st

from core.ui import load_scan_data, require_page_auth

require_page_auth()

scans, violations = load_scan_data()
st.markdown(
    """
    <div class="hero-panel">
      <div class="eyebrow">SITE SAFETY INTELLIGENCE</div>
      <h1>See risk sooner.<br>Keep every shift safer.</h1>
    """,
    unsafe_allow_html=True,
)
if st.button("Start detection  →", type="primary"):
    st.switch_page("pages/2_Detection.py")

st.markdown("#### Monitoring tools")
cards = st.columns(4)
features = [
    ("▧", "Image detection", "Inspect a still image for PPE compliance."),
    ("▶", "Video detection", "Review sampled frames with annotated playback."),
    ("⌁", "Live dashboard", "Track compliance and violation patterns."),
    ("⇩", "Violation reports", "Export evidence-backed findings."),
]
for column, (icon, title, text) in zip(cards, features):
    with column:
        st.markdown(f"<div class='feature-tile'><div>{icon}</div><strong>{title}</strong><span>{text}</span></div>", unsafe_allow_html=True)

st.markdown("#### Site pulse")
total_scans = len(scans)
total_violations = int(scans["total_violations"].sum()) if not scans.empty else 0
compliance = float(scans["compliance_pct"].mean()) if not scans.empty else 100.0
last_scan = scans.iloc[0]["timestamp"] if not scans.empty else "No scans yet"
kpis = st.columns(4)
kpis[0].metric("Total scans", f"{total_scans:,}")
kpis[1].metric("Total violations", f"{total_violations:,}")
kpis[2].metric("Compliance", f"{compliance:.1f}%")
kpis[3].metric("Last scan", str(last_scan)[:16] if total_scans else "No scans yet")