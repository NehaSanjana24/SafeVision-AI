"""Entry point and authenticated navigation for AI Safety Monitoring."""

from __future__ import annotations

import streamlit as st

from auth.auth import initialize_default_admin, login_screen
from db.database import init_database

st.set_page_config(
    page_title="AI Safety Monitoring",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_database()
initialize_default_admin()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #152520; --muted: #718079; --green: #5ec18b; --orange: #f4a340; --red: #ed665d; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    [data-testid="stAppViewContainer"] { background: radial-gradient(ellipse at 8% 4%, #18342d 0%, #101a18 42%, #111514 100%); }
    [data-testid="stSidebar"] { background: #111c19; border-right: 1px solid #294139; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMetric"] { background: #192722; border: 1px solid #2c433a; padding: 16px 18px; border-radius: 8px; }
    [data-testid="stMetricLabel"] { color: #9bac a3; }
    .auth-kicker { color: #f4a340; font-size: .76rem; font-weight: 700; letter-spacing: 2px; margin: 4rem 0 .7rem; }
    .hero-panel { padding: 3rem 0 2rem; max-width: 940px; }
    .eyebrow { color: #f4a340; font-weight: 700; text-transform: uppercase; font-size: .78rem; }
    .hero-panel h1 { font-size: 3.6rem; line-height: 1.05; margin: .6rem 0 1rem; }
    .hero-panel p { color: #aebbb5; max-width: 660px; font-size: 1.08rem; }
    .feature-tile { height: 148px; padding: 20px; border: 1px solid #2c433a; border-radius: 8px; background: linear-gradient(145deg,#1b2d26,#17211e); }
    .feature-tile strong { display: block; color: #e8efeb; font-size: 1.02rem; margin: 12px 0 5px; }
    .feature-tile span { color: #9baca3; font-size: .9rem; }
    @media (max-width: 680px) { .hero-panel h1 { font-size: 2.45rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.get("authenticated"):
    login_screen()
    st.stop()

pages = [
    st.Page("pages/1_Home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/2_Detection.py", title="Detection", icon="🦺"),
    st.Page("pages/3_Dashboard.py", title="Dashboard", icon="📊"),
    st.Page("pages/4_Reports.py", title="Reports", icon="📄"),
    st.Page("pages/5_Alert_History.py", title="Alert History", icon="🚨"),
]
navigation = st.navigation(pages)
navigation.run()