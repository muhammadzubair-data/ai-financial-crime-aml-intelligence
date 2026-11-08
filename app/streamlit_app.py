"""AML Financial Crime Intelligence Platform - Streamlit entry point.

Run locally:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="Financial Crime Intelligence", page_icon="🛡️", layout="wide")

from common import load_json  # noqa: E402

V = Path(__file__).parent / "views"
pages = {
    "Overview": [
        st.Page(V / "p1_executive.py", title="Executive Overview", icon="📊", default=True),
    ],
    "Detection": [
        st.Page(V / "p2_monitoring.py", title="Transaction Monitoring", icon="🔎"),
        st.Page(V / "p3_network.py", title="Suspicious Network Explorer", icon="🕸️"),
        st.Page(V / "p4_scoring.py", title="AI Detection & Risk Scoring", icon="🎯"),
    ],
    "Investigation": [
        st.Page(V / "p5_cases.py", title="Case Management", icon="🗂️"),
    ],
    "Governance": [
        st.Page(V / "p6_performance.py", title="Model Performance", icon="📈"),
        st.Page(V / "p7_simulator.py", title="Impact Simulator", icon="⚖️"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("**Financial Crime Intelligence**")
    if "user" not in st.session_state:
        st.session_state["user"] = "Analyst (demo)"
    st.text_input("Signed in as", key="user", help="Recorded on every case action in the audit trail.")
    m = load_json("manifest.json")
    c = m["counts"]
    st.caption(f"{c['accounts']:,} accounts · {c['transactions_analysis_window']:,} transactions · "
               f"{len(m['window']['months'])} months · synthetic data")

nav.run()
