"""Credit Risk Intelligence — Streamlit dashboard entry point."""

from __future__ import annotations

import os

import requests
import streamlit as st

# ── Shared config ──────────────────────────────────────────────────────────────

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API helpers ────────────────────────────────────────────────────────────────

def _get_health() -> dict | None:
    """Call GET /health. Returns the JSON body or None on any error."""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


def _get_model_info() -> dict | None:
    """Call GET /model/info. Returns the JSON body or None on any error."""
    try:
        r = requests.get(f"{API_BASE_URL}/model/info", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏦 Credit Risk Intelligence")

    # API status — checked on every page load
    health = _get_health()
    api_online = health is not None

    if api_online:
        st.markdown("🟢 **API Online**")
    else:
        st.markdown("🔴 **API Offline**")
        st.warning("API is unreachable. Check that the API service is running.")

    # Model info — only when API is up
    if api_online:
        model_info = _get_model_info()
        if model_info:
            st.divider()
            st.caption("**Loaded model**")
            st.text(f"Version:       {model_info.get('model_version', '—')}")
            st.text(f"Features:      {model_info.get('feature_count', '—')}")
            fair = model_info.get("fair")
            fair_label = "Yes ✓" if fair else "No ✗"
            st.text(f"Fair model:    {fair_label}")

    st.divider()

# ── Navigation ─────────────────────────────────────────────────────────────────

pages = [
    st.Page("pages/borrower.py", title="Borrower Lookup", icon="🔍"),
    st.Page("pages/portfolio.py", title="Portfolio View", icon="📊"),
    st.Page("pages/model_health.py", title="Model Health", icon="🩺"),
]

pg = st.navigation(pages)
pg.run()
