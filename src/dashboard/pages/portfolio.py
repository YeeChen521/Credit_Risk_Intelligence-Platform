"""Portfolio View — portfolio-level risk overview page."""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")

st.header("📊 Portfolio View")

# ── Period selector ────────────────────────────────────────────────────────────

period: str = st.radio(
    "Time period",
    options=["all", "6m", "1y"],
    format_func=lambda x: {"all": "All time", "6m": "Last 6 months", "1y": "Last 1 year"}[x],
    horizontal=True,
)

# ── Cached API fetchers ────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _fetch_summary(period: str) -> dict | None:
    try:
        r = requests.get(f"{API_BASE_URL}/portfolio/summary", params={"period": period}, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not load portfolio summary. ({type(exc).__name__})")
        return None


@st.cache_data(ttl=300)
def _fetch_predictions(period: str, limit: int = 1000) -> list[dict] | None:
    try:
        r = requests.get(
            f"{API_BASE_URL}/portfolio/predictions",
            params={"period": period, "limit": limit},
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("predictions", [])
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not load prediction history. ({type(exc).__name__})")
        return None


@st.cache_data(ttl=300)
def _fetch_flagged(period: str, limit: int = 100) -> dict | None:
    try:
        r = requests.get(
            f"{API_BASE_URL}/portfolio/flagged",
            params={"period": period, "limit": limit},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not load high-risk applicants. ({type(exc).__name__})")
        return None


# ── Fetch all data ─────────────────────────────────────────────────────────────

with st.spinner("Loading portfolio data..."):
    summary = _fetch_summary(period)
    predictions = _fetch_predictions(period, limit=1000)
    flagged = _fetch_flagged(period, limit=100)

# ── Section 1 — Summary metrics ───────────────────────────────────────────────

if summary:
    total = summary["total_predictions"]
    high_risk = summary["high_risk_count"]
    avg_prob = summary["avg_default_probability"]
    high_risk_rate = (high_risk / total * 100) if total > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions", f"{total:,}")
    c2.metric("High Risk Count", f"{high_risk:,}")
    c3.metric("Avg Default Probability", f"{avg_prob * 100:.2f}%")
    c4.metric("High Risk Rate", f"{high_risk_rate:.2f}%")

    st.divider()

    # ── Section 2 — Donut chart ───────────────────────────────────────────────

    tier_df = pd.DataFrame(summary["tier_breakdown"])
    if not tier_df.empty:
        tier_order = ["LOW", "MEDIUM", "HIGH"]
        tier_colors = {"LOW": "#28a745", "MEDIUM": "#ffc107", "HIGH": "#dc3545"}

        tier_df["risk_tier"] = pd.Categorical(
            tier_df["risk_tier"], categories=tier_order, ordered=True
        )
        tier_df = tier_df.sort_values("risk_tier")

        fig_donut = go.Figure(
            go.Pie(
                labels=tier_df["risk_tier"],
                values=tier_df["count"],
                hole=0.5,
                marker_colors=[tier_colors.get(t, "#adb5bd") for t in tier_df["risk_tier"]],
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Count: %{value:,}<br>"
                    "Share: %{percent}<extra></extra>"
                ),
            )
        )
        fig_donut.update_layout(
            title="Risk Tier Distribution",
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
            margin={"t": 50, "b": 10, "l": 10, "r": 10},
            height=350,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.divider()

# ── Section 3 — Default probability over time ─────────────────────────────────

if predictions:
    df_hist = pd.DataFrame(predictions)
    df_hist["predicted_at"] = pd.to_datetime(df_hist["predicted_at"], utc=True)
    df_hist["date"] = df_hist["predicted_at"].dt.date

    daily = (
        df_hist.groupby("date")["default_probability"]
        .mean()
        .reset_index()
        .rename(columns={"default_probability": "avg_probability"})
    )

    if len(daily) < 2:
        st.info("Not enough data to show trend.")
    else:
        overall_avg = daily["avg_probability"].mean()
        line_color = (
            "#dc3545" if overall_avg > 0.60
            else "#fd7e14" if overall_avg > 0.30
            else "#28a745"
        )

        fig_line = go.Figure(
            go.Scatter(
                x=daily["date"],
                y=daily["avg_probability"] * 100,
                mode="lines+markers",
                line={"color": line_color, "width": 2},
                marker={"size": 5},
                hovertemplate="Date: %{x}<br>Avg Prob: %{y:.2f}%<extra></extra>",
            )
        )
        fig_line.update_layout(
            title="Average Default Probability Over Time",
            xaxis_title="Date",
            yaxis_title="Avg Default Probability (%)",
            yaxis={"ticksuffix": "%"},
            height=350,
            margin={"t": 50, "b": 10, "l": 10, "r": 10},
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.divider()

# ── Section 4 — High-risk applicants table ────────────────────────────────────

st.subheader("🚨 High Risk Applicants")

if flagged is not None:
    flagged_records = flagged.get("predictions", [])

    if not flagged_records:
        st.success("No high-risk applicants in this period.")
    else:
        df_flagged = pd.DataFrame(flagged_records)
        df_flagged["predicted_at"] = pd.to_datetime(df_flagged["predicted_at"], utc=True)

        display_df = pd.DataFrame(
            {
                "Application ID": df_flagged["application_id"],
                "Default Probability": df_flagged["default_probability"].map(
                    lambda p: f"{p * 100:.2f}%"
                ),
                "Risk Tier": df_flagged["risk_tier"],
                "Model Version": df_flagged["model_version"],
                "Predicted At": df_flagged["predicted_at"].dt.strftime("%Y-%m-%d %H:%M UTC"),
            }
        )

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download as CSV",
            data=display_df.to_csv(index=False),
            file_name=f"high_risk_{period}.csv",
            mime="text/csv",
        )
