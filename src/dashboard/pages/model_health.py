"""Model Health — drift monitoring and model metadata page."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")

# Resolved relative to where Streamlit is launched (project root)
DRIFT_DIR = Path("data/processed/drift")

st.header("🩺 Model Health")

# ── Section 1 — Current model info ────────────────────────────────────────────

st.subheader("Loaded Model")


@st.cache_data(ttl=60)
def _fetch_model_info() -> dict | None:
    try:
        r = requests.get(f"{API_BASE_URL}/model/info", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not load model info. ({type(exc).__name__})")
        return None


model_info = _fetch_model_info()

if model_info:
    has_auc = model_info.get("training_auc") is not None
    cols = st.columns(4 if has_auc else 3)

    cols[0].metric("Model Version", model_info.get("model_version", "—"))
    cols[1].metric("Feature Count", model_info.get("feature_count", "—"))
    cols[2].metric("Fair Model", "Yes ✅" if model_info.get("fair") else "No")
    if has_auc:
        cols[3].metric("Training AUC", f"{model_info['training_auc']:.4f}")

st.divider()

# ── Section 2 — Drift monitoring ──────────────────────────────────────────────

st.subheader("Drift Monitoring")

drift_files = sorted(DRIFT_DIR.glob("*.json")) if DRIFT_DIR.exists() else []

if not drift_files:
    st.info(
        "No drift reports found. Drift monitoring activates in Phase 3. "
        "Run: python -m src.monitoring.drift_report"
    )
    with st.container(border=True):
        st.markdown("**What will appear here once drift reports are generated:**")
        st.markdown(
            "- PSI scores for income, age, credit bureau features\n"
            "- Red / amber / green status per feature group\n"
            "- Weekly PSI trend chart"
        )
else:
    # Load most recent drift report
    latest_path = drift_files[-1]
    with latest_path.open() as fh:
        latest_report: dict = json.load(fh)

    # ── a. PSI scores table ───────────────────────────────────────────────────

    psi_data: list[dict] = latest_report.get("psi_scores", [])

    if psi_data:
        def _psi_status(psi: float) -> str:
            if psi < 0.10:
                return "✅ Stable"
            if psi <= 0.20:
                return "⚠️ Warning"
            return "🚨 Alert"

        psi_df = pd.DataFrame(psi_data)
        psi_df.columns = [c.replace("_", " ").title() for c in psi_df.columns]

        # Normalise to expected column names regardless of capitalisation
        col_map = {c: c for c in psi_df.columns}
        psi_col = next((c for c in psi_df.columns if "psi" in c.lower()), None)
        group_col = next(
            (c for c in psi_df.columns if "feature" in c.lower() or "group" in c.lower()),
            psi_df.columns[0],
        )

        if psi_col:
            psi_df["Status"] = psi_df[psi_col].apply(_psi_status)

        st.dataframe(
            psi_df,
            use_container_width=True,
            hide_index=True,
            column_config=(
                {psi_col: st.column_config.NumberColumn(psi_col, format="%.4f")}
                if psi_col
                else None
            ),
        )

    # ── b. PSI trend chart ────────────────────────────────────────────────────

    if len(drift_files) > 1:
        trend_records: list[dict] = []
        for path in drift_files:
            try:
                with path.open() as fh:
                    report = json.load(fh)
                date_str = path.stem  # expects filename like drift_2024-01-15
                for entry in report.get("psi_scores", []):
                    trend_records.append({"date": date_str, **entry})
            except (json.JSONDecodeError, KeyError):
                continue

        if trend_records:
            trend_df = pd.DataFrame(trend_records)
            # Detect column names dynamically
            group_col = next(
                (c for c in trend_df.columns if c not in ("date",) and trend_df[c].dtype == object),
                None,
            )
            psi_col = next(
                (c for c in trend_df.columns if "psi" in c.lower()), None
            )

            if group_col and psi_col:
                fig_trend = go.Figure()
                for group in trend_df[group_col].unique():
                    grp = trend_df[trend_df[group_col] == group].sort_values("date")
                    fig_trend.add_trace(
                        go.Scatter(
                            x=grp["date"],
                            y=grp[psi_col],
                            mode="lines+markers",
                            name=group,
                        )
                    )

                # Reference lines
                for level, color, label in [
                    (0.10, "#fd7e14", "Warning (0.10)"),
                    (0.20, "#dc3545", "Alert (0.20)"),
                ]:
                    fig_trend.add_hline(
                        y=level,
                        line_dash="dash",
                        line_color=color,
                        annotation_text=label,
                        annotation_position="right",
                    )

                fig_trend.update_layout(
                    title="PSI Drift Over Time",
                    xaxis_title="Report Date",
                    yaxis_title="PSI Score",
                    height=400,
                    margin={"t": 50, "b": 10, "l": 10, "r": 80},
                )
                st.plotly_chart(fig_trend, use_container_width=True)

    # ── c. Raw report expander ────────────────────────────────────────────────

    with st.expander("Raw drift report (JSON)"):
        st.json(latest_report)

st.divider()

# ── Section 3 — Promotion history (stubbed) ───────────────────────────────────

st.subheader("Model Promotion History")
st.info(
    "Full promotion history will be available once MLflow integration "
    "is connected to the dashboard. For now, use the MLflow UI at "
    "http://localhost:5000 to review run history."
)
st.link_button("Open MLflow UI", "http://localhost:5000")
