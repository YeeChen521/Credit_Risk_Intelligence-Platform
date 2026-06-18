"""Borrower Lookup — single-applicant risk scoring page."""

from __future__ import annotations

import os

import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")

st.header("🔍 Borrower Lookup")

# ── Session state defaults ─────────────────────────────────────────────────────
for _k, _v in [("prev_id", None), ("fetched", {}), ("last_prediction", None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Layout ─────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 2])

with left:
    st.subheader("Applicant Data")

    sk_id_curr = st.number_input(
        "Application ID (SK_ID_CURR)", min_value=1, step=1, value=100002
    )

    # ── Fetch real data whenever the ID changes ────────────────────────────────
    id_changed = (sk_id_curr != st.session_state.prev_id)

    if id_changed:
        try:
            resp = requests.get(
                f"{API_BASE_URL}/applicant/{int(sk_id_curr)}", timeout=10
            )
            if resp.status_code == 200:
                st.session_state.fetched = resp.json()
                st.session_state.prev_id = sk_id_curr
            elif resp.status_code == 404:
                st.warning(f"ID {sk_id_curr} not found in training data — using manual inputs.")
                st.session_state.fetched = {}
                st.session_state.prev_id = sk_id_curr
            else:
                st.warning(f"Could not load applicant data (HTTP {resp.status_code}).")
        except requests.exceptions.RequestException:
            st.warning("Could not reach API to load applicant data.")

    fetched: dict = st.session_state.fetched

    def _f(key, default):
        v = fetched.get(key)
        return v if v is not None else default

    if fetched:
        st.success(f"Loaded data for applicant {sk_id_curr}")

    # ── Form fields (pre-filled from real data) ───────────────────────────────
    with st.expander("Application Details", expanded=bool(fetched)):
        amt_income_total = st.number_input("Annual Income",            value=float(_f("AMT_INCOME_TOTAL", 150000.0)))
        amt_credit       = st.number_input("Credit Amount",            value=float(_f("AMT_CREDIT",       450000.0)))
        amt_annuity      = st.number_input("Annuity Amount",           value=float(_f("AMT_ANNUITY",       20000.0)))
        amt_goods_price  = st.number_input("Goods Price",              value=float(_f("AMT_GOODS_PRICE",  400000.0)))
        days_birth       = st.number_input("Days Birth (negative)",    value=int(_f("DAYS_BIRTH",        -12000)))
        days_employed    = st.number_input("Days Employed (negative)", value=int(_f("DAYS_EMPLOYED",      -2000)))
        ext_source_1     = st.slider("EXT SOURCE 1", 0.0, 1.0, float(_f("EXT_SOURCE_1", 0.5)))
        ext_source_2     = st.slider("EXT SOURCE 2", 0.0, 1.0, float(_f("EXT_SOURCE_2", 0.6)))
        ext_source_3     = st.slider("EXT SOURCE 3", 0.0, 1.0, float(_f("EXT_SOURCE_3", 0.4)))

        def _pick(opts, key, default):
            v = _f(key, default)
            return opts.index(v) if v in opts else 0

        code_gender         = st.selectbox("Gender",          ["M", "F", "XNA"],             index=_pick(["M","F","XNA"],                       "CODE_GENDER",         "M"))
        flag_own_car        = st.selectbox("Owns Car",        ["Y", "N"],                    index=_pick(["Y","N"],                             "FLAG_OWN_CAR",        "Y"))
        flag_own_realty     = st.selectbox("Owns Realty",     ["Y", "N"],                    index=_pick(["Y","N"],                             "FLAG_OWN_REALTY",     "Y"))
        name_contract_type  = st.selectbox("Contract Type",   ["Cash loans","Revolving loans"],index=_pick(["Cash loans","Revolving loans"],     "NAME_CONTRACT_TYPE",  "Cash loans"))
        name_income_type    = st.selectbox("Income Type",     ["Working","Commercial associate","Pensioner","State servant","Unemployed"],
                                           index=_pick(["Working","Commercial associate","Pensioner","State servant","Unemployed"],              "NAME_INCOME_TYPE",    "Working"))
        name_education_type = st.selectbox("Education Type",  ["Secondary / secondary special","Higher education","Incomplete higher","Lower secondary","Academic degree"],
                                           index=_pick(["Secondary / secondary special","Higher education","Incomplete higher","Lower secondary","Academic degree"],
                                                       "NAME_EDUCATION_TYPE","Secondary / secondary special"))
        name_family_status  = st.selectbox("Family Status",   ["Married","Single / not married","Civil marriage","Separated","Widow"],
                                           index=_pick(["Married","Single / not married","Civil marriage","Separated","Widow"],                  "NAME_FAMILY_STATUS",  "Married"))
        name_housing_type   = st.selectbox("Housing Type",    ["House / apartment","With parents","Municipal apartment","Rented apartment","Office apartment","Co-op apartment"],
                                           index=_pick(["House / apartment","With parents","Municipal apartment","Rented apartment","Office apartment","Co-op apartment"],
                                                       "NAME_HOUSING_TYPE","House / apartment"))

    st.info("Sub-tables (bureau, installments, etc.) left empty.")

    score_clicked = st.button("Score Applicant", type="primary")


# ── Build payload from current form values ─────────────────────────────────────
def _build_payload() -> dict:
    return {
        "application": {
            "SK_ID_CURR": int(sk_id_curr),
            "AMT_INCOME_TOTAL": amt_income_total,
            "AMT_CREDIT": amt_credit,
            "AMT_ANNUITY": amt_annuity,
            "AMT_GOODS_PRICE": amt_goods_price,
            "DAYS_BIRTH": int(days_birth),
            "DAYS_EMPLOYED": int(days_employed),
            "EXT_SOURCE_1": ext_source_1,
            "EXT_SOURCE_2": ext_source_2,
            "EXT_SOURCE_3": ext_source_3,
            "CODE_GENDER": code_gender,
            "FLAG_OWN_CAR": flag_own_car,
            "FLAG_OWN_REALTY": flag_own_realty,
            "NAME_CONTRACT_TYPE": name_contract_type,
            "NAME_INCOME_TYPE": name_income_type,
            "NAME_EDUCATION_TYPE": name_education_type,
            "NAME_FAMILY_STATUS": name_family_status,
            "NAME_HOUSING_TYPE": name_housing_type,
        },
        "bureau": [], "bureau_balance": [], "previous_applications": [],
        "pos_cash": [], "installments": [], "credit_card": [],
    }


def _call_predict(payload: dict) -> None:
    try:
        with st.spinner("Scoring applicant..."):
            response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
            response.raise_for_status()
            st.session_state.last_prediction = response.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not reach the API. ({type(exc).__name__})")


# Auto-score when ID changes and data was fetched successfully
if id_changed and fetched:
    _call_predict(_build_payload())

# Manual re-score when button pressed (picks up any form edits)
if score_clicked:
    _call_predict(_build_payload())


# ── Right column — results ─────────────────────────────────────────────────────
with right:
    result = st.session_state.get("last_prediction")

    if result is None:
        st.info("Enter an Application ID to auto-score, or adjust fields and click **Score Applicant**.")
    else:
        prob: float = result["default_probability"]
        tier: str = result["risk_tier"]
        shap_features: list[dict] = result["top_shap_features"]
        cached: bool = result["cached"]

        gauge_color = {"LOW": "#28a745", "MEDIUM": "#fd7e14", "HIGH": "#dc3545"}.get(tier, "#6c757d")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Default Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 30],  "color": "#d4edda"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60, 100],"color": "#f8d7da"},
                ],
                "threshold": {"line": {"color": "black", "width": 4}, "value": prob * 100},
            },
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        tier_color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}.get(tier, "gray")
        st.markdown(
            f"<h2 style='color:{tier_color}; text-align:center'>{tier} RISK</h2>",
            unsafe_allow_html=True,
        )
        st.metric("Risk Tier", tier)

        sorted_feats = sorted(shap_features, key=lambda f: abs(f["shap_value"]), reverse=True)
        shap_vals  = [f["shap_value"]  for f in sorted_feats]
        feat_names = [f["feature_name"] for f in sorted_feats]
        bar_colors = ["#dc3545" if v >= 0 else "#007bff" for v in shap_vals]
        hover_texts = [f"Value: {f['feature_value']:.4f}<br>{f['description']}" for f in sorted_feats]

        fig_shap = go.Figure(go.Bar(
            x=shap_vals, y=feat_names, orientation="h",
            marker_color=bar_colors, hovertext=hover_texts, hoverinfo="text+x",
        ))
        fig_shap.update_layout(
            title="Top 10 Feature Contributions",
            xaxis_title="SHAP Value",
            yaxis={"autorange": "reversed"},
            height=420,
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        import pandas as pd
        shap_df = pd.DataFrame([{
            "Feature": f["feature_name"],
            "SHAP Value": round(f["shap_value"], 4),
            "Feature Value": f["feature_value"],
            "Description": f["description"],
        } for f in sorted_feats])
        st.dataframe(shap_df, use_container_width=True, hide_index=True)

        if cached:
            st.caption("⚡ Result served from cache")
        else:
            st.caption("🔄 Fresh prediction")
