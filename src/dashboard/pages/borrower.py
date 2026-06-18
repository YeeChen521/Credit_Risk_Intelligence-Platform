"""Borrower Lookup — single-applicant risk scoring page."""

from __future__ import annotations

import os

import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")

st.header("🔍 Borrower Lookup")

# ── Layout ─────────────────────────────────────────────────────────────────────

left, right = st.columns([1, 2])

# ── Left column — input form ──────────────────────────────────────────────────

with left:
    st.subheader("Applicant Data")

    sk_id_curr = st.number_input(
        "Application ID (SK_ID_CURR)", min_value=1, step=1, value=100002
    )

    # Fetch real applicant data from the API whenever the ID changes
    fetched: dict = {}
    fetch_error: str = ""
    if sk_id_curr:
        try:
            resp = requests.get(f"{API_BASE_URL}/applicant/{int(sk_id_curr)}", timeout=10)
            if resp.status_code == 200:
                fetched = resp.json()
            elif resp.status_code == 404:
                fetch_error = f"ID {sk_id_curr} not found in training data — using manual inputs."
            else:
                fetch_error = f"Could not load applicant data (HTTP {resp.status_code})."
        except requests.exceptions.RequestException:
            fetch_error = "Could not reach API to load applicant data."

    if fetch_error:
        st.warning(fetch_error)
    elif fetched:
        st.success(f"Loaded data for applicant {sk_id_curr}")

    def _f(key: str, default):
        """Return fetched value if present and non-None, else the default."""
        v = fetched.get(key)
        return v if v is not None else default

    with st.expander("Application Details", expanded=bool(fetched)):
        amt_income_total = st.number_input("Annual Income",        value=float(_f("AMT_INCOME_TOTAL", 150000.0)))
        amt_credit       = st.number_input("Credit Amount",        value=float(_f("AMT_CREDIT",       450000.0)))
        amt_annuity      = st.number_input("Annuity Amount",       value=float(_f("AMT_ANNUITY",       20000.0)))
        amt_goods_price  = st.number_input("Goods Price",          value=float(_f("AMT_GOODS_PRICE",  400000.0)))
        days_birth       = st.number_input("Days Birth (negative)", value=int(_f("DAYS_BIRTH",        -12000)))
        days_employed    = st.number_input("Days Employed (negative)", value=int(_f("DAYS_EMPLOYED",   -2000)))
        ext_source_1     = st.slider("EXT SOURCE 1", 0.0, 1.0, float(_f("EXT_SOURCE_1", 0.5)))
        ext_source_2     = st.slider("EXT SOURCE 2", 0.0, 1.0, float(_f("EXT_SOURCE_2", 0.6)))
        ext_source_3     = st.slider("EXT SOURCE 3", 0.0, 1.0, float(_f("EXT_SOURCE_3", 0.4)))

        gender_opts = ["M", "F", "XNA"]
        code_gender = st.selectbox(
            "Gender", gender_opts,
            index=gender_opts.index(_f("CODE_GENDER", "M")) if _f("CODE_GENDER", "M") in gender_opts else 0
        )

        car_opts = ["Y", "N"]
        flag_own_car = st.selectbox(
            "Owns Car", car_opts,
            index=car_opts.index(_f("FLAG_OWN_CAR", "Y")) if _f("FLAG_OWN_CAR", "Y") in car_opts else 0
        )

        realty_opts = ["Y", "N"]
        flag_own_realty = st.selectbox(
            "Owns Realty", realty_opts,
            index=realty_opts.index(_f("FLAG_OWN_REALTY", "Y")) if _f("FLAG_OWN_REALTY", "Y") in realty_opts else 0
        )

        contract_opts = ["Cash loans", "Revolving loans"]
        name_contract_type = st.selectbox(
            "Contract Type", contract_opts,
            index=contract_opts.index(_f("NAME_CONTRACT_TYPE", "Cash loans")) if _f("NAME_CONTRACT_TYPE", "Cash loans") in contract_opts else 0
        )

        income_opts = ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"]
        name_income_type = st.selectbox(
            "Income Type", income_opts,
            index=income_opts.index(_f("NAME_INCOME_TYPE", "Working")) if _f("NAME_INCOME_TYPE", "Working") in income_opts else 0
        )

        edu_opts = [
            "Secondary / secondary special", "Higher education",
            "Incomplete higher", "Lower secondary", "Academic degree",
        ]
        name_education_type = st.selectbox(
            "Education Type", edu_opts,
            index=edu_opts.index(_f("NAME_EDUCATION_TYPE", "Secondary / secondary special")) if _f("NAME_EDUCATION_TYPE", "Secondary / secondary special") in edu_opts else 0
        )

        family_opts = ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
        name_family_status = st.selectbox(
            "Family Status", family_opts,
            index=family_opts.index(_f("NAME_FAMILY_STATUS", "Married")) if _f("NAME_FAMILY_STATUS", "Married") in family_opts else 0
        )

        housing_opts = [
            "House / apartment", "With parents", "Municipal apartment",
            "Rented apartment", "Office apartment", "Co-op apartment",
        ]
        name_housing_type = st.selectbox(
            "Housing Type", housing_opts,
            index=housing_opts.index(_f("NAME_HOUSING_TYPE", "House / apartment")) if _f("NAME_HOUSING_TYPE", "House / apartment") in housing_opts else 0
        )

    st.info(
        "Sub-tables (bureau, installments, etc.) left empty. "
        "The model will score using application fields only."
    )

    score_clicked = st.button("Score Applicant", type="primary")

# ── API call ───────────────────────────────────────────────────────────────────

if score_clicked:
    payload = {
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
        "bureau": [],
        "bureau_balance": [],
        "previous_applications": [],
        "pos_cash": [],
        "installments": [],
        "credit_card": [],
    }

    try:
        with st.spinner("Scoring applicant..."):
            response = requests.post(
                f"{API_BASE_URL}/predict", json=payload, timeout=30
            )
            response.raise_for_status()
            st.session_state["last_prediction"] = response.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not reach the API. Please check that the service is running. ({type(exc).__name__})")

# ── Right column — results ─────────────────────────────────────────────────────

with right:
    result = st.session_state.get("last_prediction")

    if result is None:
        st.info("Enter an Application ID and click **Score Applicant** to see results.")
    else:
        prob: float = result["default_probability"]
        tier: str = result["risk_tier"]
        shap_features: list[dict] = result["top_shap_features"]
        cached: bool = result["cached"]

        # 1 — Gauge
        gauge_color = {"LOW": "#28a745", "MEDIUM": "#fd7e14", "HIGH": "#dc3545"}.get(
            tier, "#6c757d"
        )
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Default Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, 30], "color": "#d4edda"},
                        {"range": [30, 60], "color": "#fff3cd"},
                        {"range": [60, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "value": prob * 100,
                    },
                },
            )
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # 2 — Risk tier badge
        tier_color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}.get(tier, "gray")
        st.markdown(
            f"<h2 style='color:{tier_color}; text-align:center'>{tier} RISK</h2>",
            unsafe_allow_html=True,
        )
        st.metric("Risk Tier", tier)

        # 3 — SHAP waterfall (horizontal bar)
        sorted_feats = sorted(shap_features, key=lambda f: abs(f["shap_value"]), reverse=True)
        shap_vals = [f["shap_value"] for f in sorted_feats]
        feat_names = [f["feature_name"] for f in sorted_feats]
        bar_colors = ["#dc3545" if v >= 0 else "#007bff" for v in shap_vals]
        hover_texts = [
            f"Value: {f['feature_value']:.4f}<br>{f['description']}" for f in sorted_feats
        ]

        fig_shap = go.Figure(
            go.Bar(
                x=shap_vals,
                y=feat_names,
                orientation="h",
                marker_color=bar_colors,
                hovertext=hover_texts,
                hoverinfo="text+x",
            )
        )
        fig_shap.update_layout(
            title="Top 10 Feature Contributions",
            xaxis_title="SHAP Value",
            yaxis={"autorange": "reversed"},
            height=420,
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # 4 — SHAP table
        import pandas as pd

        shap_df = pd.DataFrame(
            [
                {
                    "Feature": f["feature_name"],
                    "SHAP Value": round(f["shap_value"], 4),
                    "Feature Value": f["feature_value"],
                    "Description": f["description"],
                }
                for f in sorted_feats
            ]
        )
        st.dataframe(shap_df, use_container_width=True, hide_index=True)

        # 5 — Cache indicator
        if cached:
            st.caption("⚡ Result served from cache")
        else:
            st.caption("🔄 Fresh prediction")
