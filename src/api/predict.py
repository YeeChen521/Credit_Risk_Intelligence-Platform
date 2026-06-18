"""Prediction router: POST /predict and POST /predict/batch."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on sys.path so sibling packages (features, etc.) resolve
# when this file is run directly or loaded by an interpreter without the
# project's editable install.
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

import asyncpg
import numpy as np
import pandas as pd
import polars as pl
import redis.asyncio as aioredis
from fastapi import APIRouter, HTTPException, Request

from db import log_prediction
from schemas import PredictRequest, PredictionResponse, SHAPFeature
from features.build_features import (
    assemble_master_dataset,
    bureau_balance_features,
    bureau_features,
    credit_card_features,
    fill_joined_feature_nulls,
    handle_missing_application,
    installment_payment_feature,
    pos_cash_balance_features,
    previous_application_features,
)
from features.feature_registry import FEATURE_REGISTRY

_log = logging.getLogger(__name__)

router = APIRouter(tags=["Predictions"])


# ── Empty sub-table schemas ───────────────────────────────────────────────────
# Used when a sub-table list is empty so left-joins produce null columns, not errors.

_BUREAU_SCHEMA: dict[str, pl.DataType] = {
    "SK_ID_CURR": pl.Int64, "SK_ID_BUREAU": pl.Int64, "CREDIT_ACTIVE": pl.Utf8,
    "CREDIT_CURRENCY": pl.Utf8, "DAYS_CREDIT": pl.Int64, "CREDIT_DAY_OVERDUE": pl.Int64,
    "DAYS_CREDIT_ENDDATE": pl.Float64, "DAYS_ENDDATE_FACT": pl.Float64,
    "AMT_CREDIT_SUM": pl.Float64, "AMT_CREDIT_SUM_DEBT": pl.Float64,
    "AMT_CREDIT_SUM_LIMIT": pl.Float64, "AMT_CREDIT_SUM_OVERDUE": pl.Float64,
    "CREDIT_TYPE": pl.Utf8, "DAYS_CREDIT_UPDATE": pl.Int64, "AMT_ANNUITY": pl.Float64,
    "CNT_CREDIT_PROLONG": pl.Int64,
}

_BUREAU_BALANCE_SCHEMA: dict[str, pl.DataType] = {
    "SK_ID_BUREAU": pl.Int64, "MONTHS_BALANCE": pl.Int64, "STATUS": pl.Utf8,
}

_PREV_SCHEMA: dict[str, pl.DataType] = {
    "SK_ID_CURR": pl.Int64, "SK_ID_PREV": pl.Int64,
    "NAME_CONTRACT_STATUS": pl.Utf8, "DAYS_DECISION": pl.Int64,
    "AMT_APPLICATION": pl.Float64, "AMT_CREDIT": pl.Float64,
    "AMT_ANNUITY": pl.Float64, "RATE_DOWN_PAYMENT": pl.Float64,
    "CNT_PAYMENT": pl.Float64,
}

_POS_SCHEMA: dict[str, pl.DataType] = {
    "SK_ID_CURR": pl.Int64, "SK_ID_PREV": pl.Int64, "MONTHS_BALANCE": pl.Int64,
    "CNT_INSTALMENT": pl.Float64, "CNT_INSTALMENT_FUTURE": pl.Float64,
    "NAME_CONTRACT_STATUS": pl.Utf8, "SK_DPD": pl.Int64, "SK_DPD_DEF": pl.Int64,
}

_INS_SCHEMA: dict[str, pl.DataType] = {
    "SK_ID_CURR": pl.Int64, "SK_ID_PREV": pl.Int64,
    "NUM_INSTALMENT_VERSION": pl.Float64, "NUM_INSTALMENT_NUMBER": pl.Float64,
    "DAYS_INSTALMENT": pl.Float64, "DAYS_ENTRY_PAYMENT": pl.Float64,
    "AMT_INSTALMENT": pl.Float64, "AMT_PAYMENT": pl.Float64,
}

_CC_SCHEMA: dict[str, pl.DataType] = {
    "SK_ID_CURR": pl.Int64, "SK_ID_PREV": pl.Int64, "MONTHS_BALANCE": pl.Int64,
    "AMT_BALANCE": pl.Float64, "AMT_CREDIT_LIMIT_ACTUAL": pl.Float64,
    "AMT_DRAWINGS_ATM_CURRENT": pl.Float64, "AMT_DRAWINGS_CURRENT": pl.Float64,
    "CNT_DRAWINGS_CURRENT": pl.Float64,
    "AMT_INST_MIN_REGULARITY": pl.Float64, "AMT_PAYMENT_CURRENT": pl.Float64,
    "AMT_TOTAL_RECEIVABLE": pl.Float64, "SK_DPD": pl.Int64, "SK_DPD_DEF": pl.Int64,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_empty_df(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Return an empty Polars DataFrame with the given column schema."""
    return pl.DataFrame(schema=schema)


def _to_polars(
    records: list,
    empty_schema: dict[str, pl.DataType],
    sk_id_curr: int | None = None,
) -> pl.DataFrame:
    """Convert a list of Pydantic records to a Polars DataFrame.

    Returns an empty DataFrame with empty_schema when records is empty.
    When sk_id_curr is provided, injects or overwrites the SK_ID_CURR column
    so feature functions that group by SK_ID_CURR work correctly.
    """
    if not records:
        return _build_empty_df(empty_schema)
    df = pl.from_dicts([r.model_dump() for r in records])
    if sk_id_curr is not None:
        df = df.with_columns(pl.lit(sk_id_curr).cast(pl.Int64).alias("SK_ID_CURR"))
    return df


# ── Core prediction logic ─────────────────────────────────────────────────────

async def _run_prediction(request: PredictRequest, app_state: Any) -> PredictionResponse:
    """Execute the full scoring pipeline for one applicant and return a PredictionResponse."""

    redis: aioredis.Redis = app_state.redis_client
    # Cache key includes a hash of the full payload so that changing any
    # application field produces a fresh prediction even for the same ID.
    payload_hash = hashlib.md5(
        request.model_dump_json().encode(), usedforsecurity=False
    ).hexdigest()[:8]
    key = f"prediction:{request.application.SK_ID_CURR}:{payload_hash}"

    # Step 1 — Redis cache check
    cached_raw = await redis.get(key)
    if cached_raw is not None:
        data = json.loads(cached_raw)
        data["cached"] = True
        return PredictionResponse(**data)

    # Step 2 — Convert sub-tables to Polars DataFrames
    sk_id_curr = request.application.SK_ID_CURR
    app_df = pl.from_dicts([request.application.model_dump()])
    bureau_df = _to_polars(request.bureau, _BUREAU_SCHEMA, sk_id_curr=sk_id_curr)
    bureau_balance_df = _to_polars(request.bureau_balance, _BUREAU_BALANCE_SCHEMA)
    prev_df = _to_polars(request.previous_applications, _PREV_SCHEMA, sk_id_curr=sk_id_curr)
    pos_df = _to_polars(request.pos_cash, _POS_SCHEMA, sk_id_curr=sk_id_curr)
    ins_df = _to_polars(request.installments, _INS_SCHEMA, sk_id_curr=sk_id_curr)
    cc_df = _to_polars(request.credit_card, _CC_SCHEMA, sk_id_curr=sk_id_curr)

    # Step 3 — Feature engineering
    app_cleaned = handle_missing_application(app_df, "api_request")
    bb_feats = (
        bureau_balance_features(bureau_balance_df)
        if not bureau_balance_df.is_empty()
        else _build_empty_df({"SK_ID_BUREAU": pl.Int64})
    )
    bureau_feats = bureau_features(bureau_df, bb_feats)
    ins_feats = installment_payment_feature(ins_df)
    cc_feats = credit_card_features(cc_df)
    pos_feats = pos_cash_balance_features(pos_df)
    prev_feats = previous_application_features(prev_df)
    master = assemble_master_dataset(
        app_cleaned, bureau_feats, ins_feats, cc_feats, pos_feats, prev_feats
    )
    master = fill_joined_feature_nulls(master)

    # Step 4 — Align to bundle feature_cols and encode categoricals
    bundle = app_state.bundle
    feature_cols: list[str] = bundle["feature_cols"]
    cat_cols: list[str] = bundle["cat_cols"]

    # Add any feature columns absent from master (application fields not in
    # the API request schema) with safe defaults so the encoder never sees NaN.
    cat_set = set(cat_cols)
    missing_master_cols = [c for c in feature_cols if c not in master.columns]
    if missing_master_cols:
        master = master.with_columns([
            pl.lit("Missing").alias(c) if c in cat_set else pl.lit(0.0).alias(c)
            for c in missing_master_cols
        ])

    X: pd.DataFrame = master[feature_cols].to_pandas()
    X_enc = X.copy()
    with app_state.encoder_lock:
        X_enc[cat_cols] = bundle["encoder"].transform(X[cat_cols])
    X_np: np.ndarray = X_enc.to_numpy()

    # Step 5 — Predict
    y_score: float = bundle["model"].predict_proba(X_np)[:, 1][0]

    # Step 6 — SHAP explanation (top 10 by absolute SHAP value, descending)
    shap_vals: np.ndarray = app_state.shap_explainer.shap_values(X_np)[0]
    shap_pairs = sorted(
        zip(feature_cols, shap_vals, X_np[0]),
        key=lambda t: abs(t[1]),
        reverse=True,
    )[:10]
    registry_map = {f.name: f.description for f in FEATURE_REGISTRY}
    top_shap = [
        SHAPFeature(
            feature_name=name,
            shap_value=float(sv),
            feature_value=float(fv),
            description=registry_map.get(name, ""),
        )
        for name, sv, fv in shap_pairs
    ]

    # Step 7 — Assemble response
    risk_tier = "LOW" if y_score < 0.30 else "HIGH" if y_score > 0.60 else "MEDIUM"
    response = PredictionResponse(
        application_id=request.application.SK_ID_CURR,
        default_probability=float(y_score),
        risk_tier=risk_tier,
        top_shap_features=top_shap,
        model_version=bundle["run_id"][:8],
        cached=False,
        predicted_at=datetime.now(timezone.utc),
    )

    # Step 8 — Persist to Redis cache and PostgreSQL audit log
    await redis.set(key, response.model_dump_json(), ex=3600)
    db_pool: asyncpg.Pool = app_state.db_pool
    await log_prediction(db_pool, response, request.model_dump())

    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictRequest, req: Request) -> PredictionResponse:
    """Score a single loan applicant; returns default probability and SHAP explanation."""
    try:
        return await _run_prediction(request, req.app.state)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(e).__name__}: {e}",
        ) from e


@router.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch(requests: list[PredictRequest], req: Request) -> list[PredictionResponse]:
    """Score up to 100 applicants concurrently."""
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Batch size must not exceed 100.")
    results = await asyncio.gather(*[_run_prediction(r, req.app.state) for r in requests])
    return list(results)
