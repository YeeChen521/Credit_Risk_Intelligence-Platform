"""asyncpg-based PostgreSQL connection pool and prediction audit logging."""

from __future__ import annotations

import json
import logging
import os

import asyncpg

from schemas import PredictionResponse

_log = logging.getLogger(__name__)

PREDICTIONS_TABLE_DDL: str = """
    CREATE TABLE IF NOT EXISTS predictions (
        id                  SERIAL PRIMARY KEY,
        application_id      INTEGER NOT NULL,
        default_probability FLOAT NOT NULL,
        risk_tier           VARCHAR(10) NOT NULL,
        top_shap_features   JSONB,
        model_version       VARCHAR(50),
        predicted_at        TIMESTAMPTZ DEFAULT NOW(),
        cached              BOOLEAN DEFAULT FALSE,
        request_payload     JSONB
    );
    CREATE INDEX IF NOT EXISTS idx_predictions_application_id ON predictions(application_id);
    CREATE INDEX IF NOT EXISTS idx_predictions_predicted_at ON predictions(predicted_at);
"""


async def create_db_pool() -> asyncpg.Pool:
    """Create and return an asyncpg connection pool using DATABASE_URL from the environment.

    Raises:
        RuntimeError: If DATABASE_URL is not set.
    """
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable is not set.")
    return await asyncpg.create_pool(database_url, min_size=2, max_size=10)


async def init_db(pool: asyncpg.Pool) -> None:
    """Create the predictions table and indexes if they do not exist."""
    async with pool.acquire() as conn:
        await conn.execute(PREDICTIONS_TABLE_DDL)


async def log_prediction(
    pool: asyncpg.Pool,
    response: PredictionResponse,
    raw_request: dict,
) -> None:
    """Insert one prediction record into the predictions table.

    Silently swallows errors so a DB failure never kills the scoring response.
    """
    try:
        shap_json = json.dumps([f.model_dump() for f in response.top_shap_features])
        payload_json = json.dumps(raw_request)

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO predictions (
                    application_id, default_probability, risk_tier,
                    top_shap_features, model_version, predicted_at,
                    cached, request_payload
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                response.application_id,
                response.default_probability,
                response.risk_tier,
                shap_json,
                response.model_version,
                response.predicted_at,
                response.cached,
                payload_json,
            )
    except Exception:
        _log.exception("Failed to log prediction")
