"""Batch-ingest all applicants from processed parquet files into PostgreSQL.

Reads data/processed/train_complete and data/processed/test_complete,
runs the production model bundle on every row, and inserts the results
into the predictions table.  Existing batch-ingested rows are deleted
first so the script is safely re-runnable.

Usage (from the project root, with the Docker stack running):
    python scripts/batch_ingest.py

PostgreSQL is expected at localhost:5432 (the port exposed by docker-compose).
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
import joblib
import numpy as np
import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/credit_risk"
BATCH_SIZE = 5_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Bundle ────────────────────────────────────────────────────────────────────

def _load_bundle() -> dict:
    matches = glob.glob(str(MODELS_DIR / "lgb_production_*.joblib"))
    if not matches:
        raise FileNotFoundError(
            f"No production bundle found in {MODELS_DIR}. "
            "Run `python -m src.models.promote` first."
        )
    path = max(matches, key=lambda p: Path(p).stat().st_mtime)
    log.info("Loading bundle: %s", Path(path).name)
    return joblib.load(path)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_parquet(name: str) -> pl.DataFrame | None:
    path = PROCESSED_DIR / name
    if not path.exists():
        log.warning("Parquet not found, skipping: %s", path)
        return None
    log.info("Reading %s ...", name)
    df = pl.read_parquet(path)
    log.info("  %d rows × %d cols", *df.shape)
    return df


# ── DB helpers ────────────────────────────────────────────────────────────────

async def _existing_batch_count(pool: asyncpg.Pool) -> int:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT COUNT(*) AS cnt FROM predictions "
            "WHERE request_payload->>'source' = 'batch_ingest'"
        )
    return row["cnt"]


async def _delete_batch_rows(pool: asyncpg.Pool) -> int:
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM predictions WHERE request_payload->>'source' = 'batch_ingest'"
        )
    # asyncpg returns "DELETE N" as the status string
    deleted = int(result.split()[-1]) if result else 0
    return deleted


async def _insert_batch(pool: asyncpg.Pool, records: list[tuple]) -> None:
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO predictions (
                application_id, default_probability, risk_tier,
                top_shap_features, model_version, predicted_at,
                cached, request_payload
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            records,
        )


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    bundle = _load_bundle()
    feature_cols: list[str] = bundle["feature_cols"]
    cat_cols: list[str] = bundle["cat_cols"]
    model = bundle["model"]
    encoder = bundle["encoder"]
    model_version: str = bundle["run_id"][:8]

    # Load processed parquet files
    frames: list[pl.DataFrame] = []
    for name in ("train_complete", "test_complete"):
        df = _load_parquet(name)
        if df is not None:
            frames.append(df)

    if not frames:
        log.error("No processed parquet files found in %s", PROCESSED_DIR)
        sys.exit(1)

    combined = pl.concat(frames, how="diagonal_relaxed")
    log.info("Combined dataset: %d rows", len(combined))

    if "SK_ID_CURR" not in combined.columns:
        log.error("SK_ID_CURR column missing from parquet — cannot ingest")
        sys.exit(1)

    # Add any model feature columns absent from the parquet (fill with safe defaults)
    cat_set = set(cat_cols)
    missing = [c for c in feature_cols if c not in combined.columns]
    if missing:
        log.warning("Adding %d missing feature columns with default values", len(missing))
        combined = combined.with_columns([
            pl.lit("Missing").alias(c) if c in cat_set else pl.lit(0.0).alias(c)
            for c in missing
        ])

    sk_ids: list[int] = combined["SK_ID_CURR"].cast(pl.Int64).to_list()

    # Encode and predict
    log.info("Encoding categoricals ...")
    X: pd.DataFrame = combined[feature_cols].to_pandas()
    X_enc = X.copy()
    X_enc[cat_cols] = encoder.transform(X[cat_cols])
    X_np: np.ndarray = X_enc.to_numpy(dtype=np.float64)

    log.info("Predicting %d applicants ...", len(sk_ids))
    all_probs: np.ndarray = model.predict_proba(X_np)[:, 1]
    log.info("Predictions done — score range [%.3f, %.3f]", all_probs.min(), all_probs.max())

    # Connect and ingest
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=5)
    try:
        existing = await _existing_batch_count(pool)
        if existing > 0:
            deleted = await _delete_batch_rows(pool)
            log.info("Deleted %d stale batch rows before re-ingesting", deleted)

        predicted_at = datetime.now(timezone.utc)
        source_json = json.dumps({"source": "batch_ingest"})
        total = 0

        for start in range(0, len(sk_ids), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(sk_ids))
            batch_ids = sk_ids[start:end]
            batch_probs = all_probs[start:end]

            records = []
            for sk_id, prob in zip(batch_ids, batch_probs):
                prob_f = float(prob)
                tier = "LOW" if prob_f < 0.30 else "HIGH" if prob_f > 0.60 else "MEDIUM"
                records.append((
                    int(sk_id),
                    prob_f,
                    tier,
                    "[]",          # top_shap_features — empty for batch ingest
                    model_version,
                    predicted_at,
                    False,         # cached
                    source_json,
                ))

            await _insert_batch(pool, records)
            total += len(records)
            log.info("  inserted %d / %d rows", total, len(sk_ids))

        log.info("Batch ingest complete — %d rows written to predictions table", total)

    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
