"""Weekly Evidently AI drift report generator.

Usage:
    python -m src.monitoring.drift_report

Reads recent predictions from the PostgreSQL predictions table, compares them
against the training reference distribution, and writes a JSON report to
data/processed/drift/. Also logs per-feature PSI scores to the active
production MLflow run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import date
from pathlib import Path

import asyncpg
import mlflow
import pandas as pd
import polars as pl
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ── Paths & constants ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DRIFT_DIR = PROCESSED_DIR / "drift"

DRIFT_FEATURES: list[str] = [
    "AMT_INCOME_TOTAL",
    "DAYS_BIRTH",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "bureau_loan_count",
    "debt_to_credit_ratio",
]

EVAL_HOLDOUT_FRAC: float = 0.20
LOOKBACK_DAYS: int = 7


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_reference() -> pd.DataFrame:
    """Load the training split of train_complete and return only DRIFT_FEATURES."""
    parquet_path = PROCESSED_DIR / "train_complete"
    df = pl.read_parquet(parquet_path).sort("SK_ID_CURR")
    n_train = int(len(df) * (1 - EVAL_HOLDOUT_FRAC))
    df = df.head(n_train)

    available = [c for c in DRIFT_FEATURES if c in df.columns]
    missing = set(DRIFT_FEATURES) - set(available)
    if missing:
        logger.warning("Reference data missing columns: %s", missing)

    return df.select(available).to_pandas()


async def _load_current(pool: asyncpg.Pool) -> pd.DataFrame | None:
    """Query the last LOOKBACK_DAYS days of predictions and extract DRIFT_FEATURES."""
    rows = await pool.fetch(
        "SELECT request_payload, predicted_at "
        "FROM predictions "
        "WHERE predicted_at >= NOW() - $1::interval",
        f"{LOOKBACK_DAYS} days",
    )

    if len(rows) < 10:
        logger.warning(
            "Only %d prediction rows in the last %d days — skipping drift report.",
            len(rows),
            LOOKBACK_DAYS,
        )
        return None

    records: list[dict] = []
    for row in rows:
        payload = row["request_payload"]
        # asyncpg decodes JSONB to dict; guard against unexpected string payloads
        if isinstance(payload, str):
            payload = json.loads(payload)
        app = payload.get("application", {}) if isinstance(payload, dict) else {}
        records.append({feat: app.get(feat) for feat in DRIFT_FEATURES})

    df = pd.DataFrame(records)
    # Coerce to numeric — missing derived features (bureau_loan_count, etc.) become NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _run_evidently(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Run an Evidently DataDriftPreset report and return a structured result dict."""
    from evidently.metric_preset import DataDriftPreset  # noqa: PLC0415
    from evidently.report import Report  # noqa: PLC0415

    # Restrict both DataFrames to the columns actually present in both
    shared_cols = [c for c in DRIFT_FEATURES if c in reference.columns and c in current.columns]

    report = Report(metrics=[DataDriftPreset(columns=shared_cols)])
    report.run(reference_data=reference[shared_cols], current_data=current[shared_cols])
    result = json.loads(report.json())

    # Locate the per-column drift stats (DataDriftTable metric)
    drift_by_cols: dict = {}
    for metric in result.get("metrics", []):
        res = metric.get("result", {})
        if "drift_by_columns" in res:
            drift_by_cols = res["drift_by_columns"]
            break

    features: dict[str, dict] = {}
    for feat in DRIFT_FEATURES:
        col_data = drift_by_cols.get(feat, {})
        features[feat] = {
            "psi": float(col_data.get("drift_score", 0.0)),
            "drifted": bool(col_data.get("drift_detected", False)),
            "stattest": col_data.get("stattest_name", "unknown"),
        }

    any_drift = any(v["drifted"] for v in features.values())
    max_psi = max((v["psi"] for v in features.values()), default=0.0)

    # psi_scores: list format consumed by the Streamlit dashboard
    psi_scores = [
        {
            "feature_group": name,
            "psi_score": stats["psi"],
            "stattest": stats["stattest"],
            "drifted": stats["drifted"],
        }
        for name, stats in features.items()
    ]

    return {
        "report_date": date.today().isoformat(),
        "lookback_days": LOOKBACK_DAYS,
        "n_reference": len(reference),
        "n_current": len(current),
        "features": features,
        "psi_scores": psi_scores,
        "any_drift_detected": any_drift,
        "max_psi": max_psi,
    }


def _save_report(report: dict) -> Path:
    """Persist the report dict as a JSON file in DRIFT_DIR."""
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    path = DRIFT_DIR / f"drift_{report['report_date']}.json"
    path.write_text(json.dumps(report, indent=2))
    logger.info("Drift report saved: %s", path)
    return path


def _log_to_mlflow(report: dict) -> None:
    """Log per-feature PSI scores to the active production MLflow run."""
    load_dotenv()
    client = mlflow.MlflowClient()

    runs = client.search_runs(
        experiment_ids=["0"],
        filter_string="tags.stage = 'production'",
        max_results=1,
    )
    if not runs:
        logger.warning("No production MLflow run found — skipping MLflow logging.")
        return

    run_id = runs[0].info.run_id
    metrics = {
        "drift_max_psi": report["max_psi"],
        "drift_any_detected": int(report["any_drift_detected"]),
    }
    for feat, stats in report["features"].items():
        metrics[f"drift_psi_{feat}"] = stats["psi"]

    try:
        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.log_metrics(metrics)
        logger.info("PSI metrics logged to MLflow run %s", run_id[:8])
    except Exception:
        logger.exception("Failed to log metrics to MLflow run %s", run_id[:8])


# ── Main ───────────────────────────────────────────────────────────────────────

async def generate_report() -> dict | None:
    """End-to-end drift report pipeline. Returns the report dict or None if skipped."""
    load_dotenv()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable is not set.")

    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=3)
    try:
        reference = _load_reference()
        current = await _load_current(pool)
        if current is None:
            logger.warning("Skipping drift report — insufficient current data.")
            return None

        report = _run_evidently(reference, current)
        _save_report(report)
        _log_to_mlflow(report)
        logger.info("Drift report complete. Max PSI: %.4f", report["max_psi"])
        return report
    finally:
        await pool.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(generate_report())
    if result:
        print(f"Max PSI: {result['max_psi']:.4f}")
        print(f"Any drift detected: {result['any_drift_detected']}")
        for feat, stats in result["features"].items():
            status = "DRIFT" if stats["drifted"] else "OK"
            print(f"  {feat:<35} PSI={stats['psi']:.4f}  {status}")
