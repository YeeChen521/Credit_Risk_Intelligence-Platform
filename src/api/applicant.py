"""Applicant lookup — GET /applicant/{sk_id_curr}."""

from __future__ import annotations

import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import polars as pl
from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["Applicant"])

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_APP_CSV = _PROJECT_ROOT / "data" / "raw" / "application_train.csv"

# Lazy-loaded DataFrame — read once on first request, reuse after.
_app_df: pl.DataFrame | None = None


def _get_app_df() -> pl.DataFrame:
    global _app_df
    if _app_df is None:
        if not _APP_CSV.exists():
            raise FileNotFoundError(f"application_train.csv not found at {_APP_CSV}")
        _app_df = pl.read_csv(_APP_CSV, infer_schema_length=10000)
    return _app_df


@router.get("/applicant/{sk_id_curr}")
def get_applicant(sk_id_curr: int) -> dict:
    """Return the raw application row for an applicant ID from the training CSV."""
    try:
        df = _get_app_df()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    rows = df.filter(pl.col("SK_ID_CURR") == sk_id_curr)
    if rows.is_empty():
        raise HTTPException(status_code=404, detail=f"SK_ID_CURR {sk_id_curr} not found")

    row = rows[0].to_dicts()[0]
    # Replace NaN floats with None so JSON serialisation doesn't produce 'NaN'
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}
