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
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"

# Lazy-loaded DataFrames — read once on first request, reuse after.
_train_df: pl.DataFrame | None = None
_test_df: pl.DataFrame | None = None


def _load(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found at {path}")
    return pl.read_csv(path, infer_schema_length=10000)


def _get_train_df() -> pl.DataFrame:
    global _train_df
    if _train_df is None:
        _train_df = _load(_RAW_DIR / "application_train.csv")
    return _train_df


def _get_test_df() -> pl.DataFrame:
    global _test_df
    if _test_df is None:
        _test_df = _load(_RAW_DIR / "application_test.csv")
    return _test_df


def _clean(row: dict) -> dict:
    """Replace NaN floats with None so JSON serialisation is valid."""
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}


@router.get("/applicant/{sk_id_curr}")
def get_applicant(sk_id_curr: int) -> dict:
    """Return the raw application row, searching train then test CSV."""
    for loader in (_get_train_df, _get_test_df):
        try:
            df = loader()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        rows = df.filter(pl.col("SK_ID_CURR") == sk_id_curr)
        if not rows.is_empty():
            return _clean(rows[0].to_dicts()[0])

    raise HTTPException(status_code=404, detail=f"SK_ID_CURR {sk_id_curr} not found")
