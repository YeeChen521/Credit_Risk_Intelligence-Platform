"""Portfolio analytics router: aggregate and paginated views over the predictions table."""

from __future__ import annotations

import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import logging

from fastapi import APIRouter, HTTPException, Query, Request

from schemas import (
    PortfolioListResponse,
    PortfolioSummaryResponse,
    PredictionRecord,
    RiskTierCount,
)

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])


# ── SQL helpers ───────────────────────────────────────────────────────────────

# Whitelisted fragments — never derived from unsanitised user input.
_PERIOD_FILTERS: dict[str, str] = {
    "all": "",
    "6m": "AND predicted_at >= NOW() - INTERVAL '6 months'",
    "1y": "AND predicted_at >= NOW() - INTERVAL '1 year'",
}


def _date_filter(period: str) -> str:
    """Return the SQL WHERE clause fragment for the given period.

    Raises ValueError for unrecognised period values so callers can convert
    to a 400 response before touching the database.
    """
    if period not in _PERIOD_FILTERS:
        raise ValueError(
            f"Invalid period '{period}'. Must be one of: {', '.join(_PERIOD_FILTERS)}."
        )
    return _PERIOD_FILTERS[period]


def _get_pool(request: Request):
    """Extract db_pool from app state or raise 503."""
    pool = getattr(request.app.state, "db_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="Database pool not available.")
    return pool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/summary", response_model=PortfolioSummaryResponse)
async def get_summary(
    request: Request,
    period: str = Query(default="all", description="Time window: all | 6m | 1y"),
) -> PortfolioSummaryResponse:
    """Return tier breakdown and aggregate statistics for the prediction portfolio."""
    try:
        date_clause = _date_filter(period)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    pool = _get_pool(request)

    tier_query = (
        "SELECT risk_tier, COUNT(*) AS count, AVG(default_probability) AS avg_prob "
        "FROM predictions WHERE 1=1 "
        + date_clause
        + " GROUP BY risk_tier"
    )
    totals_query = (
        "SELECT COUNT(*) AS total, COALESCE(AVG(default_probability), 0.0) AS avg_prob "
        "FROM predictions WHERE 1=1 "
        + date_clause
    )

    async with pool.acquire() as conn:
        tier_rows = await conn.fetch(tier_query)
        totals_row = await conn.fetchrow(totals_query)

    tier_breakdown = [
        RiskTierCount(
            risk_tier=row["risk_tier"],
            count=row["count"],
            avg_probability=float(row["avg_prob"]),
        )
        for row in tier_rows
    ]
    high_risk_count = next(
        (t.count for t in tier_breakdown if t.risk_tier == "HIGH"), 0
    )

    return PortfolioSummaryResponse(
        total_predictions=totals_row["total"],
        tier_breakdown=tier_breakdown,
        high_risk_count=high_risk_count,
        avg_default_probability=float(totals_row["avg_prob"]),
        period=period,
    )


@router.get("/predictions", response_model=PortfolioListResponse)
async def list_predictions(
    request: Request,
    period: str = Query(default="all", description="Time window: all | 6m | 1y"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> PortfolioListResponse:
    """Return a paginated list of prediction records ordered by predicted_at DESC."""
    try:
        date_clause = _date_filter(period)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    pool = _get_pool(request)

    rows_query = (
        "SELECT id, application_id, default_probability, risk_tier, "
        "model_version, predicted_at, cached "
        "FROM predictions WHERE 1=1 "
        + date_clause
        + " ORDER BY predicted_at DESC LIMIT $1 OFFSET $2"
    )
    count_query = (
        "SELECT COUNT(*) AS total FROM predictions WHERE 1=1 " + date_clause
    )

    async with pool.acquire() as conn:
        rows = await conn.fetch(rows_query, limit, offset)
        count_row = await conn.fetchrow(count_query)

    return PortfolioListResponse(
        predictions=[PredictionRecord(**dict(row)) for row in rows],
        total=count_row["total"],
        period=period,
    )


@router.get("/flagged", response_model=PortfolioListResponse)
async def list_flagged(
    request: Request,
    period: str = Query(default="all", description="Time window: all | 6m | 1y"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> PortfolioListResponse:
    """Return a paginated list of HIGH-risk predictions ordered by predicted_at DESC."""
    try:
        date_clause = _date_filter(period)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    pool = _get_pool(request)

    rows_query = (
        "SELECT id, application_id, default_probability, risk_tier, "
        "model_version, predicted_at, cached "
        "FROM predictions WHERE 1=1 "
        + date_clause
        + " AND risk_tier = 'HIGH'"
        + " ORDER BY predicted_at DESC LIMIT $1 OFFSET $2"
    )
    count_query = (
        "SELECT COUNT(*) AS total FROM predictions WHERE 1=1 "
        + date_clause
        + " AND risk_tier = 'HIGH'"
    )

    async with pool.acquire() as conn:
        rows = await conn.fetch(rows_query, limit, offset)
        count_row = await conn.fetchrow(count_query)

    return PortfolioListResponse(
        predictions=[PredictionRecord(**dict(row)) for row in rows],
        total=count_row["total"],
        period=period,
    )
