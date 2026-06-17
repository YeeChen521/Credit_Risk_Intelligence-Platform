"""Health check and model info routers."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from schemas import HealthResponse, ModelInfoResponse

health_router = APIRouter(tags=["Health"])
model_router = APIRouter(tags=["Model"])


@health_router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Return API liveness status."""
    return HealthResponse(status="ok", timestamp=datetime.now(timezone.utc))


@model_router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(request: Request) -> ModelInfoResponse:
    """Return metadata about the currently loaded production model bundle."""
    model_meta: dict | None = getattr(request.app.state, "model_meta", None)
    if model_meta is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check startup logs.")

    return ModelInfoResponse(
        model_version=model_meta["model_version"],
        mlflow_run_id=model_meta["mlflow_run_id"],
        model_type=model_meta["model_type"],
        fair=model_meta["fair"],
        feature_count=model_meta["feature_count"],
        promoted_at=model_meta.get("promoted_at"),
        training_auc=model_meta.get("training_auc"),
    )
