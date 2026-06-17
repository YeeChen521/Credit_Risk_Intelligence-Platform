"""FastAPI application factory and lifespan for the credit risk scoring API."""

from __future__ import annotations

import glob
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import joblib
import shap
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import redis.asyncio as aioredis

from db import create_db_pool, init_db
from health import health_router, model_router
from middleware import LoggingMiddleware, register_exception_handler
from portfolio import router as portfolio_router
from predict import router as predict_router

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


# ── Bundle loading ────────────────────────────────────────────────────────────

def _find_production_bundle() -> Path:
    """Locate the lgb_production_*.joblib bundle in models/.

    Raises FileNotFoundError if no bundle exists (user needs to run promote.py).
    If multiple bundles are found, warns and returns the most recently modified.
    """
    matches = glob.glob(str(MODELS_DIR / "lgb_production_*.joblib"))
    if not matches:
        raise FileNotFoundError(
            f"No production bundle found in {MODELS_DIR}. "
            "Run `python -m src.models.promote --run-id <run_id>` to promote a model."
        )
    if len(matches) > 1:
        logger.warning(
            "Multiple production bundles found (%d). Using the most recently modified.",
            len(matches),
        )
    return Path(max(matches, key=lambda p: Path(p).stat().st_mtime))


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model bundle, init DB pool and Redis on startup. Clean up on shutdown."""

    # 1 — Load environment variables
    load_dotenv()

    # 2 — Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # 3 — Load production bundle
    bundle_path = _find_production_bundle()
    logger.info("Loading bundle: %s", bundle_path.name)
    bundle = joblib.load(bundle_path)
    app.state.bundle = bundle
    logger.info(
        "Bundle loaded — run_id: %s, features: %d",
        bundle["run_id"],
        len(bundle["feature_cols"]),
    )

    # 4 — SHAP TreeExplainer (initialised once; reused per request via app.state)
    app.state.shap_explainer = shap.TreeExplainer(bundle["model"])
    logger.info("SHAP TreeExplainer ready")

    # 5 — Encoder lock (serialises OrdinalEncoder.transform across concurrent requests)
    app.state.encoder_lock = threading.Lock()

    # 6 — Model metadata (exposed by GET /model/info)
    app.state.model_meta = {
        "model_version": bundle["run_id"][:8],
        "mlflow_run_id": bundle["run_id"],
        "model_type": bundle.get("params", {}).get("objective", "lightgbm"),
        "fair": "group_thresholds" in bundle,
        "feature_count": len(bundle["feature_cols"]),
        "promoted_at": None,
        "training_auc": None,
    }

    # 7 — PostgreSQL connection pool
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL environment variable is not set")
    app.state.db_pool = await create_db_pool()
    await init_db(app.state.db_pool)
    logger.info("PostgreSQL pool ready")

    # 8 — Redis client
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise RuntimeError("REDIS_URL environment variable is not set")
    app.state.redis_client = aioredis.from_url(redis_url, decode_responses=True)
    await app.state.redis_client.ping()
    logger.info("Redis client ready")

    yield  # ← application runs here

    # Shutdown — close connections gracefully
    await app.state.db_pool.close()
    await app.state.redis_client.aclose()
    logger.info("Connections closed")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Credit Risk Intelligence API",
    description="Loan default probability scoring with SHAP explainability",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
register_exception_handler(app)

app.include_router(predict_router)
app.include_router(health_router)
app.include_router(model_router)
app.include_router(portfolio_router)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    # Uvicorn reload mode spawns a subprocess with a fresh sys.path.
    # The subprocess needs to import "src.api.main", so the project root must
    # be on its sys.path.  The only reliable way is via PYTHONPATH, which
    # subprocess processes inherit from their parent's os.environ.
    _root = str(Path(__file__).resolve().parents[2])
    _existing = os.environ.get("PYTHONPATH", "")
    if _root not in _existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = f"{_root}{os.pathsep}{_existing}" if _existing else _root

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
