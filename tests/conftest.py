"""Shared pytest fixtures for unit and integration tests."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import polars as pl
import pytest

# ── sys.path bootstrap ────────────────────────────────────────────────────────
# Mirrors the pattern used in test_api.py so src/ and src/api/ bare-name
# imports resolve when running with the root venv.

_src_dir = str(Path(__file__).resolve().parents[1] / "src")
_api_dir = str(Path(__file__).resolve().parents[1] / "src" / "api")
for _p in (_src_dir, _api_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Section 1 — AUC regression threshold ─────────────────────────────────────

# Minimum AUC-ROC any candidate model must achieve on the temporal holdout
# before it is eligible for promotion. Matches the val AUC from model
# selection (0.7873) with a 5-point safety margin.
# Used by: tests/unit/test_model_regression.py
# Reference: notebooks/03_model_selection.ipynb — best val AUC 0.7873
MIN_AUC_THRESHOLD: float = 0.74


# ── Section 2 — Sample feature data ──────────────────────────────────────────

@pytest.fixture(scope="session")
def feature_cols() -> list[str]:
    """Return the canonical ordered feature column list from the registry."""
    from features.feature_registry import get_feature_cols
    return get_feature_cols()


@pytest.fixture(scope="session")
def cat_cols() -> list[str]:
    """Return categorical column names from the registry."""
    from features.feature_registry import get_categorical_cols
    return get_categorical_cols()


@pytest.fixture
def sample_application_df() -> pl.DataFrame:
    """One-row Polars DataFrame mimicking a raw application_train row.

    All required columns present with realistic values.
    FLAG_OWN_CAR and FLAG_OWN_REALTY are 'Y'/'N' strings (pre-pipeline state).
    """
    return pl.DataFrame({
        "SK_ID_CURR": [100001],
        "AMT_INCOME_TOTAL": [150000.0],
        "AMT_CREDIT": [450000.0],
        "AMT_ANNUITY": [20000.0],
        "AMT_GOODS_PRICE": [400000.0],
        "DAYS_BIRTH": [-12000],
        "DAYS_EMPLOYED": [-2000],
        "DAYS_ID_PUBLISH": [-3000],
        "DAYS_REGISTRATION": [-5000.0],
        "DAYS_LAST_PHONE_CHANGE": [-500.0],
        "EXT_SOURCE_1": [0.5],
        "EXT_SOURCE_2": [0.6],
        "EXT_SOURCE_3": [0.4],
        "CODE_GENDER": ["M"],
        "FLAG_OWN_CAR": ["N"],
        "FLAG_OWN_REALTY": ["Y"],
        "NAME_CONTRACT_TYPE": ["Cash loans"],
        "NAME_INCOME_TYPE": ["Working"],
        "NAME_EDUCATION_TYPE": ["Secondary / secondary special"],
        "NAME_FAMILY_STATUS": ["Married"],
        "NAME_HOUSING_TYPE": ["House / apartment"],
        "OCCUPATION_TYPE": ["Laborers"],
        "ORGANIZATION_TYPE": ["Business Entity Type 3"],
        "WEEKDAY_APPR_PROCESS_START": ["MONDAY"],
        "HOUSETYPE_MODE": ["block of flats"],
        "WALLSMATERIAL_MODE": ["Panel"],
        "EMERGENCYSTATE_MODE": ["No"],
        "FONDKAPREMONT_MODE": ["reg oper account"],
        "REGION_RATING_CLIENT": [2],
        "REGION_RATING_CLIENT_W_CITY": [2],
        "CNT_CHILDREN": [0],
        "CNT_FAM_MEMBERS": [2.0],
        "REGION_POPULATION_RELATIVE": [0.035792],
        "HOUR_APPR_PROCESS_START": [10],
        "OWN_CAR_AGE": [None],
        "LIVE_CITY_NOT_WORK_CITY": [0],
        "REG_CITY_NOT_WORK_CITY": [0],
        "REG_CITY_NOT_LIVE_CITY": [0],
        "LIVE_REGION_NOT_WORK_REGION": [0],
        "REG_REGION_NOT_LIVE_REGION": [0],
        "REG_REGION_NOT_WORK_REGION": [0],
        "FLAG_MOBIL": [1], "FLAG_EMP_PHONE": [1], "FLAG_WORK_PHONE": [0],
        "FLAG_CONT_MOBILE": [1], "FLAG_PHONE": [0], "FLAG_EMAIL": [0],
        "AMT_REQ_CREDIT_BUREAU_HOUR": [0.0], "AMT_REQ_CREDIT_BUREAU_DAY": [0.0],
        "AMT_REQ_CREDIT_BUREAU_WEEK": [0.0], "AMT_REQ_CREDIT_BUREAU_MON": [0.0],
        "AMT_REQ_CREDIT_BUREAU_QRT": [0.0], "AMT_REQ_CREDIT_BUREAU_YEAR": [1.0],
        "OBS_30_CNT_SOCIAL_CIRCLE": [0.0], "OBS_60_CNT_SOCIAL_CIRCLE": [0.0],
        "DEF_30_CNT_SOCIAL_CIRCLE": [0.0], "DEF_60_CNT_SOCIAL_CIRCLE": [0.0],
        # Document flags — all 0
        **{f"FLAG_DOCUMENT_{i}": [0] for i in range(2, 22)},
        # Building features — all None (will be imputed by feature pipeline)
        "APARTMENTS_AVG": [None], "APARTMENTS_MODE": [None], "APARTMENTS_MEDI": [None],
        "BASEMENTAREA_AVG": [None], "BASEMENTAREA_MODE": [None], "BASEMENTAREA_MEDI": [None],
        "YEARS_BEGINEXPLUATATION_AVG": [None], "YEARS_BEGINEXPLUATATION_MODE": [None],
        "YEARS_BEGINEXPLUATATION_MEDI": [None],
        "YEARS_BUILD_AVG": [None], "YEARS_BUILD_MODE": [None], "YEARS_BUILD_MEDI": [None],
        "COMMONAREA_AVG": [None], "COMMONAREA_MODE": [None], "COMMONAREA_MEDI": [None],
        "ELEVATORS_AVG": [None], "ELEVATORS_MODE": [None], "ELEVATORS_MEDI": [None],
        "ENTRANCES_AVG": [None], "ENTRANCES_MODE": [None], "ENTRANCES_MEDI": [None],
        "FLOORSMAX_AVG": [None], "FLOORSMAX_MODE": [None], "FLOORSMAX_MEDI": [None],
        "FLOORSMIN_AVG": [None], "FLOORSMIN_MODE": [None], "FLOORSMIN_MEDI": [None],
        "LANDAREA_AVG": [None], "LANDAREA_MODE": [None], "LANDAREA_MEDI": [None],
        "LIVINGAPARTMENTS_AVG": [None], "LIVINGAPARTMENTS_MODE": [None],
        "LIVINGAPARTMENTS_MEDI": [None],
        "LIVINGAREA_AVG": [None], "LIVINGAREA_MODE": [None], "LIVINGAREA_MEDI": [None],
        "NONLIVINGAPARTMENTS_AVG": [None], "NONLIVINGAPARTMENTS_MODE": [None],
        "NONLIVINGAPARTMENTS_MEDI": [None],
        "NONLIVINGAREA_AVG": [None], "NONLIVINGAREA_MODE": [None],
        "NONLIVINGAREA_MEDI": [None],
        "TOTALAREA_MODE": [None],
        "TARGET": [0],
    })


@pytest.fixture
def sample_feature_array(feature_cols: list[str], cat_cols: list[str]) -> np.ndarray:
    """Numpy array of zeros shaped (1, n_features) — minimal valid model input.

    Categorical columns encoded as -1.0 (OrdinalEncoder unknown_value).
    """
    arr = np.zeros((1, len(feature_cols)), dtype=np.float64)
    for i, col in enumerate(feature_cols):
        if col in cat_cols:
            arr[0, i] = -1.0
    return arr


# ── Section 3 — Mock infrastructure fixtures ──────────────────────────────────

@pytest.fixture
def mock_db_pool() -> AsyncMock:
    """AsyncMock asyncpg pool. execute() and fetch() return empty results."""
    pool = AsyncMock()
    pool.execute = AsyncMock(return_value=None)
    pool.fetch = AsyncMock(return_value=[])
    pool.fetchrow = AsyncMock(return_value=None)
    pool.fetchval = AsyncMock(return_value=0)
    pool.acquire = AsyncMock()
    return pool


@pytest.fixture
def mock_redis() -> AsyncMock:
    """AsyncMock Redis client. get() returns None (cache miss) by default."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.ping = AsyncMock(return_value=True)
    return redis


@pytest.fixture
def mock_bundle(
    feature_cols: list[str],
    cat_cols: list[str],
    sample_feature_array: np.ndarray,
) -> dict:
    """Minimal mock model bundle matching the joblib bundle structure from train.py.

    model.predict_proba returns 0.25 default probability.
    encoder.transform is an identity function — returns input unchanged.
    """
    bundle = {
        "model": MagicMock(),
        "encoder": MagicMock(),
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "params": {"objective": "binary"},
        "run_id": "testrun0123456789ab",
    }
    bundle["model"].predict_proba = MagicMock(return_value=np.array([[0.75, 0.25]]))
    bundle["encoder"].transform = MagicMock(side_effect=lambda x: x)
    return bundle


@pytest.fixture
def mock_shap_explainer(feature_cols: list[str]) -> MagicMock:
    """MagicMock SHAP TreeExplainer.

    shap_values() returns an array of evenly-spaced small values shape (1, n_features).
    """
    explainer = MagicMock()
    shap_vals = np.linspace(-0.1, 0.1, len(feature_cols)).reshape(1, -1)
    explainer.shap_values = MagicMock(return_value=shap_vals)
    return explainer


@pytest.fixture
def mock_app_state(
    mock_bundle: dict,
    mock_redis: AsyncMock,
    mock_db_pool: AsyncMock,
    mock_shap_explainer: MagicMock,
) -> MagicMock:
    """Composite app.state mock combining all infrastructure mocks.

    Matches the shape of the real app.state set in main.py lifespan.
    """
    state = MagicMock()
    state.bundle = mock_bundle
    state.redis_client = mock_redis
    state.db_pool = mock_db_pool
    state.shap_explainer = mock_shap_explainer
    state.encoder_lock = threading.Lock()
    state.model_meta = {
        "model_version": "testrun01",
        "mlflow_run_id": "testrun0123456789ab",
        "model_type": "lightgbm",
        "fair": False,
        "feature_count": len(mock_bundle["feature_cols"]),
        "promoted_at": None,
        "training_auc": 0.787,
    }
    return state


# ── Section 4 — MLflow mock ───────────────────────────────────────────────────

@pytest.fixture
def mock_mlflow_client() -> MagicMock:
    """MagicMock MLflow client. search_runs returns one fake production run."""
    client = MagicMock()
    fake_run = MagicMock()
    fake_run.info.run_id = "testrun0123456789ab"
    fake_run.data.metrics = {"auc_roc": 0.787, "gini": 0.574, "ks_stat": 0.412}
    fake_run.data.tags = {"stage": "production", "model_type": "lightgbm"}
    client.search_runs = MagicMock(return_value=[fake_run])
    client.get_run = MagicMock(return_value=fake_run)
    client.set_tag = MagicMock(return_value=None)
    return client
