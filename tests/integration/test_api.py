"""Integration smoke tests for the credit risk scoring API.

Tests run in-process via FastAPI's ASGI transport — no live server, Postgres,
or Redis required.  The prediction pipeline is replaced by a deterministic mock
so tests focus on HTTP routing, Pydantic validation, and response shape.

To run against a deployed server instead, start the API first and then:
    API_BASE_URL=http://localhost:8000 pytest tests/integration/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap: mirror the sys.path setup from src/api/__init__.py so that
# bare-name imports (schemas, predict, …) resolve when running with the root
# venv that lacks the project's editable install.
_src_dir = str(Path(__file__).resolve().parents[2] / "src")
_api_dir = str(Path(__file__).resolve().parents[2] / "src" / "api")
for _p in (_src_dir, _api_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datetime import datetime, timezone
from unittest.mock import patch

import httpx
import pytest

from schemas import PredictionResponse, SHAPFeature

pytestmark = pytest.mark.asyncio


# ── Mock factory ──────────────────────────────────────────────────────────────

def _make_response(sk_id: int, cached: bool) -> PredictionResponse:
    """Return a deterministic PredictionResponse for the given applicant."""
    return PredictionResponse(
        application_id=sk_id,
        default_probability=0.25,
        risk_tier="LOW",
        top_shap_features=[
            SHAPFeature(
                feature_name=f"FEATURE_{i:02d}",
                shap_value=round(i * 0.01, 4),
                feature_value=float(i),
                description="",
            )
            for i in range(10)
        ],
        model_version="testrun01",
        cached=cached,
        predicted_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_payload() -> dict:
    return {
        "application": {
            "SK_ID_CURR": 999999,
            "AMT_INCOME_TOTAL": 150000.0,
            "AMT_CREDIT": 450000.0,
            "AMT_ANNUITY": 20000.0,
            "AMT_GOODS_PRICE": 400000.0,
            "DAYS_BIRTH": -12000,
            "DAYS_EMPLOYED": -2000,
            "DAYS_ID_PUBLISH": -3000,
            "DAYS_REGISTRATION": -5000.0,
            "DAYS_LAST_PHONE_CHANGE": -500.0,
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.6,
            "EXT_SOURCE_3": 0.4,
            "CODE_GENDER": "M",
            "FLAG_OWN_CAR": "N",
            "FLAG_OWN_REALTY": "Y",
            "NAME_CONTRACT_TYPE": "Cash loans",
            "NAME_INCOME_TYPE": "Working",
            "NAME_EDUCATION_TYPE": "Secondary / secondary special",
            "NAME_FAMILY_STATUS": "Married",
            "NAME_HOUSING_TYPE": "House / apartment",
            "REGION_RATING_CLIENT": 2,
            "CNT_CHILDREN": 0,
            "CNT_FAM_MEMBERS": 2.0,
        },
        "bureau": [],
        "bureau_balance": [],
        "previous_applications": [],
        "pos_cash": [],
        "installments": [],
        "credit_card": [],
    }


@pytest.fixture
async def client():
    """Function-scoped ASGI client.

    Uses httpx.ASGITransport so tests hit the real FastAPI routing and Pydantic
    validation without needing a live server, Postgres, or Redis.

    predict._run_prediction is patched with a deterministic mock that:
    - returns cached=False on the first call for a given SK_ID_CURR
    - returns cached=True on any subsequent call with the same SK_ID_CURR
    (simulating Redis cache behaviour within a single test function)
    """
    from main import app  # imported here so sys.path bootstrap above is in effect

    # /model/info reads app.state.model_meta; without it the endpoint returns 503
    app.state.model_meta = {
        "model_version": "testrun01",
        "mlflow_run_id": "testrun0123456789ab",
        "model_type": "LightGBM",
        "fair": True,
        "feature_count": 100,
        "promoted_at": None,
        "training_auc": 0.75,
    }

    # Fresh per-test set — a second call with the same SK_ID within one test
    # will find it already in _seen and return cached=True.
    _seen: set[int] = set()

    async def _mock_run_prediction(request, app_state):
        sk_id = request.application.SK_ID_CURR
        cached = sk_id in _seen
        _seen.add(sk_id)
        return _make_response(sk_id, cached)

    with patch("predict._run_prediction", side_effect=_mock_run_prediction):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            yield c


# ── Tests ─────────────────────────────────────────────────────────────────────


async def test_health(client: httpx.AsyncClient) -> None:
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_model_info(client: httpx.AsyncClient) -> None:
    r = await client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert "model_version" in body
    assert body["feature_count"] > 0


async def test_predict_returns_valid_response(
    client: httpx.AsyncClient, minimal_payload: dict
) -> None:
    r = await client.post("/predict", json=minimal_payload)
    assert r.status_code == 200
    body = r.json()

    assert body["application_id"] == 999999
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_tier"] in ("LOW", "MEDIUM", "HIGH")
    assert len(body["top_shap_features"]) == 10
    for feat in body["top_shap_features"]:
        assert isinstance(feat["feature_name"], str)
        assert isinstance(feat["shap_value"], float)
        assert isinstance(feat["feature_value"], float)
    assert body["cached"] is False


async def test_predict_cache_hit(
    client: httpx.AsyncClient, minimal_payload: dict
) -> None:
    # Two calls within the same test share the same _seen set in the fixture,
    # so the second call returns cached=True automatically.
    first = (await client.post("/predict", json=minimal_payload)).json()
    second = (await client.post("/predict", json=minimal_payload)).json()

    assert second["cached"] is True
    assert abs(first["default_probability"] - second["default_probability"]) < 1e-6


async def test_predict_batch(
    client: httpx.AsyncClient, minimal_payload: dict
) -> None:
    def _with_id(sk_id: int) -> dict:
        payload = dict(minimal_payload)
        payload["application"] = {**minimal_payload["application"], "SK_ID_CURR": sk_id}
        return payload

    batch = [_with_id(999998), _with_id(999997)]
    r = await client.post("/predict/batch", json=batch)
    assert r.status_code == 200
    results = r.json()
    assert len(results) == 2
    for item in results:
        assert 0.0 <= item["default_probability"] <= 1.0
        assert item["risk_tier"] in ("LOW", "MEDIUM", "HIGH")


async def test_predict_batch_limit(
    client: httpx.AsyncClient, minimal_payload: dict
) -> None:
    oversized = [minimal_payload] * 101
    r = await client.post("/predict/batch", json=oversized)
    assert r.status_code == 400


async def test_missing_application_fields(client: httpx.AsyncClient) -> None:
    sparse_payload = {
        "application": {"SK_ID_CURR": 999996},
        "bureau": [],
        "bureau_balance": [],
        "previous_applications": [],
        "pos_cash": [],
        "installments": [],
        "credit_card": [],
    }
    r = await client.post("/predict", json=sparse_payload)
    assert r.status_code == 200
