# Credit Risk Intelligence Platform

An end-to-end ML system that predicts loan default risk, explains every prediction at the individual borrower level via SHAP, and surfaces actionable signals through a FastAPI scoring service and Streamlit analyst dashboard.

Built on the **Home Credit Default Risk** dataset (350K+ applications). Designed to meet **regulatory-grade explainability** (GDPR right to explanation) and banking-sector **auditability** requirements.

## Architecture

The system is organized into three sequential phases that map directly to the folder structure.

```
Home Credit CSVs
      │
      ▼
┌─────────────────────────────────┐
│   Phase 1 — Data & ML           │
│   DuckDB · Polars · XGBoost     │
│   LightGBM · MLflow · SHAP      │
└──────────────┬──────────────────┘
               │ scored feature Parquet
               ▼
┌─────────────────────────────────┐
│   Phase 2 — API & Dashboard     │
│   FastAPI · PostgreSQL · Redis  │
│   Streamlit · Great Expectations│
└──────────────┬──────────────────┘
               │ Docker Compose
               ▼
┌─────────────────────────────────┐
│   Phase 3 — MLOps               │
│   GitHub Actions · Evidently AI │
│   Railway                       │
└─────────────────────────────────┘
```

### Key design decisions

- **Immutable audit trail**: PostgreSQL stores every scoring request before the API responds (required in regulated financial ML).
- **Low-latency lookups**: Redis caches repeated requests for the same `borrower_id`.
- **Explainability persistence**: SHAP waterfall inputs/values are computed at score time and stored alongside the prediction (never regenerated on read).
- **No random splits**: temporal (walk-forward) validation only (see [Validation rules](#validation-rules)).
- **Model governance**: MLflow tracks runs; only models promoted to **`production`** are served by the API.

## Repository / folder responsibilities

```
data/
  raw/                       # Original Home Credit CSVs — never modified
  processed/                 # Feature Parquet tables output by src/features/

notebooks/
  01_eda.ipynb               # Exploratory analysis, missing value report
  02_feature_exploration.ipynb

src/
  features/
    build_features.py        # Full pipeline: all aggregation functions + main()
    feature_registry.py      # Canonical feature definitions, get_feature_cols()

  models/
    train.py                 # Trains LightGBM, saves lgb_staging_*.joblib to models/
    train_fair.py            # AIF360 Reweighing + per-group threshold calibration
    evaluate.py              # AUC-ROC, Gini, KS statistic + fairness audit
    promote.py               # Gates on fairness; promotes run; renames bundle

  api/
    main.py                  # FastAPI app, lifespan, bundle loading, DB + Redis init
    schemas.py               # Pydantic v2 request/response models
    predict.py               # POST /predict, POST /predict/batch
    portfolio.py             # GET /portfolio/summary|predictions|flagged
    health.py                # GET /health, GET /model/info
    db.py                    # asyncpg pool, predictions table, log_prediction()
    middleware.py            # Request logging, global exception handler

  dashboard/
    app.py                   # Entry point, sidebar, API status, navigation
    pages/
      borrower.py            # Risk gauge + SHAP waterfall per applicant
      portfolio.py           # Tier breakdown, trend, flagged table + CSV export
      model_health.py        # Model info cards + Evidently PSI drift tab

  monitoring/
    drift_report.py          # Evidently DataDriftPreset → drift_YYYY-MM-DD.json
    alert.py                 # PSI threshold check → Slack or PagerDuty webhook

tests/
  unit/
  integration/
    test_api.py              # ASGI transport smoke tests (no live server needed)
  conftest.py                # Shared fixtures: mock bundle, DB, Redis, SHAP, MLflow

models/                      # Local joblib bundles (gitignored except .gitkeep)
Dockerfile                   # API image
Dockerfile.dashboard         # Dashboard image
railway.toml                 # Railway deployment config

expectations/                # Great Expectations suites
  application_suite.json
  bureau_suite.json

infra/
  docker-compose.yml         # API, dashboard, MLflow, Postgres, Redis
  .github/workflows/ci.yml   # ruff → pytest → docker build → push to GHCR

mlruns/                      # MLflow local tracking (gitignored)
.env.example                 # Template — copy to .env, never commit .env
pyproject.toml               # Ruff/pytest config + dependencies
```

## Quickstart

### Local development

```bash
# Copy and fill in secrets
cp .env.example .env

# Start full stack (API, dashboard, MLflow UI, Postgres, Redis)
docker compose up

# API:        http://localhost:8000
# Dashboard:  http://localhost:8501
# MLflow UI:  http://localhost:5000
```

### Feature pipeline and training

```bash
# Build feature table from raw CSVs
python -m src.features.build_features

# Train model (logs to MLflow, saves to staging)
python -m src.models.train

# Evaluate a run
python -m src.models.evaluate --run-id <mlflow_run_id>

# Fair model variant (AIF360)
python -m src.models.train --fair

# Promote a run to production
python -m src.models.promote --run-id <mlflow_run_id>
```

### Testing and linting

```bash
# Lint (matches CI gate)
ruff check src/ tests/

# Run all tests
pytest

# Unit tests (fast)
pytest tests/unit/

# Integration tests (ASGI transport — no live server or Docker needed)
pytest tests/integration/
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Score a single applicant, returns probability + top 10 SHAP features |
| POST | `/predict/batch` | Score up to 100 applicants concurrently |
| GET | `/portfolio/summary` | Risk tier breakdown + avg probability (`?period=all\|6m\|1y`) |
| GET | `/portfolio/predictions` | Paginated prediction history with time filter |
| GET | `/portfolio/flagged` | High-risk predictions only, downloadable as CSV |
| GET | `/health` | Liveness check |
| GET | `/model/info` | Current model version, feature count, AUC, fair model flag |

### Score a single applicant (curl example)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "application": {
      "SK_ID_CURR": 100001,
      "AMT_INCOME_TOTAL": 150000.0,
      "AMT_CREDIT": 450000.0,
      "CODE_GENDER": "M",
      "FLAG_OWN_CAR": "N",
      "FLAG_OWN_REALTY": "Y",
      "DAYS_BIRTH": -12000,
      "DAYS_EMPLOYED": -2000,
      "EXT_SOURCE_1": 0.5,
      "EXT_SOURCE_2": 0.6,
      "EXT_SOURCE_3": 0.4
    },
    "bureau": [],
    "bureau_balance": [],
    "previous_applications": [],
    "pos_cash": [],
    "installments": [],
    "credit_card": []
  }'
```

Response shape:
```json
{
  "application_id": 100001,
  "default_probability": 0.182,
  "risk_tier": "LOW",
  "top_shap_features": [
    {"feature_name": "EXT_SOURCE_2", "shap_value": -0.312, "feature_value": 0.6, "description": "..."},
    "..."
  ],
  "model_version": "a1b2c3d4",
  "cached": false,
  "predicted_at": "2024-01-15T10:30:00Z"
}
```

## Drift monitoring

```bash
# Generate weekly PSI drift report
python -m src.monitoring.drift_report

# Check alert threshold and fire webhook if exceeded
python -m src.monitoring.alert
```

Drift reports are stored in `data/processed/drift/` with an ISO date suffix.

## Coding conventions

### Python style

- **Formatter/linter**: `ruff` (config in `pyproject.toml`). No other formatters.
- **Type hints**: required for all function parameters and return values.
- **Docstrings**: public functions require at least a one-line docstring.
- **Line length**: 100 characters.
- **Imports**: no `import *`.

### Data layer

- **Polars only** for feature engineering in `src/features/` (no pandas).
- **Raw data is read-only**: `data/raw/` is never modified; pipelines write to `data/processed/`.
- **Parquet output**: snappy compression.
- **Feature registry required**: every derived feature must be registered in `feature_registry.py` with a plain-English description (used by the explanation layer).

### API layer

- **Pydantic boundary**: all request/response shapes are Pydantic v2 models in `schemas.py` (no raw dicts).
- **asyncpg directly**: all SQL lives in `db.py` and `portfolio.py`. No raw SQL outside those files.
- **Audit trail is mandatory**: every scoring request is written to PostgreSQL *before* returning.
- **SHAP persisted**: computed at score time and stored with the request record; never recomputed on read.
- **Errors**: structured JSON with `detail`; Great Expectations failures return HTTP 422.

### ML layer

- **MLflow logging required**: parameters, metrics (AUC-ROC, Gini, KS), and model artifact for every run.
- **Centralized evaluation**: metrics computed in `evaluate.py`, not inline in `train.py`.
- **Staging vs production tags**:
  - New runs should set `stage: staging`
  - Only `promote.py` sets `stage: production`
- **Optuna**: log trials as nested MLflow runs under the parent training run.

### Testing

- **Unit tests**: mock external deps (DB, Redis, MLflow); no network calls.
- **Integration tests**: use `httpx.ASGITransport` — no live server or Docker required.
- **Regression gate**: any model-touching change must satisfy the AUC floor defined in `tests/conftest.py`.
- **Data contracts**: add/extend Great Expectations suites when adding new data sources.

## Development workflow

### Branch strategy

- `main`: production-ready code only. Protected; requires CI + one review.
- `dev`: integration branch; feature branches merge here first.
- `feature/<name>`: branch from `dev`, PR back to `dev`.
- `hotfix/<name>`: branch from `main`, PR back to `main` and `dev`.

### CI pipeline (GitHub Actions)

```
ruff check → pytest → docker build → push to GHCR
```

All gates must pass; failing lint/tests blocks the Docker build. GHCR push runs on merges to `main`.

### Model promotion workflow

1. Train and evaluate a candidate run (`python -m src.models.train`)
2. Compare AUC-ROC, Gini, and KS against current production in the MLflow UI
3. If metrics improve (or match within tolerance) and fairness audit passes, run `promote.py`
4. The API loads the model tagged `production` on startup—redeploy the API container to pick up the new model
5. Archive the previous production run tag (`stage: archived`)

## Auth (placeholder)

JWT validation lives in `src/api/auth.py`. Current implementation accepts a shared secret from `.env` (`API_SECRET_KEY`).

To integrate with an identity provider, replace `verify_token` with OAuth2/OIDC token verification logic. All routers depend on `verify_token`.

## Secrets management

- Secrets are loaded from `.env` via `python-dotenv`. Never hardcode credentials.
- `.env` is gitignored; `.env.example` contains all required keys with blank/dummy values.
- In Railway (production), set secrets as environment variables (no `.env` deployed).

Required keys:

- `DATABASE_URL`
- `REDIS_URL`
- `API_SECRET_KEY`
- `MLFLOW_TRACKING_URI`
- `ALERT_WEBHOOK_URL`

## Railway deployment

`railway.toml` at the project root configures the API service.

Steps:
1. Create a new Railway project
2. Add PostgreSQL and Redis plugins — connection URLs are injected automatically
3. Connect this GitHub repo
4. Set `MLFLOW_TRACKING_URI`, `API_SECRET_KEY`, and `ALERT_WEBHOOK_URL` in the Railway dashboard
5. Railway auto-deploys on push to `main`

For the dashboard: create a second Railway service, set `dockerfile` to `Dockerfile.dashboard`
and add `API_BASE_URL` pointing to the API service's internal Railway URL.

## Validation rules

**Temporal splits are mandatory.** The dataset must always be split by time, not randomly. Random splits inflate AUC by 5–8 points on typical credit datasets and lead to production underperformance.

Recommended split for the Home Credit dataset:

- Train: applications from months 1–18
- Validation: months 19–22
- Test (held out): months 23–24

Never use `train_test_split` with `shuffle=True` on this dataset.

## Fairness audit

Every evaluation includes a fairness check across age buckets (in `evaluate.py`).

Reported metrics:

- equal opportunity difference
- disparate impact ratio

Threshold: disparate impact ratio must stay above **0.80** (four-fifths rule). Runs failing this must not be promoted without documented justification.

Fairness-constrained retraining is implemented in `src/models/train_fair.py` using AIF360 Reweighing (pre-processing) and per-group threshold calibration (post-processing). Run with: `python -m src.models.train --fair`

## Drift alerting

Evidently AI generates weekly PSI (Population Stability Index) reports comparing distributions of new applicants vs training data for key features (e.g., income, age, bureau signals).

- PSI < 0.1: no action
- PSI 0.1–0.2: log warning, flag in Model Health dashboard tab
- PSI > 0.2: trigger alert via `src/monitoring/alert.py` → `ALERT_WEBHOOK_URL` (Slack/PagerDuty)
