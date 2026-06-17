"""Entry point: loads full feature table, trains LightGBM, logs run to MLflow."""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import polars as pl
from dotenv import load_dotenv
from sklearn.preprocessing import OrdinalEncoder

from src.features.feature_registry import (
    ID_COL,
    TARGET_COL,
    get_categorical_cols,
    get_feature_cols,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Best params from 03_model_selection.ipynb — LightGBM Optuna study (30 trials, val AUC 0.7873)
# n_estimators is the early-stopping-determined iteration count from the held-out validation run.
# scale_pos_weight=11 was fixed (not tuned) to match the class imbalance ratio (~1:11).
BEST_PARAMS: dict[str, object] = {
    "max_depth": 3,
    "num_leaves": 7,
    "min_child_samples": 83,
    "min_split_gain": 0.8660993973100154,
    "reg_lambda": 1.8738013127891788,
    "reg_alpha": 4.155310054263944,
    "subsample": 0.6750591731140013,
    "colsample_bytree": 0.6014708421568282,
    "learning_rate": 0.05523743599687083,
    "scale_pos_weight": 11,
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 1532,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

# Params that are LightGBM internals, not meaningful to log as hyperparameters
_SKIP_LOG = {"verbose", "metric", "objective", "n_jobs", "random_state"}


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load and sort full training parquet, return (X_dataframe, y_array)."""
    train = pl.read_parquet(PROCESSED_DIR / "train_complete").sort(ID_COL)
    features = get_feature_cols()
    X = train[features].to_pandas()
    y = train[TARGET_COL].to_numpy()
    return X, y


def encode_categoricals(X: pd.DataFrame) -> tuple[np.ndarray, OrdinalEncoder, list[str]]:
    """Ordinal-encode categorical columns; return (X_numpy, fitted_encoder, cat_col_names)."""
    cat_cols = [c for c in get_categorical_cols() if c in X.columns]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_enc = X.copy()
    X_enc[cat_cols] = encoder.fit_transform(X[cat_cols])
    return X_enc.to_numpy(), encoder, cat_cols


def train(
    params: dict[str, object] | None = None,
    sample_weight: np.ndarray | None = None,
) -> str:
    """Train LightGBM on the full dataset, log to MLflow, return MLflow run_id.

    Metrics (AUC-ROC, Gini, KS) are intentionally not computed here — see evaluate.py.
    The run is tagged stage=staging; use promote.py to move it to production.
    """
    load_dotenv()
    if params is None:
        params = BEST_PARAMS

    X, y = load_data()
    X_np, encoder, cat_cols = encode_categoricals(X)
    feature_cols = get_feature_cols()

    MODELS_DIR.mkdir(exist_ok=True)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        mlflow.set_tags({"stage": "staging", "model_type": "lightgbm"})
        mlflow.log_params({k: v for k, v in params.items() if k not in _SKIP_LOG})
        mlflow.log_params({"n_features": X_np.shape[1], "n_train_rows": int(X_np.shape[0])})

        model = lgb.LGBMClassifier(**params)
        model.fit(X_np, y, sample_weight=sample_weight)

        mlflow.lightgbm.log_model(model.booster_, artifact_path="model")

        # Local bundle: model + encoder + metadata needed by evaluate.py
        bundle_path = MODELS_DIR / f"lgb_staging_{run_id[:8]}.joblib"
        joblib.dump(
            {
                "model": model,
                "encoder": encoder,
                "feature_cols": feature_cols,
                "cat_cols": cat_cols,
                "params": params,
                "run_id": run_id,
            },
            bundle_path,
        )
        mlflow.log_artifact(str(bundle_path), artifact_path="bundle")

        print(f"MLflow run_id : {run_id}")
        print(f"Local bundle  : {bundle_path}")

    return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LightGBM credit-risk model.")
    parser.add_argument(
        "--fair", action="store_true",
        help="Fairness-constrained training via AIF360 Reweighing (see train_fair.py)",
    )
    args = parser.parse_args()

    if args.fair:
        from src.models.train_fair import train_fair
        train_fair()
    else:
        train()
