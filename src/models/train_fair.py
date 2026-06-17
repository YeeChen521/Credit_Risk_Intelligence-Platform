"""Fairness-constrained LightGBM training.

Two-stage approach:
  Stage 1 — Pre-processing: AIF360 Reweighing (Kamiran & Calders 2012)
    Reweighs training samples so that P(Y | age_group) becomes independent of
    age_group.  Protected attribute: age_binary (0 = age < 45, 1 = age >= 45).

  Stage 2 — Post-processing: per-group threshold calibration
    Finds per age-bucket decision thresholds on the temporal holdout that
    equalise selection rates across the three buckets used by evaluate.py.
    Thresholds are stored in the bundle; evaluate.py applies them automatically.

Usage:
    python -m src.models.train_fair
    python -m src.models.train --fair   (delegates here)
"""

from __future__ import annotations

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import polars as pl
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
from dotenv import load_dotenv
from sklearn.metrics import roc_curve

from src.features.feature_registry import ID_COL, TARGET_COL, get_feature_cols
from src.models.train import (
    BEST_PARAMS,
    MODELS_DIR,
    PROCESSED_DIR,
    _SKIP_LOG,
    encode_categoricals,
)

# AIF360 binary age split — younger applicants are the unprivileged group
# (less credit history → model tends to over-flag them as risky)
AGE_SPLIT = 45
_PRIV: dict[str, float] = {"age_binary": 1.0}   # mature (>= 45)
_UNPRIV: dict[str, float] = {"age_binary": 0.0}  # young  (< 45)

# Three-bucket split matches evaluate.py — used for threshold calibration
AGE_BINS = [0, 35, 55, 150]
AGE_LABELS = ["young_lt35", "middle_35_55", "senior_gt55"]

EVAL_HOLDOUT_FRAC = 0.20


def _aif360_reweigh(y: np.ndarray, age_binary: np.ndarray) -> np.ndarray:
    """Compute AIF360 Reweighing sample weights for the age_binary protected attribute.

    Returns a normalised weight array (sum == n) that balances the label x age-group
    joint distribution so P(Y=y | A=a) becomes the same for both age groups.
    """
    df = pd.DataFrame({
        "f0": np.zeros(len(y), dtype=np.float64),  # AIF360 requires >= 1 feature column
        "age_binary": age_binary.astype(np.float64),
        "TARGET": y.astype(np.float64),
    })
    dataset = BinaryLabelDataset(
        df=df,
        label_names=["TARGET"],
        protected_attribute_names=["age_binary"],
        favorable_label=0.0,    # no default = good outcome
        unfavorable_label=1.0,
    )
    rw = Reweighing(unprivileged_groups=[_UNPRIV], privileged_groups=[_PRIV])
    rw.fit(dataset)
    return rw.transform(dataset).instance_weights


def _youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Return (threshold, selection_rate) at the Youden J maximum."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[idx])
    return threshold, float((y_score >= threshold).mean())


def _calibrate_group_thresholds(
    y_score: np.ndarray,
    days_birth: pd.Series,
    target_rate: float,
) -> dict[str, float]:
    """Find per-bucket thresholds that equalise selection rates at target_rate.

    For each age bucket, picks the score cut-off that selects the top
    target_rate fraction of applicants in that bucket, giving equal positive-
    prediction rates across all groups (disparate impact -> 1.0).
    """
    age = -days_birth.values / 365.25
    bucket = pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS)
    thresholds: dict[str, float] = {}
    for label in AGE_LABELS:
        mask = np.asarray(bucket == label)
        if mask.sum() == 0:
            continue
        scores = np.sort(y_score[mask])[::-1]
        n_select = max(1, round(target_rate * len(scores)))
        thresholds[label] = float(scores[min(n_select, len(scores)) - 1])
    return thresholds


def train_fair(params: dict[str, object] | None = None) -> str:
    """Fairness-constrained LightGBM training; returns MLflow run_id.

    Logs all hyperparameters, the LightGBM booster, and the per-group thresholds
    to MLflow.  The local .joblib bundle includes 'group_thresholds' so that
    evaluate.py applies them automatically without any extra flags.
    """
    load_dotenv()
    if params is None:
        params = BEST_PARAMS

    # ── Load full training set once — derive all splits from this frame ────────
    feature_cols = get_feature_cols()
    cols_needed = list(dict.fromkeys([ID_COL, TARGET_COL, "DAYS_BIRTH"] + feature_cols))
    full = pl.read_parquet(PROCESSED_DIR / "train_complete").sort(ID_COL).select(cols_needed)
    split_idx = int(len(full) * (1 - EVAL_HOLDOUT_FRAC))

    X_df = full[feature_cols].to_pandas()
    y = full[TARGET_COL].to_numpy()

    # ── Stage 1: AIF360 Reweighing ────────────────────────────────────────────
    days_birth_full = full["DAYS_BIRTH"].to_pandas()
    age_binary = (-days_birth_full.values / 365.25 >= AGE_SPLIT).astype(np.float64)
    del days_birth_full
    sample_weights = _aif360_reweigh(y, age_binary)

    X_np, encoder, cat_cols = encode_categoricals(X_df)
    del X_df  # free before loading holdout

    # ── Holdout for threshold calibration (temporal last 20%) ─────────────────
    val_pl = full[split_idx:]
    del full  # free the full frame now that we have X_np and the holdout slice
    X_val = val_pl[feature_cols].to_pandas()
    y_val = val_pl[TARGET_COL].to_numpy()
    days_birth_val = val_pl["DAYS_BIRTH"].to_pandas()
    del val_pl
    X_val_enc = X_val.copy()
    X_val_enc[cat_cols] = encoder.transform(X_val[cat_cols])
    X_val_np = X_val_enc.to_numpy()
    del X_val, X_val_enc

    MODELS_DIR.mkdir(exist_ok=True)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        mlflow.set_tags({
            "stage": "staging",
            "model_type": "lightgbm",
            "fair": "true",
            "fairness_method": "aif360_reweighing+group_threshold_calibration",
            "protected_attribute": "age_binary",
        })
        mlflow.log_params({k: v for k, v in params.items() if k not in _SKIP_LOG})
        mlflow.log_params({
            "n_features": X_np.shape[1],
            "n_train_rows": int(X_np.shape[0]),
            "age_split_years": AGE_SPLIT,
        })

        # Train with AIF360-computed sample weights
        model = lgb.LGBMClassifier(**params)
        model.fit(X_np, y, sample_weight=sample_weights)

        mlflow.lightgbm.log_model(model.booster_, artifact_path="model")

        # ── Stage 2: Group threshold calibration ──────────────────────────────
        y_val_score = model.predict_proba(X_val_np)[:, 1]
        _, target_rate = _youden_threshold(y_val, y_val_score)
        group_thresholds = _calibrate_group_thresholds(y_val_score, days_birth_val, target_rate)

        mlflow.log_params({f"threshold_{k}": round(v, 6) for k, v in group_thresholds.items()})

        # ── Bundle ─────────────────────────────────────────────────────────────
        bundle_path = MODELS_DIR / f"lgb_fair_{run_id[:8]}.joblib"
        joblib.dump(
            {
                "model": model,
                "encoder": encoder,
                "feature_cols": feature_cols,
                "cat_cols": cat_cols,
                "params": params,
                "run_id": run_id,
                "group_thresholds": group_thresholds,
                "age_bins": AGE_BINS,
                "age_labels": AGE_LABELS,
            },
            bundle_path,
        )
        mlflow.log_artifact(str(bundle_path), artifact_path="bundle")

        print(f"MLflow run_id   : {run_id}")
        print(f"Local bundle    : {bundle_path}")
        print(f"Target rate     : {target_rate:.4f}")
        print(f"Group thresholds: {group_thresholds}")

    return run_id


if __name__ == "__main__":
    train_fair()
