"""Evaluate a staged model: AUC-ROC, Gini, KS statistic, and fairness audit.

For fair-model bundles (those containing 'group_thresholds'), per-age-bucket
decision thresholds are applied automatically instead of a single global threshold.
Use --baseline-run-id to print a side-by-side comparison with the baseline model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from dotenv import load_dotenv
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from src.features.feature_registry import ID_COL, TARGET_COL, get_feature_cols

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

EVAL_HOLDOUT_FRAC = 0.20       # temporal last-20% of SK_ID_CURR-sorted rows
FAIRNESS_THRESHOLD = 0.80      # four-fifths rule
PRED_THRESHOLD_DEFAULT = 0.5

AGE_BINS = [0, 35, 55, 150]
AGE_LABELS = ["young_lt35", "middle_35_55", "senior_gt55"]


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_bundle(run_id: str) -> dict:
    """Locate and load the joblib bundle for the given run_id prefix."""
    prefix = run_id[:8]
    matches = list(MODELS_DIR.glob(f"lgb_*{prefix}*.joblib"))
    if not matches:
        raise FileNotFoundError(f"No bundle found for run_id prefix '{prefix}' in {MODELS_DIR}")
    return joblib.load(matches[0])


def _load_holdout(feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Load temporal 20% holdout sorted by SK_ID_CURR; return (X_with_DAYS_BIRTH, y)."""
    train = pl.read_parquet(PROCESSED_DIR / "train_complete").sort(ID_COL)
    split_idx = int(len(train) * (1 - EVAL_HOLDOUT_FRAC))
    val = train[split_idx:]
    cols_needed = list(dict.fromkeys(feature_cols + ["DAYS_BIRTH"]))
    X = val[cols_needed].to_pandas()
    y = val[TARGET_COL].to_numpy()
    return X, y


def _encode(X: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Apply the bundle's fitted OrdinalEncoder; return float numpy array."""
    X_enc = X[bundle["feature_cols"]].copy()
    X_enc[bundle["cat_cols"]] = bundle["encoder"].transform(X[bundle["cat_cols"]])
    return X_enc.to_numpy()


# ── Prediction ────────────────────────────────────────────────────────────────

def _apply_group_thresholds(
    y_score: np.ndarray,
    days_birth: pd.Series,
    group_thresholds: dict[str, float],
) -> np.ndarray:
    """Apply per-age-bucket thresholds; falls back to 0.5 for unmatched groups."""
    age = -days_birth.values / 365.25
    bucket = pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS)
    y_pred = np.zeros(len(y_score), dtype=int)
    for label in AGE_LABELS:
        mask = np.asarray(bucket == label)
        threshold = group_thresholds.get(label, 0.5)
        y_pred[mask] = (y_score[mask] >= threshold).astype(int)
    return y_pred


# ── Metrics ───────────────────────────────────────────────────────────────────

def _gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Normalised Gini = 2 * AUC - 1."""
    return float(2 * roc_auc_score(y_true, y_score) - 1)


def _ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS statistic: max separation between default and non-default score CDFs."""
    stat, _ = ks_2samp(y_score[y_true == 1], y_score[y_true == 0])
    return float(stat)


def _fairness_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    days_birth: pd.Series,
) -> dict[str, float]:
    """Compute equal opportunity difference and disparate impact ratio across age buckets.

    Equal opportunity difference: max(TPR_group) - min(TPR_group).
    Disparate impact ratio: min(selection_rate_group) / max(selection_rate_group).
    """
    age = (-days_birth / 365.25).values
    bucket = pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS)

    tpr: dict[str, float] = {}
    pos_rate: dict[str, float] = {}

    for label in AGE_LABELS:
        mask = np.asarray(bucket == label)
        if mask.sum() == 0:
            continue
        y_t, y_p = y_true[mask], y_pred[mask]
        n_pos = y_t.sum()
        tpr[label] = float((y_p[y_t == 1] == 1).sum() / n_pos) if n_pos > 0 else 0.0
        pos_rate[label] = float(y_p.mean())

    tpr_vals = list(tpr.values())
    pr_vals = list(pos_rate.values())
    eod = float(max(tpr_vals) - min(tpr_vals)) if tpr_vals else float("nan")
    dir_ = float(min(pr_vals) / max(pr_vals)) if pr_vals and max(pr_vals) > 0 else float("nan")

    result: dict[str, float] = {"equal_opportunity_diff": eod, "disparate_impact_ratio": dir_}
    for label in AGE_LABELS:
        if label in tpr:
            result[f"tpr_{label}"] = tpr[label]
            result[f"pos_rate_{label}"] = pos_rate[label]
    return result


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_report(metrics: dict[str, float], label: str = "") -> None:
    """Print a formatted evaluation report."""
    sep = "-" * 52
    header = f"  {label}" if label else ""
    print(f"\n{sep}{header}")
    print("  Performance")
    print(sep)
    print(f"  AUC-ROC  : {metrics['auc_roc']:.4f}")
    print(f"  Gini     : {metrics['gini']:.4f}")
    print(f"  KS stat  : {metrics['ks_stat']:.4f}")
    print(f"\n{sep}")
    print("  Fairness audit (age buckets)")
    print(sep)
    print(f"  Equal opportunity diff  : {metrics['equal_opportunity_diff']:.4f}")
    dir_ = metrics["disparate_impact_ratio"]
    flag = "PASS" if dir_ >= FAIRNESS_THRESHOLD else f"FAIL  (< {FAIRNESS_THRESHOLD}) -- do not promote without justification"
    print(f"  Disparate impact ratio  : {dir_:.4f}  {flag}")
    print(f"{sep}\n")


def _print_comparison(
    baseline: dict[str, float],
    fair: dict[str, float],
    baseline_label: str = "baseline",
    fair_label: str = "fair",
) -> None:
    """Print a side-by-side comparison of baseline vs fair model metrics."""
    keys = [
        "auc_roc", "gini", "ks_stat",
        "equal_opportunity_diff", "disparate_impact_ratio",
    ]
    sep = "-" * 63
    b_lbl = baseline_label[:9]
    f_lbl = fair_label[:9]
    print(f"\n{sep}")
    print(f"  {'Metric':<34}  {b_lbl:>9}  {f_lbl:>9}  {'Delta':>8}")
    print(sep)
    for k in keys:
        b = baseline.get(k, float("nan"))
        f = fair.get(k, float("nan"))
        delta = f - b
        sign = "+" if delta > 0 else ""
        print(f"  {k:<34}  {b:>9.4f}  {f:>9.4f}  {sign}{delta:>7.4f}")
    print(sep)
    dir_fair = fair.get("disparate_impact_ratio", 0.0)
    verdict = "PROMOTABLE" if dir_fair >= FAIRNESS_THRESHOLD else "NOT PROMOTABLE"
    print(f"  Fair model disparate impact: {dir_fair:.4f}  -> {verdict}")
    print(f"{sep}\n")


# ── Core evaluate function ────────────────────────────────────────────────────

def evaluate(run_id: str, pred_threshold: float = PRED_THRESHOLD_DEFAULT) -> dict[str, float]:
    """Load bundle, evaluate on temporal holdout, log metrics to MLflow run.

    For fair-model bundles (containing 'group_thresholds'), per-group thresholds
    are applied for y_pred; pred_threshold is ignored for those models.

    Args:
        run_id: MLflow run_id from train.py or train_fair.py.
        pred_threshold: Global probability cutoff (ignored when group_thresholds present).

    Returns:
        Dict of all computed metrics.
    """
    load_dotenv()
    bundle = _load_bundle(run_id)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    group_thresholds: dict[str, float] | None = bundle.get("group_thresholds")

    X, y = _load_holdout(feature_cols)
    days_birth = X["DAYS_BIRTH"]

    X_np = _encode(X, bundle)
    y_score = model.predict_proba(X_np)[:, 1]

    if group_thresholds:
        y_pred = _apply_group_thresholds(y_score, days_birth, group_thresholds)
    else:
        y_pred = (y_score >= pred_threshold).astype(int)

    metrics: dict[str, float] = {
        "auc_roc": float(roc_auc_score(y, y_score)),
        "gini": _gini(y, y_score),
        "ks_stat": _ks(y, y_score),
        **_fairness_audit(y, y_pred, days_birth),
    }

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a staged LightGBM model.")
    parser.add_argument("--run-id", required=True, help="MLflow run_id from train.py / train_fair.py")
    parser.add_argument(
        "--threshold", type=float, default=PRED_THRESHOLD_DEFAULT,
        help="Global probability threshold (ignored for fair-model bundles, default: 0.5)",
    )
    parser.add_argument(
        "--baseline-run-id",
        help="Optional baseline run_id to print a side-by-side comparison",
    )
    args = parser.parse_args()

    result = evaluate(run_id=args.run_id, pred_threshold=args.threshold)

    if args.baseline_run_id:
        baseline_result = evaluate(run_id=args.baseline_run_id, pred_threshold=args.threshold)
        _print_report(baseline_result, label="[baseline]")
        _print_report(result, label="[fair]")
        _print_comparison(baseline_result, result)
    else:
        _print_report(result)

    if result.get("disparate_impact_ratio", 1.0) < FAIRNESS_THRESHOLD:
        sys.exit(1)
