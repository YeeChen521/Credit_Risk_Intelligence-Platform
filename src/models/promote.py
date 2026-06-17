"""Promote a staged MLflow run to production, archiving the previous production run.

Typical workflow:
  1. python -m src.models.train                               # logs run, tags stage=staging
  2. python -m src.models.evaluate --run-id <run_id>         # logs metrics, checks fairness
  3. python -m src.models.promote --run-id <run_id>          # promotes to production
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

FAIRNESS_THRESHOLD = 0.80  # four-fifths rule — must match evaluate.py


# ── MLflow helpers ────────────────────────────────────────────────────────────

def _get_run(client: mlflow.MlflowClient, run_id: str) -> mlflow.entities.Run:
    """Retrieve and return an MLflow run, exiting with an error message if not found."""
    try:
        return client.get_run(run_id)
    except mlflow.exceptions.MlflowException as exc:
        print(f"ERROR: Could not retrieve run '{run_id}': {exc}")
        sys.exit(1)


def _archive_production_runs(client: mlflow.MlflowClient, experiment_id: str) -> list[str]:
    """Set stage=archived on all currently production-tagged runs; return their run_ids."""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.stage = 'production'",
    )
    archived = [r.info.run_id for r in runs]
    for run_id in archived:
        client.set_tag(run_id, "stage", "archived")
    return archived


# ── Bundle rename ─────────────────────────────────────────────────────────────

def _rename_bundle(run_id: str) -> Path | None:
    """Rename lgb_staging_<prefix>*.joblib → lgb_production_<prefix>*.joblib if present."""
    prefix = run_id[:8]
    matches = list(MODELS_DIR.glob(f"lgb_staging_{prefix}*.joblib"))
    if not matches:
        return None
    src = matches[0]
    dst = src.parent / src.name.replace("lgb_staging_", "lgb_production_", 1)
    src.rename(dst)
    return dst


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_metrics(metrics: dict[str, float]) -> None:
    """Print a concise metrics table for the run being promoted."""
    sep = "-" * 52
    print(f"\n{sep}")
    print("  Metrics on file for this run")
    print(sep)
    for key in ("auc_roc", "gini", "ks_stat", "disparate_impact_ratio", "equal_opportunity_diff"):
        if key in metrics:
            print(f"  {key:<34}: {metrics[key]:.4f}")
    print(f"{sep}\n")


# ── Core promote function ─────────────────────────────────────────────────────

def promote(run_id: str, *, force: bool = False) -> None:
    """Promote a staged MLflow run to production.

    Args:
        run_id: MLflow run_id produced by train.py or train_fair.py.
        force:  Skip evaluation and fairness gates. Requires documented justification.
    """
    load_dotenv()
    client = mlflow.MlflowClient()
    run = _get_run(client, run_id)

    tags = run.data.tags
    metrics = run.data.metrics
    current_stage = tags.get("stage", "<unset>")

    if current_stage == "production":
        print(f"Run {run_id} is already tagged 'production'. Nothing to do.")
        return

    if current_stage != "staging":
        print(
            f"WARNING: Run {run_id} has stage='{current_stage}' (expected 'staging'). "
            "Continuing anyway."
        )

    # Gate 1 — require evaluate.py to have been run
    if "auc_roc" not in metrics and not force:
        print(
            "BLOCKED: No evaluation metrics found on this run.\n"
            "  Run:  python -m src.models.evaluate --run-id " + run_id + "\n"
            "  Then retry promote, or pass --force to skip this check."
        )
        sys.exit(1)

    if "auc_roc" not in metrics and force:
        print("WARNING: Promoting without evaluation metrics. Document your justification.")

    # Gate 2 — fairness check (four-fifths rule)
    dir_ratio = metrics.get("disparate_impact_ratio")
    if dir_ratio is not None:
        if dir_ratio < FAIRNESS_THRESHOLD and not force:
            print(
                f"BLOCKED: disparate_impact_ratio={dir_ratio:.4f} is below the four-fifths "
                f"threshold ({FAIRNESS_THRESHOLD}).\n"
                "  Review the fairness report and obtain a documented justification before\n"
                "  re-running with --force."
            )
            sys.exit(1)
        if dir_ratio < FAIRNESS_THRESHOLD and force:
            print(
                f"WARNING: Promoting despite disparate_impact_ratio={dir_ratio:.4f} "
                f"< {FAIRNESS_THRESHOLD}. This requires external documented justification."
            )

    # Print metrics for audit trail
    if metrics:
        _print_metrics(metrics)

    # Archive the current production run(s)
    experiment_id = run.info.experiment_id
    archived_ids = _archive_production_runs(client, experiment_id)
    for aid in archived_ids:
        print(f"Archived previous production run : {aid}")

    # Promote the candidate
    client.set_tag(run_id, "stage", "production")
    print(f"Promoted run                    : {run_id}  →  stage=production")

    # Rename the local bundle file
    bundle_path = _rename_bundle(run_id)
    if bundle_path:
        print(f"Renamed local bundle            : {bundle_path.name}")
    else:
        print(
            f"Note: No local bundle found for prefix '{run_id[:8]}' in {MODELS_DIR}. "
            "The API uses the MLflow artifact directly."
        )

    print("\nDone. Redeploy the API container to load the promoted model.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Promote a staged MLflow run to production.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Promotion gates (both are bypassed by --force):\n"
            "  1. evaluate.py must have been run (auc_roc metric must exist)\n"
            "  2. disparate_impact_ratio must be >= 0.80 (four-fifths rule)\n"
        ),
    )
    parser.add_argument("--run-id", required=True, help="MLflow run_id to promote.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass evaluation and fairness gates. Requires documented justification.",
    )
    args = parser.parse_args()
    promote(run_id=args.run_id, force=args.force)
