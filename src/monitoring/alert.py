"""Drift threshold checker and webhook alerter.

Usage:
    python -m src.monitoring.alert

Reads the most recent drift report from data/processed/drift/, evaluates
PSI thresholds, and fires a Slack or PagerDuty webhook if any feature
crosses the alert threshold.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRIFT_DIR = PROJECT_ROOT / "data" / "processed" / "drift"

PSI_WARNING: float = 0.1
PSI_ALERT: float = 0.2


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_latest_report() -> dict | None:
    """Return the most recent drift report as a dict, or None if none exist."""
    files = sorted(DRIFT_DIR.glob("*.json"))
    if not files:
        logger.warning("No drift reports found in %s. Run drift_report.py first.", DRIFT_DIR)
        return None
    with files[-1].open() as fh:
        return json.load(fh)


def _build_slack_payload(report: dict, triggered_features: list[str]) -> dict:
    """Build a Slack Block Kit webhook payload for the drift alert."""
    feature_lines = "\n".join(
        f"  • {feat}: {stats['psi']:.4f} {'🚨' if stats['psi'] > PSI_ALERT else '⚠️'}"
        for feat, stats in report["features"].items()
        if stats["psi"] >= PSI_WARNING
    )
    return {
        "text": "🚨 Credit Risk Model — Drift Alert",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🚨 Credit Risk Model Drift Alert"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Report date:* {report['report_date']}\n"
                        f"*Max PSI:* {report['max_psi']:.4f}\n"
                        f"*Triggered features:* {', '.join(triggered_features)}"
                    ),
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Feature PSI scores:*\n" + feature_lines,
                },
            },
        ],
    }


def _build_pagerduty_payload(report: dict, triggered_features: list[str]) -> dict:
    """Build a PagerDuty Events API v2 payload for the drift alert."""
    return {
        "routing_key": "",
        "event_action": "trigger",
        "payload": {
            "summary": f"Credit Risk Drift Alert — Max PSI {report['max_psi']:.4f}",
            "severity": "critical" if report["max_psi"] > PSI_ALERT else "warning",
            "source": "credit-risk-api",
            "timestamp": report["report_date"],
            "custom_details": {
                "max_psi": report["max_psi"],
                "triggered_features": triggered_features,
                "feature_psi": {f: s["psi"] for f, s in report["features"].items()},
            },
        },
    }


def _send_webhook(url: str, report: dict, triggered_features: list[str]) -> bool:
    """POST an alert to the webhook URL. Returns True on success, False on error."""
    is_slack = url.startswith("https://hooks.slack.com")
    is_pagerduty = "pagerduty.com" in url

    if is_slack:
        body = _build_slack_payload(report, triggered_features)
    elif is_pagerduty:
        body = _build_pagerduty_payload(report, triggered_features)
    else:
        body = {"report": report, "triggered_features": triggered_features}

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        return True
    except urllib.error.URLError as exc:
        logger.error("Webhook POST failed: %s", exc)
        return False


# ── Main ───────────────────────────────────────────────────────────────────────

def check_and_alert() -> None:
    """Load the latest drift report, evaluate thresholds, and fire an alert if needed."""
    load_dotenv()

    report = _load_latest_report()
    if report is None:
        logger.info("No drift report found. Run drift_report.py first.")
        return

    alert_features = [
        f for f, s in report["features"].items() if s["psi"] > PSI_ALERT
    ]
    warning_features = [
        f for f, s in report["features"].items()
        if PSI_WARNING <= s["psi"] <= PSI_ALERT
    ]

    if warning_features:
        logger.warning("PSI warning threshold exceeded: %s", warning_features)

    if not alert_features:
        logger.info("No alert threshold exceeded. Max PSI: %.4f", report["max_psi"])
        return

    logger.error("ALERT: PSI > %.2f for features: %s", PSI_ALERT, alert_features)

    webhook_url = os.getenv("ALERT_WEBHOOK_URL", "")
    if not webhook_url:
        logger.error("ALERT_WEBHOOK_URL not set — cannot send alert.")
        return

    success = _send_webhook(webhook_url, report, alert_features)
    if success:
        logger.info("Alert webhook sent successfully.")
    else:
        logger.error("Failed to send alert webhook.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    check_and_alert()
