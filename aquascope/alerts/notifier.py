"""
Simple notification dispatch for water-quality alerts.

Supports logging, file output, webhook POST, and basic e-mail delivery.
"""

from __future__ import annotations

import json
import logging
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path

import httpx

from aquascope.alerts.checker import Alert

logger = logging.getLogger(__name__)


@dataclass
class NotificationConfig:
    """Configuration for notification channels.

    Parameters
    ----------
    webhook_url:
        URL for HTTP POST webhook delivery.
    email_to:
        Recipient address for e-mail notifications.
    email_from:
        Sender address for e-mail notifications.
    smtp_host:
        SMTP server hostname.
    smtp_port:
        SMTP server port (default 587 for STARTTLS).
    log_file:
        Path to append JSON-lines alert records.
    """

    webhook_url: str | None = None
    email_to: str | None = None
    email_from: str | None = None
    smtp_host: str | None = None
    smtp_port: int = 587
    log_file: str | None = None


def _alert_to_dict(alert: Alert) -> dict:
    """Serialise an ``Alert`` to a plain dict for JSON output."""
    return {
        "parameter": alert.parameter,
        "value": alert.value,
        "limit": alert.threshold.limit,
        "unit": alert.threshold.unit,
        "standard": alert.threshold.standard,
        "severity": alert.severity,
        "exceedance_ratio": alert.exceedance_ratio,
        "timestamp": alert.timestamp.isoformat() if alert.timestamp else None,
        "station_id": alert.station_id,
        "message": alert.message,
    }


# ---------------------------------------------------------------------------
# Channel implementations
# ---------------------------------------------------------------------------


def _notify_log(alerts: list[Alert]) -> bool:
    """Write alerts to the module logger."""
    for alert in alerts:
        if alert.severity == "critical":
            logger.critical(alert.message)
        elif alert.severity == "warning":
            logger.warning(alert.message)
        else:
            logger.info(alert.message)
    return True


def _notify_file(alerts: list[Alert], path: str) -> bool:
    """Append alerts as JSON lines to *path*."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            for alert in alerts:
                fh.write(json.dumps(_alert_to_dict(alert), default=str) + "\n")
        return True
    except OSError:
        logger.exception("Failed to write alerts to %s", path)
        return False


def _notify_webhook(alerts: list[Alert], url: str) -> bool:
    """POST alerts as JSON to a webhook URL."""
    payload = [_alert_to_dict(a) for a in alerts]
    try:
        resp = httpx.post(url, json=payload, timeout=15.0)
        resp.raise_for_status()
        logger.info("Webhook delivered %d alerts to %s (HTTP %d)", len(alerts), url, resp.status_code)
        return True
    except (httpx.HTTPError, OSError):
        logger.exception("Webhook delivery failed for %s", url)
        return False


def _notify_email(alerts: list[Alert], config: NotificationConfig) -> bool:
    """Send alerts via SMTP e-mail."""
    if not (config.email_to and config.email_from and config.smtp_host):
        logger.error("Incomplete e-mail configuration — skipping e-mail notification")
        return False

    body_lines = [f"AquaScope Alert Report — {len(alerts)} alert(s)\n"]
    for alert in alerts:
        body_lines.append(f"[{alert.severity.upper()}] {alert.message}")
    body = "\n".join(body_lines)

    msg = EmailMessage()
    msg["Subject"] = f"AquaScope Alerts — {len(alerts)} exceedance(s)"
    msg["From"] = config.email_from
    msg["To"] = config.email_to
    msg.set_content(body)

    try:
        with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=15) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.send_message(msg)
        logger.info("E-mail alert sent to %s", config.email_to)
        return True
    except (smtplib.SMTPException, OSError):
        logger.exception("E-mail delivery failed")
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def notify(alerts: list[Alert], config: NotificationConfig) -> dict[str, bool]:
    """Dispatch alerts through all configured notification channels.

    Parameters
    ----------
    alerts:
        List of ``Alert`` objects to dispatch.
    config:
        ``NotificationConfig`` describing which channels to use.

    Returns
    -------
    dict[str, bool]
        Mapping of channel name to delivery success/failure.
    """
    if not alerts:
        logger.info("No alerts to dispatch")
        return {}

    results: dict[str, bool] = {}

    # Log channel — always active
    results["log"] = _notify_log(alerts)

    # File channel
    if config.log_file:
        results["file"] = _notify_file(alerts, config.log_file)

    # Webhook channel
    if config.webhook_url:
        results["webhook"] = _notify_webhook(alerts, config.webhook_url)

    # Email channel
    if config.email_to:
        results["email"] = _notify_email(alerts, config)

    return results
