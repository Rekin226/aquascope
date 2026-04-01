"""
Threshold alerting system for water-quality data.

Compares measurements against WHO, US EPA, and EU Water Framework
Directive standards and dispatches notifications when exceedances are
detected.
"""

from __future__ import annotations

from aquascope.alerts.checker import (
    Alert,
    AlertReport,
    check_dataframe,
    check_sample,
    check_timeseries,
    severity_from_exceedance,
)
from aquascope.alerts.notifier import NotificationConfig, notify
from aquascope.alerts.thresholds import (
    ALL_THRESHOLDS,
    EPA_THRESHOLDS,
    EU_WFD_THRESHOLDS,
    WHO_THRESHOLDS,
    Threshold,
    get_thresholds,
    list_parameters,
    list_standards,
)

__all__ = [
    "ALL_THRESHOLDS",
    "Alert",
    "AlertReport",
    "EPA_THRESHOLDS",
    "EU_WFD_THRESHOLDS",
    "NotificationConfig",
    "Threshold",
    "WHO_THRESHOLDS",
    "check_dataframe",
    "check_sample",
    "check_timeseries",
    "get_thresholds",
    "list_parameters",
    "list_standards",
    "notify",
    "severity_from_exceedance",
]
