"""
Alert checking engine for water-quality threshold exceedances.

Compares water-quality measurements against regulatory thresholds and
produces structured alert reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from aquascope.alerts.thresholds import Threshold, get_thresholds, list_standards
from aquascope.schemas.water_data import WaterQualitySample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Alert:
    """A single threshold exceedance alert.

    Parameters
    ----------
    parameter:
        The water-quality parameter that exceeded a threshold.
    value:
        The measured value.
    threshold:
        The ``Threshold`` that was exceeded.
    severity:
        ``"critical"``, ``"warning"``, or ``"info"``.
    exceedance_ratio:
        ``value / limit`` (or ``limit / value`` for lower-bound params).
    timestamp:
        When the measurement was taken, if known.
    station_id:
        Monitoring station identifier, if known.
    message:
        Human-readable alert description.
    """

    parameter: str
    value: float
    threshold: Threshold
    severity: str
    exceedance_ratio: float
    timestamp: datetime | None
    station_id: str | None
    message: str


@dataclass
class AlertReport:
    """Aggregated alert report for a batch of samples.

    Parameters
    ----------
    alerts:
        All generated alerts.
    total_samples:
        Number of samples checked.
    samples_with_alerts:
        Number of samples that triggered at least one alert.
    parameters_checked:
        List of parameter names that were evaluated.
    standards_used:
        List of standard names used for checking.
    summary:
        Count of alerts grouped by severity.
    """

    alerts: list[Alert]
    total_samples: int
    samples_with_alerts: int
    parameters_checked: list[str]
    standards_used: list[str]
    summary: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Severity helper
# ---------------------------------------------------------------------------

# Parameters where the limit is a *minimum* (value must be ≥ limit).
_LOWER_BOUND_PARAMS = frozenset({"dissolved_oxygen"})


def severity_from_exceedance(ratio: float) -> str:
    """Determine alert severity from the exceedance ratio.

    Parameters
    ----------
    ratio:
        ``value / limit`` for upper-bound parameters, or
        ``limit / value`` for lower-bound parameters.
        Values < 1.0 are compliant.

    Returns
    -------
    str
        ``"info"`` if compliant (ratio < 1.0),
        ``"warning"`` if 1.0 ≤ ratio < 1.5,
        ``"critical"`` if ratio ≥ 1.5.
    """
    if ratio < 1.0:
        return "info"
    if ratio < 1.5:
        return "warning"
    return "critical"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_value(
    parameter: str,
    value: float,
    standards: list[str] | None,
    timestamp: datetime | None = None,
    station_id: str | None = None,
) -> list[Alert]:
    """Check a single measured value against relevant thresholds."""
    alerts: list[Alert] = []
    std_list = standards if standards else list_standards()

    for std in std_list:
        thresholds = get_thresholds(parameter, standard=std)
        for thresh in thresholds:
            is_lower = parameter in _LOWER_BOUND_PARAMS

            if is_lower:
                if value < thresh.limit:
                    ratio = thresh.limit / value if value != 0 else float("inf")
                    sev = severity_from_exceedance(ratio)
                    msg = (
                        f"{parameter} = {value} {thresh.unit} is below "
                        f"{thresh.standard} minimum of {thresh.limit} {thresh.unit} "
                        f"(ratio {ratio:.2f})"
                    )
                    alerts.append(Alert(
                        parameter=parameter,
                        value=value,
                        threshold=thresh,
                        severity=sev,
                        exceedance_ratio=ratio,
                        timestamp=timestamp,
                        station_id=station_id,
                        message=msg,
                    ))
            else:
                if value > thresh.limit:
                    ratio = value / thresh.limit if thresh.limit != 0 else float("inf")
                    sev = severity_from_exceedance(ratio)
                    msg = (
                        f"{parameter} = {value} {thresh.unit} exceeds "
                        f"{thresh.standard} limit of {thresh.limit} {thresh.unit} "
                        f"(ratio {ratio:.2f})"
                    )
                    alerts.append(Alert(
                        parameter=parameter,
                        value=value,
                        threshold=thresh,
                        severity=sev,
                        exceedance_ratio=ratio,
                        timestamp=timestamp,
                        station_id=station_id,
                        message=msg,
                    ))
    return alerts


def _build_summary(alerts: list[Alert]) -> dict[str, int]:
    """Count alerts by severity."""
    summary: dict[str, int] = {"critical": 0, "warning": 0, "info": 0}
    for alert in alerts:
        summary[alert.severity] = summary.get(alert.severity, 0) + 1
    return summary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_sample(
    sample: WaterQualitySample,
    standards: list[str] | None = None,
) -> list[Alert]:
    """Check a single water-quality sample against thresholds.

    Parameters
    ----------
    sample:
        A ``WaterQualitySample`` Pydantic model instance.
    standards:
        Standards to check against (e.g. ``["WHO", "EPA"]``).
        If ``None``, all standards are used.

    Returns
    -------
    list[Alert]
        Any alerts triggered.  Empty list if compliant.
    """
    return _check_value(
        parameter=sample.parameter,
        value=sample.value,
        standards=standards,
        timestamp=sample.sample_datetime,
        station_id=sample.station_id,
    )


def check_dataframe(
    df: pd.DataFrame,
    value_col: str = "value",
    param_col: str = "parameter",
    standards: list[str] | None = None,
) -> AlertReport:
    """Check every row of a DataFrame for threshold exceedances.

    Parameters
    ----------
    df:
        DataFrame with at least *value_col* and *param_col* columns.
    value_col:
        Column containing measured values.
    param_col:
        Column containing parameter names.
    standards:
        Standards to check against.  Defaults to all.

    Returns
    -------
    AlertReport
        Full report with alerts and summary statistics.
    """
    all_alerts: list[Alert] = []
    rows_with_alerts: set[int] = set()
    checked_params: set[str] = set()

    ts_col = None
    for candidate in ("sample_datetime", "timestamp", "datetime", "date"):
        if candidate in df.columns:
            ts_col = candidate
            break

    station_col = "station_id" if "station_id" in df.columns else None

    for idx, row in df.iterrows():
        param = row.get(param_col)
        value = row.get(value_col)
        if param is None or value is None or pd.isna(value):
            continue

        param = str(param)
        checked_params.add(param)
        timestamp = pd.to_datetime(row[ts_col]) if ts_col and pd.notna(row.get(ts_col)) else None
        station = str(row[station_col]) if station_col and pd.notna(row.get(station_col)) else None

        row_alerts = _check_value(param, float(value), standards, timestamp=timestamp, station_id=station)
        if row_alerts:
            rows_with_alerts.add(int(idx) if not isinstance(idx, int) else idx)
            all_alerts.extend(row_alerts)

    used = standards if standards else list_standards()
    return AlertReport(
        alerts=all_alerts,
        total_samples=len(df),
        samples_with_alerts=len(rows_with_alerts),
        parameters_checked=sorted(checked_params),
        standards_used=used,
        summary=_build_summary(all_alerts),
    )


def check_timeseries(
    df: pd.DataFrame,
    parameter: str,
    value_col: str = "value",
    standards: list[str] | None = None,
) -> AlertReport:
    """Check a time-series DataFrame for a single parameter.

    Parameters
    ----------
    df:
        DataFrame with at least *value_col*.  The index or a
        ``datetime``/``timestamp`` column is used for timestamps.
    parameter:
        Parameter name (e.g. ``"nitrate"``).
    value_col:
        Column containing measured values.
    standards:
        Standards to check against.

    Returns
    -------
    AlertReport
        Report with alerts and summary.
    """
    all_alerts: list[Alert] = []
    rows_with_alerts: set[int] = set()

    for i, (idx, row) in enumerate(df.iterrows()):
        value = row.get(value_col)
        if value is None or pd.isna(value):
            continue

        timestamp = idx if isinstance(idx, (datetime, pd.Timestamp)) else None
        row_alerts = _check_value(parameter, float(value), standards, timestamp=timestamp)
        if row_alerts:
            rows_with_alerts.add(i)
            all_alerts.extend(row_alerts)

    used = standards if standards else list_standards()
    return AlertReport(
        alerts=all_alerts,
        total_samples=len(df),
        samples_with_alerts=len(rows_with_alerts),
        parameters_checked=[parameter],
        standards_used=used,
        summary=_build_summary(all_alerts),
    )
