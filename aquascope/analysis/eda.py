"""
Exploratory Data Analysis (EDA) module.

Auto-profiles water quality datasets and generates summary reports
with statistics, distributions, correlations, and coverage maps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from aquascope.ai_engine.recommender import DatasetProfile

logger = logging.getLogger(__name__)


@dataclass
class ParameterStats:
    """Summary statistics for a single water-quality parameter."""

    name: str
    count: int
    missing: int
    mean: float
    std: float
    min: float
    q25: float
    median: float
    q75: float
    max: float
    outlier_count: int = 0  # IQR-based


@dataclass
class EDAReport:
    """Full EDA report for a dataset."""

    n_records: int
    n_stations: int
    n_parameters: int
    date_range: tuple[str, str] | None = None
    time_span_years: float = 0.0
    parameters: list[ParameterStats] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    completeness_pct: float = 0.0
    correlation_matrix: pd.DataFrame | None = None


def _count_outliers_iqr(series: pd.Series) -> int:
    """Count outliers using the IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """
    Auto-detect dataset characteristics and return a DatasetProfile
    suitable for the AI recommender.

    Expects columns: parameter, value, station_id, sample_datetime (or reading_datetime), source.
    """
    dt_col = "sample_datetime" if "sample_datetime" in df.columns else "reading_datetime"

    parameters = sorted(df["parameter"].dropna().unique().tolist()) if "parameter" in df.columns else []
    n_stations = df["station_id"].nunique() if "station_id" in df.columns else 0
    sources = sorted(df["source"].dropna().unique().tolist()) if "source" in df.columns else []

    time_span = 0.0
    scope = ""
    if dt_col in df.columns:
        dates = pd.to_datetime(df[dt_col], errors="coerce").dropna()
        if len(dates) > 1:
            time_span = (dates.max() - dates.min()).days / 365.25

    if "county" in df.columns:
        counties = df["county"].dropna().unique()
        scope = f"Taiwan — {', '.join(counties[:3])}" if len(counties) <= 3 else "Taiwan — multiple counties"
    elif "basin" in df.columns:
        basins = df["basin"].dropna().unique()
        scope = f"Taiwan — {basins[0]} basin" if len(basins) == 1 else "Taiwan"

    return DatasetProfile(
        parameters=parameters,
        n_records=len(df),
        n_stations=n_stations,
        time_span_years=round(time_span, 1),
        geographic_scope=scope,
        data_sources=sources,
    )


def generate_eda_report(df: pd.DataFrame) -> EDAReport:
    """
    Generate a comprehensive EDA report from a water data DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have at least: ``parameter``, ``value``.
        Optional: ``station_id``, ``sample_datetime``, ``source``.
    """
    dt_col = "sample_datetime" if "sample_datetime" in df.columns else "reading_datetime"

    n_records = len(df)
    n_stations = df["station_id"].nunique() if "station_id" in df.columns else 0
    sources = sorted(df["source"].dropna().unique().tolist()) if "source" in df.columns else []

    # Date range
    date_range = None
    time_span = 0.0
    if dt_col in df.columns:
        dates = pd.to_datetime(df[dt_col], errors="coerce").dropna()
        if len(dates) > 1:
            date_range = (str(dates.min().date()), str(dates.max().date()))
            time_span = (dates.max() - dates.min()).days / 365.25

    # Per-parameter stats
    param_stats: list[ParameterStats] = []
    if "parameter" in df.columns and "value" in df.columns:
        for param, group in df.groupby("parameter"):
            vals = pd.to_numeric(group["value"], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            desc = vals.describe()
            param_stats.append(
                ParameterStats(
                    name=str(param),
                    count=int(desc["count"]),
                    missing=int(len(group) - desc["count"]),
                    mean=round(float(desc["mean"]), 4),
                    std=round(float(desc["std"]), 4) if desc["std"] == desc["std"] else 0.0,
                    min=round(float(desc["min"]), 4),
                    q25=round(float(desc["25%"]), 4),
                    median=round(float(desc["50%"]), 4),
                    q75=round(float(desc["75%"]), 4),
                    max=round(float(desc["max"]), 4),
                    outlier_count=_count_outliers_iqr(vals),
                )
            )

    # Completeness
    total_cells = df.shape[0] * df.shape[1]
    non_null_cells = df.notna().sum().sum()
    completeness = round(non_null_cells / total_cells * 100, 1) if total_cells > 0 else 0.0

    # Correlation matrix (pivot parameters as columns)
    corr_matrix = None
    if "parameter" in df.columns and "value" in df.columns and dt_col in df.columns:
        try:
            pivot = df.pivot_table(index=[dt_col, "station_id"], columns="parameter", values="value", aggfunc="mean")
            if pivot.shape[1] >= 2:
                corr_matrix = pivot.corr().round(3)
        except Exception:
            pass

    return EDAReport(
        n_records=n_records,
        n_stations=n_stations,
        n_parameters=len(param_stats),
        date_range=date_range,
        time_span_years=round(time_span, 1),
        parameters=param_stats,
        sources=sources,
        completeness_pct=completeness,
        correlation_matrix=corr_matrix,
    )


def print_eda_report(report: EDAReport) -> str:
    """Format an EDA report as a readable text summary."""
    lines = [
        "=" * 70,
        "  AquaScope — Exploratory Data Analysis Report",
        "=" * 70,
        "",
        f"  Records       : {report.n_records:,}",
        f"  Stations      : {report.n_stations}",
        f"  Parameters    : {report.n_parameters}",
        f"  Date range    : {report.date_range[0]} → {report.date_range[1]}" if report.date_range else "",
        f"  Time span     : {report.time_span_years:.1f} years",
        f"  Completeness  : {report.completeness_pct:.1f}%",
        f"  Data sources  : {', '.join(report.sources)}",
        "",
        "  Parameter Statistics:",
        "  " + "-" * 66,
        f"  {'Parameter':<18} {'Count':>7} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Outliers':>8}",
        "  " + "-" * 66,
    ]
    for p in report.parameters:
        line = f"  {p.name:<18} {p.count:>7} {p.mean:>10.2f} {p.std:>10.2f}"
        line += f" {p.min:>10.2f} {p.max:>10.2f} {p.outlier_count:>8}"
        lines.append(line)
    lines.append("  " + "-" * 66)

    if report.correlation_matrix is not None:
        lines.extend(["", "  Top Correlations (|r| > 0.5):"])
        corr = report.correlation_matrix
        seen = set()
        for i in corr.columns:
            for j in corr.columns:
                if i != j and (j, i) not in seen:
                    r = corr.loc[i, j]
                    if abs(r) > 0.5:
                        lines.append(f"    {i} ↔ {j} : r = {r:.3f}")
                    seen.add((i, j))

    return "\n".join(lines)
