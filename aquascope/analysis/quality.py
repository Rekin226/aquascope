"""
Data quality assessment and preprocessing pipeline.

Evaluates completeness, consistency, outliers, and duplicates in
collected water data, then applies configurable preprocessing steps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Results of a data quality assessment."""

    n_records: int
    n_duplicates: int
    completeness_pct: float
    null_counts: dict[str, int] = field(default_factory=dict)
    outlier_counts: dict[str, int] = field(default_factory=dict)
    temporal_gaps: list[dict] = field(default_factory=list)
    unit_issues: list[str] = field(default_factory=list)
    recommended_steps: list[str] = field(default_factory=list)


def assess_quality(df: pd.DataFrame) -> QualityReport:
    """
    Run a full quality assessment on a water data DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Expected columns: parameter, value, station_id, sample_datetime / reading_datetime.

    Returns
    -------
    QualityReport with completeness, outliers, gaps, and recommendations.
    """
    dt_col = "sample_datetime" if "sample_datetime" in df.columns else "reading_datetime"
    n_records = len(df)

    # Duplicates
    dup_cols = [c for c in ["station_id", dt_col, "parameter"] if c in df.columns]
    n_duplicates = int(df.duplicated(subset=dup_cols).sum()) if dup_cols else 0

    # Null counts per column
    null_counts = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0}

    # Completeness
    total_cells = df.shape[0] * df.shape[1]
    completeness = round((1 - df.isna().sum().sum() / total_cells) * 100, 1) if total_cells > 0 else 0.0

    # Outlier detection (IQR) per parameter
    outlier_counts: dict[str, int] = {}
    if "parameter" in df.columns and "value" in df.columns:
        for param, group in df.groupby("parameter"):
            vals = pd.to_numeric(group["value"], errors="coerce").dropna()
            if len(vals) < 10:
                continue
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            n_out = int(((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).sum())
            if n_out > 0:
                outlier_counts[str(param)] = n_out

    # Temporal gap detection
    temporal_gaps: list[dict] = []
    if dt_col in df.columns and "station_id" in df.columns:
        df_sorted = df.copy()
        df_sorted[dt_col] = pd.to_datetime(df_sorted[dt_col], errors="coerce")
        for station, grp in df_sorted.groupby("station_id"):
            dates = grp[dt_col].dropna().sort_values()
            if len(dates) < 3:
                continue
            diffs = dates.diff().dropna()
            median_interval = diffs.median()
            if median_interval.days == 0:
                continue
            big_gaps = diffs[diffs > median_interval * 3]
            for gap in big_gaps.head(3):  # report top 3 gaps per station
                idx = diffs[diffs == gap].index[0]
                temporal_gaps.append({
                    "station": str(station),
                    "gap_start": str(dates.loc[dates.index[dates.index.get_loc(idx) - 1]].date()),
                    "gap_days": int(gap.days),
                })
        temporal_gaps = sorted(temporal_gaps, key=lambda x: x["gap_days"], reverse=True)[:10]

    # Unit consistency check
    unit_issues: list[str] = []
    if "parameter" in df.columns and "unit" in df.columns:
        for param, grp in df.groupby("parameter"):
            units = grp["unit"].dropna().unique()
            if len(units) > 1:
                unit_issues.append(f"{param}: mixed units ({', '.join(str(u) for u in units)})")

    # Recommendations
    recommended_steps: list[str] = []
    if n_duplicates > 0:
        recommended_steps.append("remove_duplicates")
    if any(v > n_records * 0.05 for v in null_counts.values()):
        recommended_steps.append("fill_missing")
    if sum(outlier_counts.values()) > n_records * 0.01:
        recommended_steps.append("remove_outliers")
    if unit_issues:
        recommended_steps.append("standardize_units")
    if dt_col in df.columns:
        recommended_steps.append("resample_daily")

    return QualityReport(
        n_records=n_records,
        n_duplicates=n_duplicates,
        completeness_pct=completeness,
        null_counts=null_counts,
        outlier_counts=outlier_counts,
        temporal_gaps=temporal_gaps,
        unit_issues=unit_issues,
        recommended_steps=recommended_steps,
    )


def preprocess(df: pd.DataFrame, steps: list[str] | None = None) -> pd.DataFrame:
    """
    Apply preprocessing steps to a water data DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    steps : list[str] | None
        Ordered list of steps to apply. Supported:
        ``remove_duplicates``, ``fill_missing``, ``remove_outliers``,
        ``normalize``, ``resample_daily``.
        If None, applies all except normalize and resample.

    Returns
    -------
    pd.DataFrame — preprocessed copy.
    """
    result = df.copy()
    dt_col = "sample_datetime" if "sample_datetime" in result.columns else "reading_datetime"

    if steps is None:
        steps = ["remove_duplicates", "fill_missing", "remove_outliers"]

    for step in steps:
        n_before = len(result)

        if step == "remove_duplicates":
            dup_cols = [c for c in ["station_id", dt_col, "parameter"] if c in result.columns]
            if dup_cols:
                result = result.drop_duplicates(subset=dup_cols, keep="first")
                logger.info("remove_duplicates: %d → %d rows", n_before, len(result))

        elif step == "fill_missing":
            if "value" in result.columns:
                result["value"] = pd.to_numeric(result["value"], errors="coerce")
                if "parameter" in result.columns and "station_id" in result.columns:
                    result["value"] = result.groupby(["station_id", "parameter"])["value"].transform(
                        lambda s: s.fillna(s.median())
                    )
                else:
                    result["value"] = result["value"].fillna(result["value"].median())
                logger.info("fill_missing: filled NaN values with group median")

        elif step == "remove_outliers":
            if "parameter" in result.columns and "value" in result.columns:
                result["value"] = pd.to_numeric(result["value"], errors="coerce")
                mask = pd.Series(True, index=result.index)
                for _, grp in result.groupby("parameter"):
                    vals = grp["value"]
                    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                    iqr = q3 - q1
                    outlier_mask = (vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)
                    mask.loc[outlier_mask[outlier_mask].index] = False
                result = result[mask]
                logger.info("remove_outliers: %d → %d rows", n_before, len(result))

        elif step == "normalize":
            if "parameter" in result.columns and "value" in result.columns:
                result["value"] = pd.to_numeric(result["value"], errors="coerce")
                result["value_normalized"] = result.groupby("parameter")["value"].transform(
                    lambda s: (s - s.mean()) / s.std() if s.std() > 0 else 0
                )
                logger.info("normalize: added value_normalized column (z-score per parameter)")

        elif step == "resample_daily":
            if dt_col in result.columns and "value" in result.columns:
                result[dt_col] = pd.to_datetime(result[dt_col], errors="coerce")
                group_cols = [c for c in ["station_id", "parameter"] if c in result.columns]
                if group_cols:
                    result = (
                        result.set_index(dt_col)
                        .groupby(group_cols)
                        .resample("D")["value"]
                        .mean()
                        .reset_index()
                    )
                logger.info("resample_daily: resampled to daily mean")

        else:
            logger.warning("Unknown preprocessing step: %s", step)

    return result


def print_quality_report(report: QualityReport) -> str:
    """Format a QualityReport as readable text."""
    lines = [
        "=" * 70,
        "  AquaScope — Data Quality Report",
        "=" * 70,
        "",
        f"  Total records    : {report.n_records:,}",
        f"  Duplicates       : {report.n_duplicates:,}",
        f"  Completeness     : {report.completeness_pct:.1f}%",
    ]

    if report.null_counts:
        lines.append("\n  Missing Values:")
        for col, cnt in sorted(report.null_counts.items(), key=lambda x: -x[1]):
            pct = cnt / report.n_records * 100
            lines.append(f"    {col:<25} {cnt:>7} ({pct:.1f}%)")

    if report.outlier_counts:
        lines.append("\n  Outliers (IQR method):")
        for param, cnt in sorted(report.outlier_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {param:<25} {cnt:>7}")

    if report.temporal_gaps:
        lines.append("\n  Largest Temporal Gaps:")
        for gap in report.temporal_gaps[:5]:
            lines.append(f"    Station {gap['station']}: {gap['gap_days']} days (from {gap['gap_start']})")

    if report.unit_issues:
        lines.append("\n  Unit Consistency Issues:")
        for issue in report.unit_issues:
            lines.append(f"    ⚠ {issue}")

    if report.recommended_steps:
        lines.append("\n  Recommended Preprocessing:")
        for i, step in enumerate(report.recommended_steps, 1):
            lines.append(f"    {i}. {step}")

    return "\n".join(lines)
