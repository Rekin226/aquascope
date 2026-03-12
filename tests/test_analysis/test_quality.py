"""Tests for the data quality assessment module."""

import numpy as np
import pandas as pd

from aquascope.analysis.quality import (
    QualityReport,
    assess_quality,
    preprocess,
    print_quality_report,
)


def _make_df(n_rows: int = 100, add_dupes: bool = False, add_nulls: bool = False) -> pd.DataFrame:
    """Create a sample DataFrame for quality testing."""
    rng = np.random.default_rng(42)
    stations = [f"ST{i:03d}" for i in range(5)]
    params = ["DO", "BOD5", "COD", "NH3-N", "SS"]
    rows = []
    for i in range(n_rows):
        val = round(rng.normal(5.0, 2.0), 2)
        rows.append({
            "source": "taiwan_moenv",
            "station_id": stations[i % 5],
            "parameter": params[i % 5],
            "value": val if not add_nulls or i % 10 != 0 else None,
            "unit": "mg/L",
            "sample_datetime": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}T10:00:00",
        })
    df = pd.DataFrame(rows)
    if add_dupes:
        df = pd.concat([df, df.head(5)], ignore_index=True)
    return df


class TestQualityAssessment:
    def test_assess_returns_report(self):
        df = _make_df()
        report = assess_quality(df)
        assert isinstance(report, QualityReport)
        assert report.n_records == 100

    def test_detects_duplicates(self):
        df = _make_df(add_dupes=True)
        report = assess_quality(df)
        assert report.n_duplicates > 0

    def test_no_duplicates_when_clean(self):
        df = _make_df()
        report = assess_quality(df)
        assert report.n_duplicates == 0

    def test_detects_nulls(self):
        df = _make_df(add_nulls=True)
        report = assess_quality(df)
        assert "value" in report.null_counts

    def test_completeness_below_100_with_nulls(self):
        df = _make_df(add_nulls=True)
        report = assess_quality(df)
        assert report.completeness_pct < 100.0

    def test_recommended_steps(self):
        df = _make_df(add_dupes=True, add_nulls=True)
        report = assess_quality(df)
        assert "remove_duplicates" in report.recommended_steps

    def test_print_quality_report(self):
        df = _make_df()
        report = assess_quality(df)
        text = print_quality_report(report)
        assert "AquaScope" in text
        assert "Total records" in text


class TestPreprocessing:
    def test_remove_duplicates(self):
        df = _make_df(add_dupes=True)
        cleaned = preprocess(df, steps=["remove_duplicates"])
        assert len(cleaned) < len(df)

    def test_fill_missing(self):
        df = _make_df(add_nulls=True)
        assert df["value"].isna().sum() > 0
        cleaned = preprocess(df, steps=["fill_missing"])
        assert cleaned["value"].isna().sum() < df["value"].isna().sum()

    def test_default_steps(self):
        df = _make_df(add_dupes=True, add_nulls=True)
        cleaned = preprocess(df)
        assert len(cleaned) <= len(df)

    def test_normalize_adds_column(self):
        df = _make_df()
        result = preprocess(df, steps=["normalize"])
        assert "value_normalized" in result.columns
