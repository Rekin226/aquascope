"""Tests for the data quality assessment module."""

import numpy as np
import pandas as pd

from aquascope.analysis.quality import (
    QualityReport,
    assess_quality,
    preprocess,
    print_quality_report,
)
from aquascope.schemas.water_data import Quality


def _make_df(
    n_rows: int = 100,
    add_dupes: bool = False,
    add_nulls: bool = False,
    add_quality: bool = False,
) -> pd.DataFrame:
    """Create a sample DataFrame for quality testing."""
    rng = np.random.default_rng(42)
    stations = [f"ST{i:03d}" for i in range(5)]
    params = ["DO", "BOD5", "COD", "NH3-N", "SS"]
    # Mix enum members (not bare strings) so assess_quality must normalize
    # via ``.value`` rather than ``str(Quality.APPROVED)``.
    quality_cycle = (
        Quality.APPROVED,
        Quality.PROVISIONAL,
        Quality.SUSPECT,
        Quality.ESTIMATED,
        Quality.UNKNOWN,
    )
    rows = []
    for i in range(n_rows):
        val = round(rng.normal(5.0, 2.0), 2)
        row = {
            "source": "taiwan_moenv",
            "station_id": stations[i % 5],
            "parameter": params[i % 5],
            "value": val if not add_nulls or i % 10 != 0 else None,
            "unit": "mg/L",
            "sample_datetime": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}T10:00:00",
        }
        if add_quality:
            row["quality"] = quality_cycle[i % len(quality_cycle)]
        rows.append(row)
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


class TestQualityFlagBreakdown:
    def test_counts_and_fractions_when_quality_present(self):
        df = _make_df(n_rows=100, add_quality=True)
        report = assess_quality(df)

        assert report.quality_counts
        assert set(report.quality_counts) <= {
            "approved",
            "provisional",
            "estimated",
            "suspect",
            "unknown",
        }
        assert "Quality." not in "".join(report.quality_counts)
        assert report.quality_counts.get("provisional", 0) > 0
        assert report.quality_counts.get("suspect", 0) > 0
        assert 0 < report.provisional_fraction < 1
        assert 0 < report.suspect_fraction < 1

    def test_defaults_when_quality_column_absent(self):
        df = _make_df()
        report = assess_quality(df)
        assert report.quality_counts == {}
        assert report.provisional_fraction == 0.0
        assert report.suspect_fraction == 0.0

    def test_print_includes_quality_flags(self):
        df = _make_df(n_rows=100, add_quality=True)
        report = assess_quality(df)
        text = print_quality_report(report)
        assert "Quality Flags" in text
        assert "provisional" in text
        assert "suspect" in text


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
