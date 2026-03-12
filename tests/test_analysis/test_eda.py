"""Tests for the EDA analysis module."""

import pandas as pd

from aquascope.analysis.eda import (
    EDAReport,
    ParameterStats,
    generate_eda_report,
    print_eda_report,
    profile_dataset,
)


def _make_df(n_rows: int = 100) -> pd.DataFrame:
    """Create a sample water quality DataFrame for testing."""
    import numpy as np

    rng = np.random.default_rng(42)
    stations = [f"ST{i:03d}" for i in range(5)]
    params = ["DO", "BOD5", "COD", "NH3-N", "SS"]
    rows = []
    for i in range(n_rows):
        rows.append({
            "source": "taiwan_moenv",
            "station_id": stations[i % 5],
            "parameter": params[i % 5],
            "value": round(rng.normal(5.0, 2.0), 2),
            "unit": "mg/L",
            "sample_datetime": f"2024-{(i % 12) + 1:02d}-15T10:00:00",
        })
    return pd.DataFrame(rows)


class TestEDA:
    def test_generate_report_returns_report(self):
        df = _make_df()
        report = generate_eda_report(df)
        assert isinstance(report, EDAReport)
        assert report.n_records == 100

    def test_report_counts_stations(self):
        df = _make_df()
        report = generate_eda_report(df)
        assert report.n_stations == 5

    def test_report_counts_parameters(self):
        df = _make_df()
        report = generate_eda_report(df)
        assert report.n_parameters == 5

    def test_report_date_range(self):
        df = _make_df()
        report = generate_eda_report(df)
        assert report.date_range is not None
        assert "2024" in report.date_range[0]

    def test_report_completeness(self):
        df = _make_df()
        report = generate_eda_report(df)
        assert report.completeness_pct > 50.0

    def test_parameter_stats(self):
        df = _make_df()
        report = generate_eda_report(df)
        assert len(report.parameters) == 5
        for ps in report.parameters:
            assert isinstance(ps, ParameterStats)
            assert ps.count > 0
            assert ps.mean != 0 or ps.std != 0

    def test_print_report_returns_string(self):
        df = _make_df()
        report = generate_eda_report(df)
        text = print_eda_report(report)
        assert isinstance(text, str)
        assert "AquaScope" in text
        assert "Records" in text

    def test_profile_dataset(self):
        df = _make_df()
        profile = profile_dataset(df)
        assert len(profile.parameters) == 5
        assert profile.n_records == 100
        assert profile.n_stations == 5

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["parameter", "value", "station_id", "sample_datetime", "source"])
        report = generate_eda_report(df)
        assert report.n_records == 0
        assert report.n_parameters == 0
