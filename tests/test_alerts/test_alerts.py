"""Comprehensive tests for the AquaScope threshold alerting system."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

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
from aquascope.schemas.water_data import DataSource, WaterQualitySample


class TestThresholdData:
    """Verify threshold database completeness and correctness."""

    def test_who_has_at_least_10_parameters(self):
        assert len(WHO_THRESHOLDS) >= 10

    def test_epa_has_at_least_10_parameters(self):
        assert len(EPA_THRESHOLDS) >= 10

    def test_eu_wfd_has_at_least_10_parameters(self):
        assert len(EU_WFD_THRESHOLDS) >= 10

    def test_all_thresholds_is_union(self):
        all_params = set(ALL_THRESHOLDS.keys())
        who_params = set(WHO_THRESHOLDS.keys())
        epa_params = set(EPA_THRESHOLDS.keys())
        eu_params = set(EU_WFD_THRESHOLDS.keys())
        assert who_params.issubset(all_params)
        assert epa_params.issubset(all_params)
        assert eu_params.issubset(all_params)

    def test_threshold_dataclass_fields(self):
        t = WHO_THRESHOLDS["nitrate"][0]
        assert isinstance(t, Threshold)
        assert t.parameter == "nitrate"
        assert t.limit == 50.0
        assert t.unit == "mg/L"
        assert t.standard == "WHO"
        assert t.category == "drinking"
        assert isinstance(t.description, str)

    def test_list_standards_returns_all_three(self):
        standards = list_standards()
        assert "WHO" in standards
        assert "EPA" in standards
        assert "EU_WFD" in standards

    def test_list_parameters_no_filter(self):
        params = list_parameters()
        assert "nitrate" in params
        assert "dissolved_oxygen" in params
        assert params == sorted(params)

    def test_list_parameters_with_filter(self):
        who_params = list_parameters(standard="WHO")
        assert "nitrate" in who_params
        eu_params = list_parameters(standard="EU_WFD")
        assert "BOD" in eu_params

    def test_get_thresholds_no_filter(self):
        results = get_thresholds("nitrate")
        assert len(results) >= 2
        standards = {t.standard for t in results}
        assert len(standards) >= 2

    def test_get_thresholds_with_standard(self):
        results = get_thresholds("nitrate", standard="WHO")
        assert all(t.standard == "WHO" for t in results)

    def test_get_thresholds_unknown_param(self):
        results = get_thresholds("nonexistent_param")
        assert results == []


class TestSeverity:
    """Test severity_from_exceedance logic."""

    def test_compliant(self):
        assert severity_from_exceedance(0.5) == "info"
        assert severity_from_exceedance(0.99) == "info"

    def test_warning_range(self):
        assert severity_from_exceedance(1.0) == "warning"
        assert severity_from_exceedance(1.2) == "warning"
        assert severity_from_exceedance(1.49) == "warning"

    def test_critical_range(self):
        assert severity_from_exceedance(1.5) == "critical"
        assert severity_from_exceedance(2.0) == "critical"
        assert severity_from_exceedance(10.0) == "critical"


class TestCheckSample:
    """Test check_sample with WaterQualitySample objects."""

    def setup_method(self):
        self.compliant_sample = WaterQualitySample(
            source=DataSource.USGS,
            station_id="TEST001",
            station_name="Test Station",
            sample_datetime=datetime(2025, 1, 15, 10, 0),
            parameter="nitrate",
            value=5.0,
            unit="mg/L",
        )
        self.non_compliant_sample = WaterQualitySample(
            source=DataSource.USGS,
            station_id="TEST002",
            station_name="Bad Station",
            sample_datetime=datetime(2025, 1, 15, 10, 0),
            parameter="nitrate",
            value=75.0,
            unit="mg/L",
        )
        self.critical_lead_sample = WaterQualitySample(
            source=DataSource.USGS,
            station_id="TEST003",
            sample_datetime=datetime(2025, 1, 15, 10, 0),
            parameter="lead",
            value=0.05,
            unit="mg/L",
        )

    def test_compliant_sample_no_alerts(self):
        alerts = check_sample(self.compliant_sample)
        assert alerts == []

    def test_non_compliant_sample_triggers_alerts(self):
        alerts = check_sample(self.non_compliant_sample)
        assert len(alerts) > 0
        assert all(isinstance(a, Alert) for a in alerts)
        assert any(a.severity in ("warning", "critical") for a in alerts)

    def test_alert_contains_correct_metadata(self):
        alerts = check_sample(self.non_compliant_sample, standards=["WHO"])
        assert len(alerts) >= 1
        alert = alerts[0]
        assert alert.parameter == "nitrate"
        assert alert.value == 75.0
        assert alert.station_id == "TEST002"
        assert alert.timestamp == datetime(2025, 1, 15, 10, 0)
        assert alert.exceedance_ratio == 75.0 / 50.0

    def test_standard_filter(self):
        alerts_who = check_sample(self.non_compliant_sample, standards=["WHO"])
        alerts_epa = check_sample(self.non_compliant_sample, standards=["EPA"])
        for a in alerts_who:
            assert a.threshold.standard == "WHO"
        for a in alerts_epa:
            assert a.threshold.standard == "EPA"

    def test_critical_lead_severity(self):
        alerts = check_sample(self.critical_lead_sample, standards=["WHO"])
        assert len(alerts) >= 1
        assert any(a.severity == "critical" for a in alerts)

    def test_dissolved_oxygen_below_minimum(self):
        low_do = WaterQualitySample(
            source=DataSource.USGS,
            station_id="DO001",
            sample_datetime=datetime(2025, 3, 1),
            parameter="dissolved_oxygen",
            value=2.0,
            unit="mg/L",
        )
        alerts = check_sample(low_do, standards=["WHO"])
        assert len(alerts) >= 1
        assert alerts[0].parameter == "dissolved_oxygen"

    def test_dissolved_oxygen_above_minimum_no_alert(self):
        good_do = WaterQualitySample(
            source=DataSource.USGS,
            station_id="DO002",
            sample_datetime=datetime(2025, 3, 1),
            parameter="dissolved_oxygen",
            value=8.0,
            unit="mg/L",
        )
        alerts = check_sample(good_do)
        assert alerts == []


class TestCheckDataFrame:
    """Test check_dataframe with mixed data."""

    def setup_method(self):
        self.df = pd.DataFrame({
            "parameter": ["nitrate", "nitrate", "lead", "dissolved_oxygen", "pH"],
            "value": [5.0, 75.0, 0.05, 2.0, 7.0],
            "station_id": ["S1", "S2", "S3", "S4", "S5"],
            "sample_datetime": pd.to_datetime([
                "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05",
            ]),
        })

    def test_report_type(self):
        report = check_dataframe(self.df)
        assert isinstance(report, AlertReport)

    def test_total_samples(self):
        report = check_dataframe(self.df)
        assert report.total_samples == 5

    def test_samples_with_alerts(self):
        report = check_dataframe(self.df, standards=["WHO"])
        assert report.samples_with_alerts >= 2

    def test_parameters_checked(self):
        report = check_dataframe(self.df)
        assert "nitrate" in report.parameters_checked
        assert "lead" in report.parameters_checked

    def test_summary_counts(self):
        report = check_dataframe(self.df, standards=["WHO"])
        assert "critical" in report.summary
        assert "warning" in report.summary
        assert "info" in report.summary
        total = sum(report.summary.values())
        assert total == len(report.alerts)

    def test_standards_used(self):
        report = check_dataframe(self.df, standards=["WHO", "EPA"])
        assert "WHO" in report.standards_used
        assert "EPA" in report.standards_used

    def test_empty_dataframe(self):
        empty = pd.DataFrame({"parameter": [], "value": []})
        report = check_dataframe(empty)
        assert report.total_samples == 0
        assert report.alerts == []

    def test_all_compliant(self):
        clean = pd.DataFrame({
            "parameter": ["nitrate", "nitrate"],
            "value": [1.0, 2.0],
        })
        report = check_dataframe(clean, standards=["WHO"])
        assert report.alerts == []
        assert report.samples_with_alerts == 0


class TestCheckTimeseries:
    """Test check_timeseries for a single parameter."""

    def setup_method(self):
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        self.df = pd.DataFrame(
            {"value": [5.0, 55.0, 60.0, 3.0, 80.0]},
            index=dates,
        )

    def test_detects_exceedances(self):
        report = check_timeseries(self.df, parameter="nitrate", standards=["WHO"])
        assert len(report.alerts) >= 2

    def test_report_structure(self):
        report = check_timeseries(self.df, parameter="nitrate", standards=["WHO"])
        assert report.total_samples == 5
        assert report.parameters_checked == ["nitrate"]
        assert "WHO" in report.standards_used


class TestNotifier:
    """Test notification dispatch."""

    def setup_method(self):
        self.alerts = [
            Alert(
                parameter="nitrate",
                value=75.0,
                threshold=Threshold("nitrate", 50.0, "mg/L", "WHO", "drinking", "Max nitrate"),
                severity="warning",
                exceedance_ratio=1.5,
                timestamp=datetime(2025, 1, 15),
                station_id="S1",
                message="nitrate = 75.0 mg/L exceeds WHO limit of 50.0 mg/L (ratio 1.50)",
            ),
            Alert(
                parameter="lead",
                value=0.05,
                threshold=Threshold("lead", 0.01, "mg/L", "WHO", "drinking", "Max lead"),
                severity="critical",
                exceedance_ratio=5.0,
                timestamp=datetime(2025, 1, 15),
                station_id="S2",
                message="lead = 0.05 mg/L exceeds WHO limit of 0.01 mg/L (ratio 5.00)",
            ),
        ]

    def test_notify_log_only(self):
        config = NotificationConfig()
        results = notify(self.alerts, config)
        assert results["log"] is True
        assert "file" not in results
        assert "webhook" not in results

    def test_notify_file_channel(self):
        log_path = Path("test_alert_output.jsonl")
        try:
            config = NotificationConfig(log_file=str(log_path))
            results = notify(self.alerts, config)
            assert results["file"] is True
            assert log_path.exists()

            lines = log_path.read_text().strip().split("\n")
            assert len(lines) == 2
            record = json.loads(lines[0])
            assert record["parameter"] == "nitrate"
            assert record["severity"] == "warning"
            assert record["standard"] == "WHO"
        finally:
            log_path.unlink(missing_ok=True)

    def test_notify_empty_alerts(self):
        config = NotificationConfig()
        results = notify([], config)
        assert results == {}
