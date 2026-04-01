"""Tests for the EU Water Framework Directive (WFD) collector."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from aquascope.collectors.eu_wfd import (
    _EQS_THRESHOLDS,
    WFD_STATUS_CLASSES,
    EUWFDCollector,
    WFDComplianceResult,
    check_wfd_compliance,
)
from aquascope.schemas.water_data import DataSource, WaterQualitySample

SAMPLE_RAW = [
    {
        "monitoringSiteIdentifier": "DE_SITE_001",
        "waterBodyName": "Rhine at Cologne",
        "countryCode": "DE",
        "parameterWaterBodyCategory": "Dissolved oxygen",
        "resultMeanValue": 8.5,
        "resultUom": "mg/L",
        "phenomenonTimeSamplingDate": "2023-06-15",
        "lat": 50.94,
        "lon": 6.96,
    },
    {
        "monitoringSiteIdentifier": "FR_SITE_042",
        "waterBodyName": "Seine at Paris",
        "countryCode": "FR",
        "parameterWaterBodyCategory": "BOD5",
        "resultMeanValue": 3.2,
        "resultUom": "mg/L",
        "phenomenonTimeSamplingDate": "2023-07-20T10:30:00",
        "lat": 48.86,
        "lon": 2.35,
    },
]


# ── normalise tests ──────────────────────────────────────────────────


class TestEUWFDNormalise:
    def setup_method(self):
        self.collector = EUWFDCollector()
        self.raw = list(SAMPLE_RAW)

    def test_normalise_returns_water_quality_samples(self):
        records = self.collector.normalise(self.raw)
        assert len(records) == 2
        assert all(isinstance(r, WaterQualitySample) for r in records)

    def test_normalise_sets_correct_source(self):
        records = self.collector.normalise(self.raw)
        for r in records:
            assert r.source == DataSource.EU_WFD
            assert r.source.value == "eu_wfd"

    def test_normalise_maps_station_fields(self):
        records = self.collector.normalise(self.raw)
        assert records[0].station_id == "DE_SITE_001"
        assert records[0].station_name == "Rhine at Cologne"
        assert records[1].station_id == "FR_SITE_042"

    def test_normalise_maps_parameter_and_value(self):
        records = self.collector.normalise(self.raw)
        assert records[0].parameter == "Dissolved oxygen"
        assert records[0].value == 8.5
        assert records[0].unit == "mg/L"

    def test_normalise_parses_date_only(self):
        records = self.collector.normalise(self.raw)
        assert records[0].sample_datetime == datetime(2023, 6, 15)

    def test_normalise_parses_datetime_with_time(self):
        records = self.collector.normalise(self.raw)
        assert records[1].sample_datetime == datetime(2023, 7, 20, 10, 30, 0)

    def test_normalise_parses_geolocation(self):
        records = self.collector.normalise(self.raw)
        loc = records[0].location
        assert loc is not None
        assert abs(loc.latitude - 50.94) < 0.01
        assert abs(loc.longitude - 6.96) < 0.01

    def test_normalise_sets_country_as_county(self):
        records = self.collector.normalise(self.raw)
        assert records[0].county == "DE"
        assert records[1].county == "FR"

    def test_normalise_skips_missing_value(self):
        raw_with_none = [
            {"monitoringSiteIdentifier": "X", "resultMeanValue": None, "phenomenonTimeSamplingDate": "2023-01-01"},
        ]
        assert self.collector.normalise(raw_with_none) == []

    def test_normalise_skips_missing_date(self):
        raw_no_date = [{"monitoringSiteIdentifier": "X", "resultMeanValue": 5.0, "phenomenonTimeSamplingDate": ""}]
        assert self.collector.normalise(raw_no_date) == []

    def test_normalise_empty_input(self):
        assert self.collector.normalise([]) == []


# ── fetch_raw tests ──────────────────────────────────────────────────


class TestEUWFDFetchRaw:
    def setup_method(self):
        self.collector = EUWFDCollector()

    def test_fetch_raw_handles_connection_error(self):
        """API unavailability returns empty list with no exception."""
        import httpx

        with patch.object(
            self.collector.client, "get_json", side_effect=httpx.ConnectError("offline")
        ):
            result = self.collector.fetch_raw(country="DE")
        assert result == []

    def test_build_query_country_filter(self):
        q = self.collector._build_query(country="DE")
        assert "countryCode = 'DE'" in q

    def test_build_query_year_filter(self):
        q = self.collector._build_query(year=2023)
        assert "2023" in q

    def test_build_query_water_body_river(self):
        q = self.collector._build_query(water_body_type="river")
        assert "waterBodyCategory = 'RW'" in q

    def test_build_query_water_body_lake(self):
        q = self.collector._build_query(water_body_type="lake")
        assert "waterBodyCategory = 'LW'" in q

    def test_build_query_water_body_groundwater(self):
        q = self.collector._build_query(water_body_type="groundwater")
        assert "waterBodyCategory = 'GW'" in q

    def test_fetch_raw_with_results_key(self):
        """API returning {'results': [...]} is unwrapped correctly."""
        fake_data = {"results": [{"monitoringSiteIdentifier": "A"}]}
        with patch.object(self.collector.client, "get_json", return_value=fake_data):
            result = self.collector.fetch_raw()
        assert result == [{"monitoringSiteIdentifier": "A"}]

    def test_fetch_raw_with_list_response(self):
        """API returning a bare list is returned as-is."""
        fake_data = [{"monitoringSiteIdentifier": "B"}]
        with patch.object(self.collector.client, "get_json", return_value=fake_data):
            result = self.collector.fetch_raw()
        assert result == fake_data


# ── WFD compliance tests ─────────────────────────────────────────────


def _make_sample(parameter: str, value: float) -> WaterQualitySample:
    """Helper to create a WaterQualitySample for compliance tests."""
    return WaterQualitySample(
        source=DataSource.EU_WFD,
        station_id="TEST",
        sample_datetime=datetime(2023, 1, 1),
        parameter=parameter,
        value=value,
        unit="mg/L",
    )


class TestWFDCompliance:
    def test_dissolved_oxygen_good(self):
        samples = [_make_sample("Dissolved oxygen", 7.0)]
        result = check_wfd_compliance(samples, "Dissolved oxygen")
        assert isinstance(result, WFDComplianceResult)
        assert result.n_compliant == 1
        assert result.compliance_pct == 100.0

    def test_dissolved_oxygen_moderate(self):
        samples = [_make_sample("Dissolved oxygen", 5.0)]
        result = check_wfd_compliance(samples, "Dissolved oxygen")
        assert result.n_compliant == 0

    def test_bod5_good(self):
        samples = [_make_sample("BOD5", 3.0)]
        result = check_wfd_compliance(samples, "BOD5")
        assert result.n_compliant == 1

    def test_bod5_moderate(self):
        samples = [_make_sample("BOD5", 5.0)]
        result = check_wfd_compliance(samples, "BOD5")
        assert result.n_compliant == 0

    def test_phosphate_good(self):
        samples = [_make_sample("Phosphate", 0.05)]
        result = check_wfd_compliance(samples, "Phosphate")
        assert result.n_compliant == 1

    def test_nitrate_good(self):
        samples = [_make_sample("Nitrate", 20.0)]
        result = check_wfd_compliance(samples, "Nitrate")
        assert result.n_compliant == 1

    def test_ammonium_good(self):
        samples = [_make_sample("Ammonium", 0.2)]
        result = check_wfd_compliance(samples, "Ammonium")
        assert result.n_compliant == 1

    def test_ph_good(self):
        samples = [_make_sample("pH", 7.5)]
        result = check_wfd_compliance(samples, "pH")
        assert result.n_compliant == 1
        assert result.eqs_threshold == "6.0-9.0"

    def test_ph_out_of_range(self):
        samples = [_make_sample("pH", 5.0)]
        result = check_wfd_compliance(samples, "pH")
        assert result.n_compliant == 0

    def test_unknown_parameter_raises(self):
        samples = [_make_sample("Dissolved oxygen", 7.0)]
        try:
            check_wfd_compliance(samples, "FakeParam")
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "FakeParam" in str(exc)

    def test_empty_samples(self):
        result = check_wfd_compliance([], "Dissolved oxygen")
        assert result.n_samples == 0
        assert result.status_class == "Unknown"
        assert result.compliance_pct == 0.0

    def test_status_class_high(self):
        """100 % compliance → 'High' status."""
        samples = [_make_sample("Dissolved oxygen", 9.0) for _ in range(20)]
        result = check_wfd_compliance(samples, "Dissolved oxygen")
        assert result.status_class == "High"

    def test_status_class_bad(self):
        """0 % compliance → 'Bad' status."""
        samples = [_make_sample("Dissolved oxygen", 2.0) for _ in range(20)]
        result = check_wfd_compliance(samples, "Dissolved oxygen")
        assert result.status_class == "Bad"

    def test_status_class_moderate_mix(self):
        """~60 % compliance → 'Moderate'."""
        good = [_make_sample("Nitrate", 10.0) for _ in range(12)]
        bad = [_make_sample("Nitrate", 50.0) for _ in range(8)]
        result = check_wfd_compliance(good + bad, "Nitrate")
        assert result.status_class == "Moderate"


class TestWFDStatusClasses:
    def test_status_classes_dict_has_five_entries(self):
        assert len(WFD_STATUS_CLASSES) == 5

    def test_eqs_thresholds_cover_all_parameters(self):
        expected = {"Dissolved oxygen", "BOD5", "Phosphate", "Nitrate", "Ammonium", "pH"}
        assert set(_EQS_THRESHOLDS.keys()) == expected
