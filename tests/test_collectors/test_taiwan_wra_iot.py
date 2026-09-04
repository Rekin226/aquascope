"""Tests for the Taiwan WRA IoT collector (#169).

Covers two things:
1. TLS: ``iot.wra.gov.tw`` chains to the Taiwan Government Root CA, whose
   certs lack the Subject Key Identifier extension — Python 3.13+ rejects
   that under its default strict profile. The collector must create its
   client with ``relax_strict_tls=True`` (never ``verify=False``).
2. Shape: the v2 ``/groundwaterlevel/stations`` endpoint returns a flat list
   of station objects with a nested ``Measurements`` list, not the flat
   per-record shape the old (pre-#169-fix) code assumed. Fixture below is a
   trimmed, real response (captured 2026-08-31), including the API's own
   ``Longtiude`` field-name typo.
"""

from __future__ import annotations

import ssl
import sys
from unittest.mock import MagicMock

import pytest

from aquascope.collectors.taiwan_wra_iot import IOT_BASE, TaiwanWRAIoTCollector
from aquascope.schemas.water_data import DataSource

STATIONS = [
    {
        "IoWStationId": "774cc41f-8fe9-4a67-a5a1-45cb67fc4930",
        "StationId": "A130600GW0703",
        "Name": "金門高中",
        "CountyCode": "09020",
        "CountyName": "福建省金門縣",
        "TownCode": "09020010",
        "TownName": "金城鎮",
        "Latitude": 24.435572,
        "Longtiude": 118.31309,  # sic — API's own typo, not ours
        "AdminName": "經濟部水利署水文技術組",
        "Measurements": [
            {
                "IoWPhysicalQuantityId": "9e174a7c-00a8-4f13-893a-390e64029bb9",
                "TimeStamp": "2026-08-31T17:10:00+08:00",
                "Name": "地下水位",
                "FullName": "即時地下水位-金門高中",
                "SIUnit": "m",
                "Value": 3.697,
            }
        ],
    },
    {
        "IoWStationId": "a5825e0b-5170-4334-9186-3ae29dbbbf3b",
        "StationId": "A130600GW0704",
        "Name": "何浦國小",
        "CountyCode": "09020",
        "CountyName": "福建省金門縣",
        "TownCode": "09020020",
        "TownName": "金沙鎮",
        "Latitude": 24.477219,
        "Longtiude": 118.395485,
        "AdminName": "經濟部水利署水文技術組",
        "Measurements": [
            {
                "IoWPhysicalQuantityId": "361929bf-5c45-41ea-a8d3-8ae955c33e9b",
                "TimeStamp": "2026-08-31T17:10:00+08:00",
                "Name": "地下水位",
                "FullName": "即時地下水位-何浦國小",
                "SIUnit": "m",
                "Value": 15.416,
            }
        ],
    },
    {
        # Station with no readings — must be skipped, not crash.
        "StationId": "A130600GW0999",
        "Name": "No Data Station",
        "CountyName": "測試縣",
        "Latitude": 24.0,
        "Longtiude": 118.0,
        "Measurements": [],
    },
]


def _collector() -> TaiwanWRAIoTCollector:
    col = TaiwanWRAIoTCollector()
    col.client = MagicMock()
    return col


class TestRelaxStrictTLSWiring:
    def test_default_client_relaxes_only_strict_profile(self):
        collector = TaiwanWRAIoTCollector()
        ctx = collector.client._client._transport._pool._ssl_context

        # Full verification stays on ...
        assert ctx.verify_mode == ssl.CERT_REQUIRED
        assert ctx.check_hostname is True
        # ... only the strict X.509 profile check (which rejects certs
        # missing the Subject Key Identifier extension) is relaxed.
        assert not ctx.verify_flags & ssl.VERIFY_X509_STRICT

    def test_client_points_at_iot_base(self):
        collector = TaiwanWRAIoTCollector()
        assert collector.client.base_url == IOT_BASE

    @pytest.mark.skipif(
        sys.version_info < (3, 13),
        reason="VERIFY_X509_STRICT only defaults on in Python 3.13+; the bug "
        "this guards against can't reproduce on older interpreters.",
    )
    def test_never_disables_verification_outright(self):
        # Hard requirement from #169: verify=False is not an acceptable fix,
        # anywhere.
        collector = TaiwanWRAIoTCollector()
        ctx = collector.client._client._transport._pool._ssl_context
        assert ctx.verify_mode != ssl.CERT_NONE


class TestRainfallUnsupported:
    def test_rainfall_raises_notimplementederror(self):
        # v2 API gates rainfall behind paid membership; must fail loudly and
        # explain why, not 404-loop through dead guessed paths.
        with pytest.raises(NotImplementedError, match="rainfall"):
            TaiwanWRAIoTCollector(data_type="rainfall")

    def test_unknown_data_type_raises_valueerror(self):
        with pytest.raises(ValueError):
            TaiwanWRAIoTCollector(data_type="bogus")


class TestFetchRawUsesRealEndpoint:
    def test_fetch_raw_calls_groundwaterlevel_stations(self):
        col = _collector()
        col.client.get_json.return_value = STATIONS
        col.fetch_raw()
        col.client.get_json.assert_called_once_with(
            "groundwaterlevel/stations", headers={"Accept": "application/json"}
        )

    def test_fetch_raw_wraps_non_json_error(self):
        col = _collector()
        col.client.get_json.side_effect = ValueError("Expected JSON but received HTML")
        with pytest.raises(RuntimeError, match="groundwaterlevel/stations"):
            col.fetch_raw()


class TestNormaliseFlattensMeasurements:
    def test_one_sample_per_measurement(self):
        col = _collector()
        samples = col.normalise(STATIONS)
        # Two stations have one reading each; the third has none.
        assert len(samples) == 2

    def test_field_mapping_and_typo_handling(self):
        col = _collector()
        samples = col.normalise(STATIONS)
        s = next(s for s in samples if s.station_id == "A130600GW0703")

        assert s.source == DataSource.TAIWAN_WRA_IOT
        assert s.station_name == "金門高中"
        assert s.county == "福建省金門縣"
        assert s.parameter == "GroundwaterLevel"
        assert s.value == pytest.approx(3.697)
        assert s.unit == "m"
        # Longtiude (API typo) must still populate longitude correctly.
        assert s.location is not None
        assert s.location.latitude == pytest.approx(24.435572)
        assert s.location.longitude == pytest.approx(118.31309)
        assert s.sample_datetime.year == 2026
        assert s.sample_datetime.month == 8
        assert s.sample_datetime.day == 31

    def test_station_with_no_measurements_is_skipped_not_crashed(self):
        col = _collector()
        samples = col.normalise(STATIONS)
        assert all(s.station_id != "A130600GW0999" for s in samples)

    def test_missing_lat_lon_yields_no_location_not_crash(self):
        col = _collector()
        raw = [
            {
                "StationId": "X1",
                "Name": "No Coords",
                "Measurements": [
                    {"TimeStamp": "2026-08-31T00:00:00+08:00", "Value": 1.0, "SIUnit": "m"}
                ],
            }
        ]
        samples = col.normalise(raw)
        assert len(samples) == 1
        assert samples[0].location is None
