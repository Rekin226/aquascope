from __future__ import annotations

from aquascope.collectors.usgs import USGSCollector, _map_usgs_quality
from aquascope.schemas.water_data import Quality, StreamflowReading, WaterLevelReading, WaterQualitySample


def _feature(param_code="00060", value="500", **extra_props):
    props = {
        "monitoring_location_id": "USGS-01646500",
        "parameter_code": param_code,
        "value": value,
        "time": "2026-06-01T00:00:00Z",
        "unit_of_measure": "ft3/s",
        **extra_props,
    }
    return {"type": "Feature", "geometry": None, "properties": props}


def test_maps_approved():
    readings = USGSCollector(api_key="X").normalise(
        [_feature(approval_status="Approved", qualifier="")]
    )
    assert isinstance(readings[0], StreamflowReading)
    assert readings[0].quality == Quality.APPROVED
    assert readings[0].quality_raw


def test_maps_provisional():
    readings = USGSCollector(api_key="X").normalise(
        [_feature(approval_status="Provisional", qualifier="")]
    )
    assert readings[0].quality == Quality.PROVISIONAL


def test_maps_ice_affected_as_suspect_even_if_approved():
    readings = USGSCollector(api_key="X").normalise(
        [_feature(approval_status="Approved", qualifier="Ice affected")]
    )
    assert readings[0].quality == Quality.SUSPECT
    assert "ice" in readings[0].quality_raw.lower()


def test_defaults_to_unknown_when_source_says_nothing():
    readings = USGSCollector(api_key="X").normalise([_feature()])
    assert readings[0].quality == Quality.UNKNOWN
    assert readings[0].quality_raw is None


def test_quality_raw_is_kept_verbatim():
    q, raw = _map_usgs_quality({"approval_status": "Provisional", "qualifier": "Estimated"})
    assert q == Quality.ESTIMATED
    assert "Provisional" in raw and "Estimated" in raw


def test_maps_quality_for_water_quality_sample():
    readings = USGSCollector(api_key="X").normalise(
        [
            _feature(
                param_code="00010",  # temperature — not streamflow/water level
                value="22.5",
                approval_status="Approved",
                qualifier="",
                unit_of_measure="deg C",
            )
        ]
    )

    assert len(readings) == 1
    assert isinstance(readings[0], WaterQualitySample)
    assert readings[0].quality == Quality.APPROVED


def test_maps_approved_water_level():
    readings = USGSCollector(api_key="X").normalise(
        [
            _feature(
                param_code="00065",
                value="10.5",
                unit_of_measure="ft",
                approval_status="Approved",
                qualifier="",
            )
        ]
    )
    assert len(readings) == 1
    assert isinstance(readings[0], WaterLevelReading)
    assert readings[0].quality == Quality.APPROVED
    assert "Approved" in readings[0].quality_raw


def test_maps_provisional_water_level():
    readings = USGSCollector(api_key="X").normalise(
        [
            _feature(
                param_code="00065",
                value="10.5",
                unit_of_measure="ft",
                approval_status="Provisional",
                qualifier="",
            )
        ]
    )
    assert isinstance(readings[0], WaterLevelReading)
    assert readings[0].quality == Quality.PROVISIONAL
    assert "Provisional" in readings[0].quality_raw


def test_maps_ice_affected_water_level_as_suspect():
    readings = USGSCollector(api_key="X").normalise(
        [
            _feature(
                param_code="00065",
                value="10.5",
                unit_of_measure="ft",
                approval_status="Approved",
                qualifier="Ice affected",
            )
        ]
    )
    assert isinstance(readings[0], WaterLevelReading)
    assert readings[0].quality == Quality.SUSPECT
    assert "ice" in readings[0].quality_raw.lower()


def test_maps_estimated_water_level():
    readings = USGSCollector(api_key="X").normalise(
        [
            _feature(
                param_code="00065",
                value="10.5",
                unit_of_measure="ft",
                approval_status="Provisional",
                qualifier="Estimated",
            )
        ]
    )
    assert isinstance(readings[0], WaterLevelReading)
    assert readings[0].quality == Quality.ESTIMATED
    assert "Provisional" in readings[0].quality_raw
    assert "Estimated" in readings[0].quality_raw
