"""OGC filter mapping for the keyed USGS path (#160).

Checks that the keyed ``fetch_raw`` branch forwards station/parameter/
state/county/huc filters as the OGC API's own query parameters, using a
fake client that records the params sent. No network access.
"""

from __future__ import annotations

import pytest

from aquascope.collectors.usgs import USGSCollector


class FakeClient:
    """Minimal client that records the params sent to ``get_json``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def get_json(self, url, params=None, **kwargs) -> dict:  # noqa: ANN001, ANN002, ANN201
        self.calls.append((url, params or {}))
        return {"features": [], "links": []}


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def collector(fake_client: FakeClient) -> USGSCollector:
    return USGSCollector(api_key="REAL_KEY", client=fake_client)


def test_keyed_forwards_all_filters(collector, fake_client):
    collector.fetch_raw(
        days=7,
        huc="0204",
        parameter="00060",
        stateCd="MD",
        countyCd="24033",
        station_id="01646500",
        bbox="-77.2,38.8,-77.0,39.0",
    )

    assert fake_client.calls[0][0] == "collections/daily/items"
    params = fake_client.calls[0][1]
    assert params["monitoring_location_id"] == "USGS-01646500"
    assert params["parameter_code"] == "00060"
    assert params["state_code"] == "24"
    assert params["county_code"] == "033"
    assert params["hydrologic_unit_code"] == "0204"
    assert params["bbox"] == "-77.2,38.8,-77.0,39.0"
    assert params["api_key"] == "REAL_KEY"


def test_keyed_station_id_maps_to_monitoring_location_id(collector, fake_client):
    collector.fetch_raw(days=7, station_id="01646500")

    params = fake_client.calls[0][1]
    assert params["monitoring_location_id"] == "USGS-01646500"


def test_keyed_prefixed_station_id_has_usgs_prefix(collector, fake_client):
    collector.fetch_raw(days=7, station_id="USGS-01646500")

    params = fake_client.calls[0][1]
    assert params["monitoring_location_id"] == "USGS-01646500"


def test_keyed_prefixed_sites_list_has_usgs_prefix(collector, fake_client):
    collector.fetch_raw(days=7, sites="USGS-01646500, USGS-01646510")

    params = fake_client.calls[0][1]
    assert params["monitoring_location_id"] == "USGS-01646500,USGS-01646510"


def test_keyed_sites_alias_maps_to_monitoring_location_id(collector, fake_client):
    collector.fetch_raw(days=7, sites="01646500,01646510")

    params = fake_client.calls[0][1]
    assert params["monitoring_location_id"] == "USGS-01646500,USGS-01646510"


def test_keyed_monitoring_location_id_kwarg_sends_prefixed_id(collector, fake_client):
    collector.fetch_raw(days=7, monitoring_location_id="01461500")

    params = fake_client.calls[0][1]
    assert params["monitoring_location_id"] == "USGS-01461500"


def test_keyed_prefixed_monitoring_location_id_kwarg_preserved(collector, fake_client):
    collector.fetch_raw(days=7, monitoring_location_id="USGS-01461500")

    params = fake_client.calls[0][1]
    assert params["monitoring_location_id"] == "USGS-01461500"


def test_keyed_numeric_state_code_passes_through(collector, fake_client):
    collector.fetch_raw(days=7, stateCd="24")

    assert fake_client.calls[0][1]["state_code"] == "24"


@pytest.mark.parametrize(
    ("alpha_code", "numeric_code"),
    [
        ("AQ", "60"),
        ("FM", "64"),
        ("MH", "68"),
        ("PW", "70"),
    ],
)
def test_normalise_added_state_codes(alpha_code, numeric_code):
    assert USGSCollector._normalise_state_code(alpha_code) == numeric_code


def test_keyed_three_digit_county_code_passes_through(collector, fake_client):
    collector.fetch_raw(days=7, countyCd="033")

    assert fake_client.calls[0][1]["county_code"] == "033"


def test_keyed_five_digit_county_code_infers_state_code(collector, fake_client, caplog):
    with caplog.at_level("WARNING"):
        collector.fetch_raw(days=7, countyCd="24033")

    params = fake_client.calls[0][1]
    assert params["county_code"] == "033"
    assert params["state_code"] == "24"
    assert not caplog.records


def test_keyed_padded_five_digit_county_code_infers_state_code(collector, fake_client, caplog):
    with caplog.at_level("WARNING"):
        collector.fetch_raw(days=7, countyCd=" 24033")

    params = fake_client.calls[0][1]
    assert params["county_code"] == "033"
    assert params["state_code"] == "24"
    assert not caplog.records


def test_keyed_three_digit_county_code_without_state_warns_multistate(collector, fake_client, caplog):
    with caplog.at_level("WARNING"):
        collector.fetch_raw(days=7, countyCd="033")

    params = fake_client.calls[0][1]
    assert params["county_code"] == "033"
    assert "state_code" not in params
    assert any("multiple states" in r.message for r in caplog.records)


def test_keyed_three_digit_county_code_with_state_does_not_warn(collector, fake_client, caplog):
    with caplog.at_level("WARNING"):
        collector.fetch_raw(days=7, countyCd="033", stateCd="MD")

    params = fake_client.calls[0][1]
    assert params["county_code"] == "033"
    assert params["state_code"] == "24"
    assert not caplog.records


def test_keyed_unmappable_state_code_warns_and_drops(collector, fake_client, caplog):
    with caplog.at_level("WARNING"):
        collector.fetch_raw(days=7, stateCd="XX", station_id="01646500")

    params = fake_client.calls[0][1]
    assert "state_code" not in params
    assert params["monitoring_location_id"] == "USGS-01646500"
    assert any("stateCd" in r.message and "dropped" in r.message for r in caplog.records)


def test_keyed_unmappable_county_code_warns_and_drops(collector, fake_client, caplog):
    with caplog.at_level("WARNING"):
        collector.fetch_raw(days=7, countyCd="24033x", station_id="01646500")

    params = fake_client.calls[0][1]
    assert "county_code" not in params
    assert params["monitoring_location_id"] == "USGS-01646500"
    assert any("countyCd" in r.message and "dropped" in r.message for r in caplog.records)


def test_keyed_comma_state_list_uses_first_value(collector, fake_client, caplog):
    with caplog.at_level("WARNING"):
        collector.fetch_raw(days=7, stateCd="MD,VA")

    params = fake_client.calls[0][1]
    assert params["state_code"] == "24"
    assert any("state_code" in r.message and "MD" in r.message for r in caplog.records)


def test_keyed_comma_county_list_uses_first_value(collector, fake_client, caplog):
    with caplog.at_level("WARNING"):
        collector.fetch_raw(days=7, countyCd="24033,51013")

    params = fake_client.calls[0][1]
    assert params["county_code"] == "033"
    assert params["state_code"] == "24"
    assert any("county_code" in r.message and "24033" in r.message for r in caplog.records)


def test_keyed_comma_huc_list_uses_first_value(collector, fake_client, caplog):
    with caplog.at_level("WARNING"):
        collector.fetch_raw(days=7, huc="0204,0205")

    params = fake_client.calls[0][1]
    assert params["hydrologic_unit_code"] == "0204"
    assert any("hydrologic_unit_code" in r.message and "0204" in r.message for r in caplog.records)


def test_keyed_comma_list_with_invalid_first_warns_list_then_drops(collector, fake_client, caplog):
    with caplog.at_level("WARNING"):
        collector.fetch_raw(days=7, stateCd="XX,MD")

    params = fake_client.calls[0][1]
    assert "state_code" not in params
    messages = [r.message for r in caplog.records]
    assert any("state_code" in m for m in messages)
    list_warn = next(i for i, m in enumerate(messages) if "comma-separated" in m)
    drop_warn = next(i for i, m in enumerate(messages) if "dropped" in m)
    assert list_warn < drop_warn


def test_keyed_comma_sites_and_parameter_lists_left_untouched(collector, fake_client):
    collector.fetch_raw(days=7, sites="01646500,01646510", parameter="00060,00065")

    params = fake_client.calls[0][1]
    assert params["monitoring_location_id"] == "USGS-01646500,USGS-01646510"
    assert params["parameter_code"] == "00060,00065"


def test_keyed_bbox_kwarg_maps_to_bbox_param(collector, fake_client):
    collector.fetch_raw(days=7, bBox="-77.2,38.8,-77.0,39.0")

    assert fake_client.calls[0][1]["bbox"] == "-77.2,38.8,-77.0,39.0"


def test_keyed_parameter_aliases_map_to_parameter_code(collector, fake_client):
    collector.fetch_raw(days=7, parameterCd="00060")

    assert fake_client.calls[0][1]["parameter_code"] == "00060"


def test_keyed_filters_no_longer_warn(collector, caplog):
    with caplog.at_level("WARNING"):
        collector.fetch_raw(
            days=7, huc="0204", parameter="00060",
            stateCd="MD", countyCd="24033", station_id="01646500",
        )

    assert not [r for r in caplog.records if "ignored" in r.message.lower()]


def test_keyed_without_filters_sends_only_core_params(collector, fake_client):
    collector.fetch_raw(days=7, limit=100)

    params = fake_client.calls[0][1]
    assert params == {
        "f": "json",
        "limit": 100,
        "datetime": params["datetime"],
        "api_key": "REAL_KEY",
    }
