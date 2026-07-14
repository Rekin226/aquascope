"""Tests for the UK Environment Agency hydrology collector."""

from __future__ import annotations

from unittest.mock import MagicMock

from aquascope.collectors.uk_ea import UKEACollector
from aquascope.schemas.water_data import DataSource


class TestUKEACollector:
    def test_init(self):
        collector = UKEACollector()
        assert collector.name == "uk_ea"

    def test_fetch_raw_builds_request_params(self):
        mock_client = MagicMock()
        mock_client.get_json.return_value = {"items": []}
        collector = UKEACollector(client=mock_client)

        collector.fetch_raw(
            observed_property="waterFlow",
            min_date="2024-01-01",
            max_date="2024-01-02",
            limit=10,
            max_items=100,
        )

        mock_client.get_json.assert_called_once()
        _, kwargs = mock_client.get_json.call_args
        assert kwargs["params"]["observedProperty"] == "waterFlow"
        assert kwargs["params"]["observedProperty"] == "waterFlow"
        assert kwargs["params"]["min-date"] == "2024-01-01"
        assert kwargs["params"]["max-date"] == "2024-01-02"
        assert kwargs["params"]["_limit"] == 10
        assert kwargs["params"]["_offset"] == 0

    def test_fetch_raw_paginates_with_offset(self):
        responses = [
            {
                "items": [
                    {
                        "measure": {
                            "@id": (
                                "http://environment.data.gov.uk/hydrology/id/measures/"
                                "00000000-0000-0000-0000-000000000000-flow-i-900-m3s-qualified"
                            ),
                            "parameter": "flow",
                        }
                    }
                ]
            },
            {
                "items": [
                    {
                        "measure": {
                            "@id": (
                                "http://environment.data.gov.uk/hydrology/id/measures/"
                                "00000000-0000-0000-0000-000000000000-flow-i-900-m3s-qualified"
                            ),
                            "parameter": "flow",
                        }
                    }
                ]
            },
            {"items": []},
        ]
        recorded_params: list[dict] = []

        def get_json(path, params=None, **kwargs):
            recorded_params.append(dict(params or {}))
            return responses[len(recorded_params) - 1]

        mock_client = MagicMock()
        mock_client.get_json.side_effect = get_json
        collector = UKEACollector(client=mock_client)

        result = collector.fetch_raw(observed_property="waterFlow", limit=1, max_items=None)

        assert len(result) == 2
        assert len(recorded_params) == 3
        assert recorded_params[0]["_offset"] == 0
        assert recorded_params[1]["_offset"] == 1
        assert recorded_params[2]["_offset"] == 2

    def test_normalise_valid_item(self):
        raw = [
            {
                "measure": {
                    "@id": "http://environment.data.gov.uk/hydrology/id/measures/052d0819-2a32-47df-9b99-c243c9c8235b-flow-i-900-m3s-qualified",
                    "parameter": "flow",
                },
                "dateTime": "2024-07-14T12:00:00",
                "value": "10.3",
                "quality": "Unchecked",
            }
        ]
        collector = UKEACollector()
        samples = collector.normalise(raw)

        assert len(samples) == 1
        sample = samples[0]
        assert sample.source == DataSource.UK_EA
        assert sample.station_id == "052d0819-2a32-47df-9b99-c243c9c8235b"
        assert sample.parameter == "Flow"
        assert sample.value == 10.3
        assert sample.unit == "m3/s"
        assert sample.remark == "Unchecked"

    def test_normalise_uses_station_metadata(self):
        raw = [
            {
                "measure": {
                    "@id": "http://environment.data.gov.uk/hydrology/id/measures/052d0819-2a32-47df-9b99-c243c9c8235b-flow-i-900-m3s-qualified",
                    "parameter": "flow",
                },
                "dateTime": "2024-07-14T12:00:00",
                "value": 15.4,
                "_station": {
                    "stationGuid": "052d0819-2a32-47df-9b99-c243c9c8235b",
                    "label": "Ulting Sarasota",
                    "riverName": "River Chelmer",
                    "lat": 51.746683,
                    "long": 0.624437,
                },
            }
        ]
        collector = UKEACollector()
        samples = collector.normalise(raw)

        assert len(samples) == 1
        sample = samples[0]
        assert sample.station_name == "Ulting Sarasota"
        assert sample.basin == "River Chelmer"
        assert sample.location is not None
        assert sample.location.latitude == 51.746683
        assert sample.location.longitude == 0.624437

    def test_normalise_skips_missing_values(self):
        raw = [
            {
                "measure": {
                    "@id": "http://environment.data.gov.uk/hydrology/id/measures/052d0819-2a32-47df-9b99-c243c9c8235b-flow-i-900-m3s-qualified",
                    "parameter": "flow",
                },
                "dateTime": "2024-07-14T12:00:00",
                "value": None,
            }
        ]
        collector = UKEACollector()
        samples = collector.normalise(raw)

        assert samples == []
