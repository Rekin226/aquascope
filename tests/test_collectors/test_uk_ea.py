from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from aquascope.collectors.uk_ea import (
    MAPPED_OBSERVED_PROPERTY_UNITS,
    UKEACollector,
)
from aquascope.schemas.water_data import (
    DataSource,
    GeoLocation,
    StreamflowReading,
    WaterLevelReading,
    WaterQualitySample,
)


class DummyClient:
    def __init__(self, behaviour=None):
        # behaviour can be a dict mapping (path, offset) to return values or a callable
        self.behaviour = behaviour or {}
        self.calls = []

    def get_json(self, path, params=None):
        params = params or {}
        self.calls.append((path, dict(params)))

        # if behaviour is callable, let it handle
        if callable(self.behaviour):
            return self.behaviour(path, params)

        # if specific path provided
        key = (path, params.get("_offset"))
        if key in self.behaviour:
            return self.behaviour[key]

        # default: return empty items
        return {"items": []}


def test_parse_bbox_valid_and_invalid():
    # Valid Inputs
    assert UKEACollector._parse_bbox("2.0, 51.1, 3.3, 52.7") == (2.0, 51.1, 3.3, 52.7)
    assert UKEACollector._parse_bbox("  2,51, 3,52 ") == (2.0, 51.0, 3.0, 52.0)

    # Invalid Inputs
    assert UKEACollector._parse_bbox(123) is None
    assert UKEACollector._parse_bbox("1,2,3") is None
    assert UKEACollector._parse_bbox("a,b,c,d") is None


def test_extract_station_suid_from_measure_id():
    assert UKEACollector._extract_station_suid_from_measure_id(None) is None
    # Produce a measure string that contains a station's SUID in the first 36 characters
    measure = "m" * 100
    assert UKEACollector._extract_station_suid_from_measure_id(measure) == measure[:36]


def test_extract_observed_property_from_measure_id():
    assert UKEACollector._extract_observed_property_from_measure_id(None) is None
    measure = "a" * 36 + "-flow-123"
    assert UKEACollector._extract_observed_property_from_measure_id(measure) == "flow"
    assert UKEACollector._extract_observed_property_from_measure_id("a" * 36 + "_") is None


def test_fetch_raw_with_measure_sets_observed_property_and_supports_normalisation():
    suid = "".join(["s" for _ in range(36)])
    item = {
        "measure": {"@id": f"http://measures/{suid}-flow-info"},
        "value": "3.14",
        "dateTime": "2025-03-01T10:00:00",
        "completeness": "N/A",
        "quality": "Good",
    }

    def behaviour(path, params):
        if path == "id/stations.json":
            return {"items": []}
        if params.get("_offset", 0) == 0:
            return {"items": [item]}
        return {"items": []}

    collector = UKEACollector(client=DummyClient(behaviour=behaviour))
    raw = collector.fetch_raw(measure=f"{suid}-flow-info")

    assert raw[0]["observedProperty"] == "waterFlow"
    assert raw[0]["measure"] == f"{suid}-flow-info"

    records = collector.normalise(raw)
    assert len(records) == 1
    assert isinstance(records[0], StreamflowReading)
    assert records[0].discharge_cms == pytest.approx(3.14)
    assert records[0].source_type == "in_situ"


def test_fetch_raw_with_max_items_none():
    suid = "".join(["s" for _ in range(36)])
    item = {
        "measure": {"@id": f"http://measures/{suid}-measure-info"},
        "value": "1.0",
        "dateTime": "2025-01-01T00:00:00",
    }

    def behaviour(path, params):
        if path == "id/stations.json":
            return {"items": []}
        if params.get("_offset", 0) == 0:
            return {"items": [item]}
        return {"items": []}

    collector = UKEACollector(client=DummyClient(behaviour=behaviour))
    raw = collector.fetch_raw(observed_property="waterLevel", max_items=None, limit=1)

    assert raw[0]["observedProperty"] == "waterLevel"
    assert len(raw) >= 2


def test_build_location_from_lat_long_and_invalid():
    loc = UKEACollector._build_location_from_lat_long(51.5, -0.12)
    assert isinstance(loc, GeoLocation)
    assert loc.latitude == pytest.approx(51.5)
    assert loc.longitude == pytest.approx(-0.12)

    # Non-numeric values produce None
    assert UKEACollector._build_location_from_lat_long("not-a-number", "x") is None
    assert UKEACollector._build_location_from_lat_long("not-a-number", 1) is None


def test_extract_water_quality_and_water_level_metadata():
    # Typical metadata for a water quality reading
    water_quality_metadata = {"label": "Stn A", "riverName": "River B", "lat": "51.5", "long": "-0.12"}
    stn_name, river, location = UKEACollector._extract_water_quality_sample_metadata(water_quality_metadata)
    assert stn_name == "Stn A"
    assert river == "River B"
    assert isinstance(location, GeoLocation)

    # Typical metadata for a water level reading, which does not include a river name
    water_level_metadata = {"label": "Stn B", "lat": "51.6", "long": "-0.13"}
    stn_name2, location2 = UKEACollector._extract_water_level_reading_metadata(water_level_metadata)
    assert stn_name2 == "Stn B"
    assert isinstance(location2, GeoLocation)

    # Typical metadata for a streamflow reading, including catchment area
    streamflow_metadata = {
        "label": "Stn C",
        "riverName": "River C",
        "lat": "51.7",
        "long": "-0.14",
        "catchmentArea": "123.4"
    }
    stn_name3, river3, location3, catchment_area = UKEACollector._extract_streamflow_reading_metadata(
        streamflow_metadata
    )
    assert stn_name3 == "Stn C"
    assert river3 == "River C"
    assert isinstance(location3, GeoLocation)
    assert catchment_area == pytest.approx(123.4)

    # Zero-valued coordinates should still build a location.
    zero_coords_metadata = {"label": "Zero", "lat": 0, "long": 0}
    stn_name3, location3 = UKEACollector._extract_water_level_reading_metadata(zero_coords_metadata)
    assert stn_name3 == "Zero"
    assert isinstance(location3, GeoLocation)
    assert location3.latitude == pytest.approx(0.0)
    assert location3.longitude == pytest.approx(0.0)

    # Empty metadata
    empty_metadata = {}
    stn_name3, river3, loc3 = UKEACollector._extract_water_quality_sample_metadata(empty_metadata)
    assert stn_name3 is None
    assert river3 is None
    assert loc3 is None

    stn_name3, loc3 = UKEACollector._extract_water_level_reading_metadata(empty_metadata)
    assert stn_name3 is None
    assert loc3 is None


def test_extract_reading_data_success_and_failure():
    # valid item
    suid = "".join(["s" for _ in range(36)])
    item = {
        "measure": {"@id": f"http://measures/{suid}-measure-info"},
        "value": "12.3",
        "dateTime": "2025-01-01T12:00:00",
        "completeness": "Complete",
        "quality": "Good",
    }
    station_suid, value, sample_dt, remark = UKEACollector._extract_reading_data(item)
    assert station_suid == suid[:36]
    assert value == "12.3"
    assert isinstance(sample_dt, datetime)
    assert sample_dt == datetime.fromisoformat(item["dateTime"])
    assert remark == "Data Completeness: Complete; Data Quality: Good."

    # Missing measure should raise ValueError
    bad_item = {
        "dateTime": "2025-01-01T12:00:00",
        "value": "1.0",
    }
    with pytest.raises(ValueError):
        UKEACollector._extract_reading_data(bad_item)

    # Missing value should raise ValueError
    bad_item = {
        "measure": {"@id": f"http://measures/{suid}-measure-info"},
        "dateTime": "2025-01-01T12:00:00",
    }
    with pytest.raises(ValueError):
        UKEACollector._extract_reading_data(bad_item)

    # Missing date should raise ValueError
    bad_item2 = {
        "measure": {"@id": f"http://measures/{suid}-measure-info"},
        "value": "1.0",
    }
    with pytest.raises(ValueError):
        UKEACollector._extract_reading_data(bad_item2)


def test_compute_date_range_variants():
    # Both None -> last 30 days
    min_d, max_d = UKEACollector._compute_date_range(None, None, None)
    end = date.today()
    start = end - timedelta(days=30)
    assert min_d == start.isoformat()
    assert max_d == end.isoformat()

    # min_date only
    min_only = "2020-01-01"
    _, max_d2 = UKEACollector._compute_date_range(min_only, None, None)
    expected_max = date.fromisoformat(min_only) + timedelta(days=30)
    if expected_max > date.today():
        expected_max = date.today()
    assert max_d2 == expected_max.isoformat()

    # max_date only
    max_only = "2020-02-01"
    min_d3, _ = UKEACollector._compute_date_range(None, max_only, None)
    expected_min = date.fromisoformat(max_only) - timedelta(days=30)
    assert min_d3 == expected_min.isoformat()

    # min_date, max_date and days provided -> days ignored and input returned
    min_in = "2020-01-01"
    max_in = "2020-02-01"
    min_out, max_out = UKEACollector._compute_date_range(min_in, max_in, 10)
    assert min_out == min_in
    assert max_out == max_in


def test_fetch_station_metadata_behaviour(monkeypatch):
    # no station or station_wiski_id
    collector = UKEACollector(client=DummyClient())
    assert collector._fetch_station_metadata() is None

    # client raises exception
    def raise_exc(path, params):
        raise RuntimeError("boom")

    c = DummyClient(behaviour=raise_exc)
    collector2 = UKEACollector(client=c)
    assert collector2._fetch_station_metadata(station="s") is None

    # client returns multiple items
    items = [{"label": "A"}, {"label": "B"}]
    d = DummyClient(behaviour={("id/stations.json", None): {"items": items}})
    coll = UKEACollector(client=d)
    meta = coll._fetch_station_metadata(station="s")
    assert meta == items[0]

    # client returns single item
    items2 = [{"label": "Only"}]
    d2 = DummyClient(behaviour={("id/stations.json", None): {"items": items2}})
    coll2 = UKEACollector(client=d2)
    meta2 = coll2._fetch_station_metadata(station_wiski_id="w")
    assert meta2 == items2[0]


def test_fetch_raw_errors_and_behaviour(monkeypatch):
    # missing observed_property and measure -> ValueError
    coll = UKEACollector(client=DummyClient())
    with pytest.raises(ValueError):
        coll.fetch_raw()

    # invalid bbox string -> ValueError
    with pytest.raises(ValueError):
        coll.fetch_raw(observed_property="waterLevel", bbox="1,2,3")

    # client.get_json raises -> returns []
    def bad_behaviour(path, params):
        raise RuntimeError("network")

    bad_client = DummyClient(behaviour=bad_behaviour)
    coll_bad = UKEACollector(client=bad_client)
    res = coll_bad.fetch_raw(observed_property="waterLevel")
    assert res == []

    # pagination and station metadata injection
    suid = "".join(["s" for _ in range(36)])
    item1 = {
        "measure": {"@id": f"http://measures/{suid}-measure-info"},
        "value": "1.1",
        "dateTime": "2025-01-01T01:00:00"
    }
    item2 = {
        "measure":{"@id": f"http://measures/{suid}-measure-info"},
        "value": "2.2",
        "dateTime": "2025-01-02T01:00:00"
    }

    def behaviour(path, params):
        if path == "id/stations.json":
            return {"items": [{"label": "S1"}]}
        # simulate one page then empty
        if params.get("_offset", 0) == 0:
            return {"items": [item1, item2]}
        return {"items": []}

    client = DummyClient(behaviour=behaviour)
    coll2 = UKEACollector(client=client)
    all_items = coll2.fetch_raw(observed_property="waterLevel", station="SOMEID", limit=2)
    # first element is params metadata
    assert isinstance(all_items, list)
    assert isinstance(all_items[0], dict)
    # subsequent items are the returned ones and should have _station injected
    assert all("_station" in it for it in all_items[1:])

    # test measure+collection: collection ignored
    def behaviour2(path, params):
        return {"items": []}

    cli = DummyClient(behaviour=behaviour2)
    coll3 = UKEACollector(client=cli)
    out = coll3.fetch_raw(
        observed_property="waterLevel",
        measure=f"{suid}-flow-123",
        collection="15min"
    )
    # first entry contains params, should NOT include 'period' because collection ignored when measure present
    assert "period" not in out[0]
    assert "measure" in out[0]

    # test max_items truncation: with max_items small
    def behaviour3(path, params):
        # always return 2 items per page
        return {"items": [item1, item2]}

    cli2 = DummyClient(behaviour=behaviour3)
    coll4 = UKEACollector(client=cli2)
    # max_items counts the prepended metadata row as well, so the result can contain 3 entries.
    res2 = coll4.fetch_raw(observed_property="waterLevel", limit=2, max_items=2)
    assert len(res2) == 3


def test_fetch_raw_with_bbox_queries_two_stations():
    station1 = {
        "stationGuid": "s" * 36,
        "label": "Station 1",
        "lat": "51.1",
        "long": "0.1",
    }
    station2 = {
        "stationGuid": "t" * 36,
        "label": "Station 2",
        "lat": "51.2",
        "long": "0.2",
    }
    item1 = {
        "measure": {"@id": "http://measures/ssssssssssssssssssssssssssssssssssss-measure-info"},
        "value": "1.1",
        "dateTime": "2025-01-01T01:00:00",
    }
    item2 = {
        "measure": {"@id": "http://measures/tttttttttttttttttttttttttttttttttttt-measure-info"},
        "value": "2.2",
        "dateTime": "2025-01-02T01:00:00",
    }

    def behaviour(path, params):
        if path == "id/stations.json":
            assert params["_limit"] == 100
            assert params["observedProperty"] == "waterLevel"
            assert params["mineq-long"] == 0.0
            assert params["mineq-lat"] == 51.0
            assert params["maxeq-long"] == 1.0
            assert params["maxeq-lat"] == 51.1
            if params.get("_offset", 0) == 0:
                return {"items": [station1]}
            return {"items": []}

        if path == "data/readings.json":
            assert params["_limit"] == 10000
            assert params["observedProperty"] == "waterLevel"
            assert params["station"] == station1["stationGuid"]
            assert "mineq-long" not in params
            assert "mineq-lat" not in params
            assert "maxeq-long" not in params
            assert "maxeq-lat" not in params
            if params.get("_offset", 0) == 0:
                return {"items": [item1] if params["station"] == station1["stationGuid"] else [item2]}
            return {"items": []}

        return {"items": []}

    client = DummyClient(behaviour=behaviour)
    collector = UKEACollector(client=client)
    raw = collector.fetch_raw(observed_property="waterLevel", bbox="0.0,51.0,1.0,51.1", limit=10000)

    assert len(raw) == 2
    assert raw[1]["_station"]["stationGuid"] == station1["stationGuid"]

    station_calls = [call for call in client.calls if call[0] == "id/stations.json"]
    assert len(station_calls) == 2
    assert station_calls[0][1]["_offset"] == 0
    assert station_calls[1][1]["_offset"] == 100

    data_calls = [call for call in client.calls if call[0] == "data/readings.json"]
    assert len(data_calls) == 2
    assert {call[1]["station"] for call in data_calls} == {station1["stationGuid"]}
    assert [call[1]["_offset"] for call in data_calls] == [0, 10000]


def test_fetch_raw_fetches_station_metadata_for_wiski_id_only():
    suid = "".join(["s" for _ in range(36)])
    item = {
        "measure": {"@id": f"http://measures/{suid}-measure-info"},
        "value": "1.1",
        "dateTime": "2025-01-01T01:00:00",
    }

    def behaviour(path, params):
        if path == "id/stations.json":
            return {"items": [{"label": "Station A", "lat": "51.5", "long": "-0.1"}]}
        return {"items": [item]}

    client = DummyClient(behaviour=behaviour)
    coll = UKEACollector(client=client)
    raw = coll.fetch_raw(observed_property="waterLevel", station_wiski_id="wiski-123")

    assert any(path == "id/stations.json" for path, _ in client.calls)
    assert raw[1]["_station"]["label"] == "Station A"


def test_fetch_raw_measure_only_populates_observed_property_metadata():
    measure = "s" * 36 + "-flow-123"

    def behaviour(path, params):
        return {"items": []}

    client = DummyClient(behaviour=behaviour)
    coll = UKEACollector(client=client)
    out = coll.fetch_raw(measure=measure)

    assert len(out) == 1
    assert out[0]["measure"] == measure
    assert out[0]["observedProperty"] == "waterFlow"


def test_fetch_raw_max_items_none_returns_all_items():
    suid = "".join(["s" for _ in range(36)])
    item1 = {
        "measure": {"@id": f"http://measures/{suid}-measure-info"},
        "value": "1.1",
        "dateTime": "2025-01-01T01:00:00"
    }
    item2 = {
        "measure": {"@id": f"http://measures/{suid}-measure-info"},
        "value": "2.2",
        "dateTime": "2025-01-02T01:00:00"
    }

    def behaviour(path, params):
        if params.get("_offset", 0) == 0:
            return {"items": [item1, item2]}
        return {"items": []}

    client = DummyClient(behaviour=behaviour)
    coll = UKEACollector(client=client)
    out = coll.fetch_raw(observed_property="waterLevel", limit=2, max_items=None)

    assert len(out) == 3
    assert out[0]["_limit"] == 2
    assert out[1] == item1
    assert out[2] == item2


def test_normalise_streamflow_and_water_quality_and_level_and_skipping():
    suid = "".join(["s" for _ in range(36)])
    # waterFlow (StreamflowReading)
    request_meta = {"observedProperty": "waterFlow"}
    item = {
        "measure": {"@id": f"http://measures/{suid}-measure-info"},
        "value": "3.14",
        "dateTime": "2025-03-01T10:00:00",
        "completeness": "N/A",
        "category": "Good",
        "_station": {"label": "QStn", "riverName": "BigRiver", "lat": "51.0", "long": "-0.1", "catchmentArea": "123.4"},
    }
    coll = UKEACollector(client=DummyClient())
    samples = coll.normalise([request_meta, item])
    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, StreamflowReading)
    assert sample.source == DataSource.UK_EA
    assert sample.station_id == suid[:36]
    assert sample.discharge_cms == pytest.approx(3.14)
    assert sample.unit == MAPPED_OBSERVED_PROPERTY_UNITS["waterFlow"]
    assert sample.station_name == "QStn"
    assert isinstance(sample.location, GeoLocation)
    assert sample.source_type == "in_situ"
    assert sample.catchment_area_km2 == pytest.approx(123.4)

    # rainfall (WaterQualitySample)
    request_meta_rain = {"observedProperty": "rainfall"}
    item_rain = dict(item)
    item_rain.update({"value": "0.5"})
    item_rain["_station"] = {"label": "RStn", "lat": "51.1", "long": "-0.11"}
    rain_samples = coll.normalise([request_meta_rain, item_rain])
    assert isinstance(rain_samples[0], WaterQualitySample)

    # waterLevel (WaterLevelReading)
    request_meta_lvl = {"observedProperty": "waterLevel"}
    item_lvl = {
        "measure": {"@id": f"http://measures/{suid}-measure-info"},
        "value": "1.23",
        "dateTime": "2025-03-02T11:00:00",
        "completeness": "90%",
        "category": "Fair",
        "_station": {"label": "LStn", "lat": "51.2", "long": "-0.12"},
    }
    lvl_samples = coll.normalise([request_meta_lvl, item_lvl])
    assert len(lvl_samples) == 1
    lvl = lvl_samples[0]
    assert isinstance(lvl, WaterLevelReading)
    assert lvl.water_level == pytest.approx(1.23)
    assert "Parameter: waterLevel" in lvl.remark

    # groundwaterLevel (WaterLevelReading)
    request_meta_gw = {"observedProperty": "groundwaterLevel"}
    item_gw = {
        "measure": {"@id": f"http://measures/{suid}-measure-info"},
        "value": "2.34",
        "dateTime": "2025-03-03T12:00:00",
        "completeness": "Complete",
        "category": "Good",
        "_station": {"label": "GWStn", "lat": "51.3", "long": "-0.13"},
    }
    gw_samples = coll.normalise([request_meta_gw, item_gw])
    assert len(gw_samples) == 1
    gw = gw_samples[0]
    assert isinstance(gw, WaterLevelReading)
    assert gw.water_level == pytest.approx(2.34)

    # ensure bad items are skipped (e.g., missing value)
    bad_item = {"measure": {"@id": f"http://measures/{suid}-measure-info"}, "dateTime": "2025-03-01T10:00:00"}
    skipped = coll.normalise([request_meta, bad_item])
    assert skipped == []

    request_meta_rain = {"observedProperty": "rainfall"}
    item_rain = dict(item)
    item_rain.update({"value": "0.5"})
    item_rain["_station"] = {"label": "RStn", "lat": "51.1", "long": "-0.11"}
    rain_samples = coll.normalise([request_meta_rain, item_rain])
