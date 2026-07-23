from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from aquascope.collectors.uk_ea import (
    UKEACollector,
    MAPPED_OBSERVED_PROPERTIES,
    MAPPED_OBSERVED_PROPERTY_UNITS,
    COLLECTION_PERIOD_VALUES,
)
from aquascope.schemas.water_data import GeoLocation, WaterQualitySample, WaterLevelReading, DataSource


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


def test_build_location_from_lat_lon_and_invalid():
    loc = UKEACollector._build_location_from_lat_lon(51.5, -0.12)
    assert isinstance(loc, GeoLocation)
    assert loc.latitude == pytest.approx(51.5)
    assert loc.longitude == pytest.approx(-0.12)

    # Non-numeric values produce None
    assert UKEACollector._build_location_from_lat_lon("not-a-number", "x") is None
    assert UKEACollector._build_location_from_lat_lon("not-a-number", 1) is None


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
    suid = "".join(["g" for _ in range(40)])
    item1 = {"measure": {"@id": f"http://measures/{suid}-measure-info"}, "value": "1.1", "dateTime": "2025-01-01T01:00:00"}
    item2 = {"measure": {"@id": f"http://measures/{suid}-measure-info"}, "value": "2.2", "dateTime": "2025-01-02T01:00:00"}

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
    out = coll3.fetch_raw(observed_property="waterLevel", measure="M-SUID-123456789012345678901234567890123456", collection="15min")
    # first entry contains params, should NOT include 'period' because collection ignored when measure present
    assert "period" not in out[0]
    assert "measure" in out[0]

    # test max_items truncation: with max_items small
    def behaviour3(path, params):
        # always return 2 items per page
        return {"items": [item1, item2]}

    cli2 = DummyClient(behaviour=behaviour3)
    coll4 = UKEACollector(client=cli2)
    # set max_items to 2 (remember first entry is params so this will truncate quickly)
    res2 = coll4.fetch_raw(observed_property="waterLevel", limit=2, max_items=2)
    assert len(res2) <= 2


def test_normalise_water_quality_and_level_and_skipping():
    suid = "".join(["g" for _ in range(40)])
    # water quality (waterFlow)
    request_meta = {"observedProperty": "waterFlow"}
    item = {
        "measure": {"@id": f"http://measures/{suid}-measure-info"},
        "value": "3.14",
        "dateTime": "2025-03-01T10:00:00",
        "completeness": "N/A",
        "category": "Good",
        "_station": {"label": "QStn", "riverName": "BigRiver", "lat": "51.0", "long": "-0.1"},
    }
    coll = UKEACollector(client=DummyClient())
    samples = coll.normalise([request_meta, item])
    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, WaterQualitySample)
    assert sample.source == DataSource.UK_EA
    assert sample.station_id == suid[:36]
    assert sample.value == pytest.approx(3.14)
    assert sample.unit == MAPPED_OBSERVED_PROPERTY_UNITS["waterFlow"]
    assert sample.station_name == "QStn"
    assert isinstance(sample.location, GeoLocation)

    # rainfall sets river to N/A
    request_meta_rain = {"observedProperty": "rainfall"}
    item_rain = dict(item)
    item_rain.update({"value": "0.5"})
    item_rain["_station"] = {"label": "RStn", "riverName": "ShouldBeIgnored", "lat": "51.1", "long": "-0.11"}
    rain_samples = coll.normalise([request_meta_rain, item_rain])
    assert rain_samples[0].river == "N/A"

    # water level reading
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

    # ensure bad items are skipped (e.g., missing value)
    bad_item = {"measure": {"@id": f"http://measures/{suid}-measure-info"}, "dateTime": "2025-03-01T10:00:00"}
    skipped = coll.normalise([request_meta, bad_item])
    assert skipped == []
