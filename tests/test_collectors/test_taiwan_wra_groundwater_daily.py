"""Tests for TaiwanWRAGroundwaterDailyCollector (gweb HydroInfo portal).

The live portal cannot be reached from CI, so these tests inject a fake client
that replays the validated endpoint shapes (area list -> station list ->
history span -> daily chart array).
"""

from __future__ import annotations

from datetime import date

import pytest

from aquascope.collectors.taiwan_wra import (
    _GWEB_AREA_LIST,
    _GWEB_CHART,
    _GWEB_HISTORY,
    _GWEB_STATION_LIST,
    TaiwanWRAGroundwaterDailyCollector,
    _build_gweb_well_metadata,
)
from aquascope.schemas.water_data import DataSource


class _FakeClient:
    """Replays gweb responses for one zone / one well over 2020-2021."""

    SPAN_LO = date(2020, 1, 1)
    SPAN_HI = date(2021, 12, 31)

    def __init__(self):
        self.calls: list[str] = []

    def get_text(self, path, headers=None, use_cache=True):  # session bootstrap
        self.calls.append(f"GET {path}")
        return ""

    def post_json(self, path, json_body=None, headers=None, use_cache=True):
        self.calls.append(f"POST {path}")
        if path == _GWEB_AREA_LIST:
            return [{"Text": "濁水溪沖積扇", "Value": "050"},
                    {"Text": "臺北盆地", "Value": "010"}]
        if path == _GWEB_STATION_LIST:
            assert json_body == {"region": "050"}  # zone filter resolved
            return [{"Text": "東芳(1)", "Value": "07010211"}]
        if path == _GWEB_HISTORY:
            return {"AVG_MIN_DATE": "2020-01-01", "AVG_MAX_DATE": "2021-12-31"}
        if path == _GWEB_CHART:
            start = date.fromisoformat(json_body["startDate"])
            end = date.fromisoformat(json_body["endDate"])
            n = (end - start).days + 1
            # Daily ramp with one missing day to exercise null handling.
            data = [None if i == 5 else round(-1.0 - 0.001 * i, 4) for i in range(n)]
            return {"StationID": json_body["stationNo"], "WaterLevelData": data}
        raise AssertionError(f"unexpected path {path}")


def test_monthly_aggregation_one_well():
    c = TaiwanWRAGroundwaterDailyCollector(zones=["zhuoshui fan"], with_metadata=False,
                                           client=_FakeClient())
    recs = c.collect()
    # 24 months across 2020-2021.
    assert len(recs) == 24
    r0 = recs[0]
    assert r0.source == DataSource.TAIWAN_WRA
    assert r0.station_id == "07010211"
    assert r0.aquifer_name == "濁水溪沖積扇"
    assert r0.measurement_datetime.year == 2020 and r0.measurement_datetime.month == 1
    # stamped mid-month, monotonic time order
    assert r0.measurement_datetime.day == 15
    assert recs[-1].measurement_datetime.year == 2021


def test_daily_aggregation_drops_nulls():
    c = TaiwanWRAGroundwaterDailyCollector(zones=["zhuoshui fan"], aggregate="daily",
                                           with_metadata=False, client=_FakeClient())
    recs = c.collect()
    total_days = (_FakeClient.SPAN_HI - _FakeClient.SPAN_LO).days + 1
    assert len(recs) == total_days - 1  # one null day dropped
    assert recs[0].water_level_m == pytest.approx(-1.0)


def test_zone_alias_and_session_bootstrap():
    fake = _FakeClient()
    c = TaiwanWRAGroundwaterDailyCollector(zones=["050"], with_metadata=False, client=fake)
    c.collect()
    # Session GET happens once before any POST.
    assert fake.calls[0].startswith("GET ")
    assert fake.calls.count(fake.calls[0]) == 1


def test_start_end_clip():
    c = TaiwanWRAGroundwaterDailyCollector(
        zones=["zhuoshui fan"], start="2021-01-01", end="2021-12-31",
        with_metadata=False, client=_FakeClient()
    )
    recs = c.collect()
    assert {r.measurement_datetime.year for r in recs} == {2021}
    assert len(recs) == 12


def test_explicit_stations_skip_zone_discovery():
    fake = _FakeClient()
    c = TaiwanWRAGroundwaterDailyCollector(stations=["07010211"], with_metadata=False,
                                           client=fake)
    c.collect()
    assert f"POST {_GWEB_AREA_LIST}" not in fake.calls
    assert f"POST {_GWEB_STATION_LIST}" not in fake.calls


def test_bad_aggregate_raises():
    with pytest.raises(ValueError, match="aggregate"):
        TaiwanWRAGroundwaterDailyCollector(aggregate="weekly")


def test_well_metadata_builder_keys_by_gw_suffix_and_name():
    rows = [{
        "wellidentifier": "3132014GW07010211", "wellname": "東芳(1)",
        "locationbytwd97": "200779.08 2662059.14", "welldepth": "48.0",
    }]
    meta = _build_gweb_well_metadata(rows)
    assert "code::07010211" in meta  # gweb id == suffix after GW
    assert "name::東芳(1)" in meta
    assert meta["code::07010211"]["well_depth_m"] == pytest.approx(48.0)
    # TWD97 -> WGS84 lands inside Taiwan (pyproj optional; skip if absent).
    loc = meta["code::07010211"]["location"]
    if loc is not None:
        assert 21.5 <= loc.latitude <= 26.5 and 118.0 <= loc.longitude <= 122.5


def test_metadata_join_populates_location_and_depth():
    from aquascope.schemas.water_data import GeoLocation

    c = TaiwanWRAGroundwaterDailyCollector(zones=["zhuoshui fan"], with_metadata=False,
                                           client=_FakeClient())
    # Inject a pre-built metadata index (skips the open-data fetch).
    c._well_meta = {
        "code::07010211": {
            "location": GeoLocation(latitude=23.5, longitude=120.3),
            "well_depth_m": 48.0,
        }
    }
    recs = c.collect()
    assert recs[0].location is not None
    assert recs[0].location.latitude == pytest.approx(23.5)
    assert recs[0].well_depth_m == pytest.approx(48.0)
