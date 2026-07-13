"""Tests for the xarray / GeoPandas interop bridge (aquascope.io.interop).

Guarded by importorskip so a bare core install skips cleanly; CI installs the
interop/spatial/scientific extras and runs them."""

from __future__ import annotations

from datetime import datetime

import pytest

pytest.importorskip("xarray", reason="xarray not installed (aquascope[interop])")
pytest.importorskip("geopandas", reason="geopandas not installed (aquascope[interop])")

from aquascope.collectors.base import BaseCollector
from aquascope.io.interop import records_to_geodataframe, records_to_xarray
from aquascope.schemas.water_data import (
    DataSource,
    GeoLocation,
    StreamflowReading,
    WaterLevelReading,
    WaterQualitySample,
)


def _sf(station, dt, discharge, source_type="in_situ", loc=None):
    return StreamflowReading(
        source=DataSource.GRDC,
        station_id=station,
        reading_datetime=dt,
        discharge_cms=discharge,
        source_type=source_type,
        location=loc,
    )


def _wq(station, dt, param, value, unit="mg/l", loc=None):
    return WaterQualitySample(
        source=DataSource.WQP,
        station_id=station,
        sample_datetime=dt,
        parameter=param,
        value=value,
        unit=unit,
        location=loc,
    )


def _wl(station, dt, level, loc=None):
    return WaterLevelReading(
        source=DataSource.TAIWAN_WRA,
        station_id=station,
        reading_datetime=dt,
        water_level=level,
        location=loc,
    )


class TestRecordsToXarray:
    def test_water_quality_dims_and_vars(self):
        loc = GeoLocation(latitude=24.1, longitude=120.7)
        recs = [
            _wq("S1", datetime(2026, 1, 1), "DO", 8.5, loc=loc),
            _wq("S1", datetime(2026, 1, 2), "DO", 8.1, loc=loc),
            _wq("S1", datetime(2026, 1, 1), "pH", 7.2, unit="", loc=loc),
        ]
        ds = records_to_xarray(recs)
        assert set(ds.dims) == {"time", "station_id"}
        assert "DO" in ds.data_vars and "pH" in ds.data_vars
        assert ds["DO"].attrs.get("units") == "mg/l"
        assert ds["DO"].sel(time=datetime(2026, 1, 1), station_id="S1").item() == 8.5
        assert ds.attrs["source"] == "wqp"

    def test_station_coords_present(self):
        loc = GeoLocation(latitude=24.1, longitude=120.7)
        ds = records_to_xarray([_wq("S1", datetime(2026, 1, 1), "DO", 8.5, loc=loc)])
        assert float(ds["lat"].sel(station_id="S1")) == 24.1
        assert float(ds["lon"].sel(station_id="S1")) == 120.7

    def test_missing_location_gives_nan_coords(self):
        ds = records_to_xarray([_wq("S1", datetime(2026, 1, 1), "DO", 8.5, loc=None)])
        import math

        assert math.isnan(float(ds["lat"].sel(station_id="S1")))

    def test_water_level_records(self):
        ds = records_to_xarray(
            [
                _wl("G1", datetime(2026, 1, 1), 12.3),
                _wl("G1", datetime(2026, 1, 2), 12.9),
            ]
        )
        assert "water_level" in ds.data_vars
        assert ds["water_level"].attrs.get("units") == "m"
        assert set(ds.dims) == {"time", "station_id"}

    def test_streamflow_records(self):
        ds = records_to_xarray(
            [
                _sf("6435060", datetime(2026, 1, 1), 45.2),
                _sf("6435060", datetime(2026, 2, 1), 47.8),
            ]
        )
        assert "discharge" in ds.data_vars
        assert ds["discharge"].attrs.get("units") == "m3/s"
        assert set(ds.dims) == {"time", "station_id"}

    def test_streamflow_does_not_collide_with_water_level_branch(self):
        # Regression test: StreamflowReading shares reading_datetime with
        # WaterLevelReading, so discharge_cms must be checked first or this
        # raises AttributeError trying to read r.water_level.
        ds = records_to_xarray([_sf("6435060", datetime(2026, 1, 1), 45.2)])
        assert "water_level" not in ds.data_vars
        assert "discharge" in ds.data_vars

    def test_streamflow_station_coords_present(self):
        loc = GeoLocation(latitude=51.2, longitude=7.9)
        ds = records_to_xarray([_sf("6435060", datetime(2026, 1, 1), 45.2, loc=loc)])
        assert float(ds["lat"].sel(station_id="6435060")) == 51.2
        assert float(ds["lon"].sel(station_id="6435060")) == 7.9

    def test_empty_input(self):
        ds = records_to_xarray([])
        assert len(ds.data_vars) == 0

    def test_unsupported_type_raises(self):
        from aquascope.schemas.water_data import ReservoirStatus

        rec = ReservoirStatus(
            source=DataSource.TAIWAN_WRA, reservoir_name="R1", date=datetime(2026, 1, 1)
        )
        with pytest.raises(TypeError):
            records_to_xarray([rec])


class TestRecordsToGeoDataFrame:
    def test_geometry_and_crs(self):
        loc = GeoLocation(latitude=24.1, longitude=120.7)
        gdf = records_to_geodataframe([_wq("S1", datetime(2026, 1, 1), "DO", 8.5, loc=loc)])
        assert gdf.crs.to_epsg() == 4326
        assert gdf.geometry.iloc[0].x == 120.7
        assert gdf.geometry.iloc[0].y == 24.1
        assert gdf["latitude"].iloc[0] == 24.1
        assert gdf["source"].iloc[0] == "wqp"

    def test_missing_location_null_geometry_not_dropped(self):
        gdf = records_to_geodataframe(
            [
                _wq("S1", datetime(2026, 1, 1), "DO", 8.5, loc=None),
                _wq(
                    "S2",
                    datetime(2026, 1, 1),
                    "DO",
                    8.0,
                    loc=GeoLocation(latitude=1.0, longitude=2.0),
                ),
            ]
        )
        assert len(gdf) == 2
        assert gdf.geometry.iloc[0] is None
        assert gdf.geometry.iloc[1] is not None

    def test_empty_input(self):
        gdf = records_to_geodataframe([])
        assert len(gdf) == 0
        assert gdf.crs.to_epsg() == 4326


class _FakeCollector(BaseCollector):
    name = "fake"

    def fetch_raw(self, **kwargs):
        return [1]

    def normalise(self, raw):
        loc = GeoLocation(latitude=24.1, longitude=120.7)
        return [_wq("S1", datetime(2026, 1, 1), "DO", 8.5, loc=loc)]


class TestCollectConvenience:
    def test_collect_as_xarray(self):
        import xarray as xr

        out = _FakeCollector().collect(as_xarray=True)
        assert isinstance(out, xr.Dataset)
        assert "DO" in out.data_vars

    def test_collect_as_geodataframe(self):
        import geopandas as gpd

        out = _FakeCollector().collect(as_geodataframe=True)
        assert isinstance(out, gpd.GeoDataFrame)
        assert len(out) == 1

    def test_collect_default_returns_records(self):
        out = _FakeCollector().collect()
        assert isinstance(out, list)
        assert out[0].parameter == "DO"

    def test_mutually_exclusive(self):
        with pytest.raises(ValueError):
            _FakeCollector().collect(as_xarray=True, as_geodataframe=True)
