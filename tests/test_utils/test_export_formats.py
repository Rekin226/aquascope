"""Tests for scientific data format export functions in aquascope.utils.storage."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import BaseModel

from aquascope.utils.storage import export_geojson

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _GeoLoc(BaseModel):
    latitude: float
    longitude: float


class _RecordWithLocation(BaseModel):
    station_id: str
    value: float
    location: _GeoLoc


class _RecordNoLocation(BaseModel):
    station_id: str
    value: float


# ---------------------------------------------------------------------------
# NetCDF
# ---------------------------------------------------------------------------

class TestExportNetCDF:
    def setup_method(self) -> None:
        self.df = pd.DataFrame(
            {"value": [1.0, 2.0, 3.0], "latitude": [25.0, 25.1, 25.2], "longitude": [121.0, 121.1, 121.2]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

    def test_export_netcdf(self, tmp_path: Path) -> None:
        xr = pytest.importorskip("xarray")  # noqa: F841
        from aquascope.utils.storage import export_netcdf

        out = export_netcdf(self.df, tmp_path / "test.nc", variable_name="temperature")
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# GeoJSON
# ---------------------------------------------------------------------------

class TestExportGeoJSON:
    def setup_method(self) -> None:
        self.records = [
            _RecordWithLocation(station_id="S1", value=7.0, location=_GeoLoc(latitude=25.0, longitude=121.5)),
            _RecordWithLocation(station_id="S2", value=6.5, location=_GeoLoc(latitude=24.0, longitude=120.5)),
        ]

    def test_export_geojson(self, tmp_path: Path) -> None:
        out = export_geojson(self.records, tmp_path / "test.geojson")
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 2
        feat = data["features"][0]
        assert feat["type"] == "Feature"
        assert feat["geometry"]["type"] == "Point"
        assert feat["geometry"]["coordinates"] == [121.5, 25.0]
        assert "station_id" in feat["properties"]
        assert "location" not in feat["properties"]

    def test_export_geojson_no_location(self, tmp_path: Path) -> None:
        no_loc_records: list[BaseModel] = [
            _RecordNoLocation(station_id="X1", value=1.0),
            _RecordNoLocation(station_id="X2", value=2.0),
        ]
        out = export_geojson(no_loc_records, tmp_path / "empty.geojson")
        data = json.loads(out.read_text())
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 0


# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------

class TestExportHDF5:
    def setup_method(self) -> None:
        self.df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

    def test_export_hdf5(self, tmp_path: Path) -> None:
        pytest.importorskip("tables")
        from aquascope.utils.storage import export_hdf5

        out = export_hdf5(self.df, tmp_path / "test.h5", key="measurements")
        assert out.exists()
        assert out.stat().st_size > 0
