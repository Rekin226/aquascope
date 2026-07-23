"""Tests for the aquascope.io module — WaterML 2.0, HEC, and SWMM formats."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from aquascope.io.hec import HECDSSRecord, dataframe_to_hec_format, write_hec_dss_csv, write_hec_ras_flow
from aquascope.io.swmm import write_swmm_rainfall, write_swmm_timeseries
from aquascope.io.waterml import (
    WaterMLTimeSeries,
    dataframe_to_waterml,
    read_waterml,
    waterml_to_dataframe,
    write_waterml,
)
from aquascope.schemas.water_data import DataSource, GeoLocation, WaterQualitySample


def _make_ts(
    station_id: str = "ST001",
    station_name: str = "Test Station",
    parameter: str = "Discharge",
    unit: str = "m3/s",
    n: int = 5,
) -> WaterMLTimeSeries:
    """Helper — build a simple WaterMLTimeSeries for tests."""
    return WaterMLTimeSeries(
        station_id=station_id,
        station_name=station_name,
        parameter=parameter,
        unit=unit,
        latitude=25.03,
        longitude=121.56,
        timestamps=[datetime(2024, 1, 1, h, tzinfo=timezone.utc) for h in range(n)],
        values=[float(i * 10) for i in range(n)],
        quality_codes=["good"] * n,
        metadata={"description": "test data"},
    )


def _make_dataframe(n: int = 5) -> pd.DataFrame:
    """Helper — build a simple DataFrame matching WaterML column conventions."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "station_id": ["ST001"] * n,
            "station_name": ["Test Station"] * n,
            "parameter": ["Discharge"] * n,
            "value": [float(i * 10) for i in range(n)],
            "unit": ["m3/s"] * n,
            "quality_code": ["good"] * n,
            "latitude": [25.03] * n,
            "longitude": [121.56] * n,
        }
    )


# ── WaterML tests ────────────────────────────────────────────────────────────
class TestWaterMLRoundtrip:
    def test_write_read_roundtrip(self, tmp_path: Path):
        """Write a WaterMLTimeSeries, read it back, verify equality."""
        ts = _make_ts()
        out = tmp_path / "roundtrip.xml"
        write_waterml(ts, out)
        result = read_waterml(out)

        assert len(result) == 1
        r = result[0]
        assert r.station_id == ts.station_id
        assert r.station_name == ts.station_name
        assert r.parameter == ts.parameter
        assert r.unit == ts.unit
        assert r.latitude == ts.latitude
        assert r.longitude == ts.longitude
        assert len(r.timestamps) == len(ts.timestamps)
        assert r.values == ts.values
        assert r.quality_codes == ts.quality_codes

    def test_read_waterml_multiple_series(self, tmp_path: Path):
        """A file with two time series should return two objects."""
        ts1 = _make_ts(station_id="A1", parameter="Discharge")
        ts2 = _make_ts(station_id="A2", parameter="pH")
        out = tmp_path / "multi.xml"
        write_waterml([ts1, ts2], out)

        result = read_waterml(out)
        assert len(result) == 2
        ids = {r.station_id for r in result}
        assert ids == {"A1", "A2"}

    def test_waterml_to_dataframe(self):
        """Verify DataFrame columns and values produced from time series."""
        ts = _make_ts(n=3)
        df = waterml_to_dataframe([ts])

        assert list(df.columns) == [
            "timestamp", "station_id", "parameter", "value",
            "unit", "quality_code", "latitude", "longitude",
        ]
        assert len(df) == 3
        assert df["station_id"].iloc[0] == "ST001"
        assert df["value"].iloc[1] == 10.0

    def test_dataframe_to_waterml(self):
        """Roundtrip DataFrame → WaterMLTimeSeries → DataFrame."""
        df = _make_dataframe(n=4)
        ts_list = dataframe_to_waterml(df)

        assert len(ts_list) == 1
        ts = ts_list[0]
        assert ts.station_id == "ST001"
        assert ts.parameter == "Discharge"
        assert len(ts.values) == 4

    def test_write_waterml_validates_xml(self, tmp_path: Path):
        """Output must be parseable by ElementTree."""
        ts = _make_ts()
        out = tmp_path / "valid.xml"
        write_waterml(ts, out)

        tree = ET.parse(out)
        root = tree.getroot()
        assert "Collection" in root.tag

    def test_waterml_metadata_preserved(self, tmp_path: Path):
        """Station name, unit, and quality codes survive a roundtrip."""
        ts = _make_ts(station_name="My River Gauge", unit="ft3/s")
        ts.quality_codes = ["good", "suspect", "good", "good", "good"]
        out = tmp_path / "meta.xml"
        write_waterml(ts, out)
        result = read_waterml(out)

        r = result[0]
        assert r.station_name == "My River Gauge"
        assert r.unit == "ft3/s"
        assert r.quality_codes[1] == "suspect"

    def test_waterml_empty_timeseries(self, tmp_path: Path):
        """An empty time series should write and read back without error."""
        ts = WaterMLTimeSeries(
            station_id="EMPTY",
            station_name="Empty Station",
            parameter="Temp",
            unit="C",
        )
        out = tmp_path / "empty.xml"
        write_waterml(ts, out)
        result = read_waterml(out)

        assert len(result) == 1
        assert result[0].station_id == "EMPTY"
        assert result[0].values == []
        assert result[0].timestamps == []


# ── HEC format tests ─────────────────────────────────────────────────────────
class TestHECFormat:
    def test_write_hec_dss_csv(self, tmp_path: Path):
        """Verify CSV output format with header and data rows."""
        rec = HECDSSRecord(
            pathname="/WATERSHED/LOC/FLOW/01JAN2024/1HOUR/SRC/",
            timestamps=[datetime(2024, 1, 1, h, tzinfo=timezone.utc) for h in range(3)],
            values=[100.0, 110.0, 105.0],
            unit="CFS",
            data_type="INST-VAL",
        )
        out = tmp_path / "dss.csv"
        write_hec_dss_csv([rec], out)

        lines = out.read_text().strip().splitlines()
        assert lines[0] == "pathname,timestamp,value,unit,type"
        assert len(lines) == 4  # header + 3 data rows
        assert "WATERSHED" in lines[1]
        assert "CFS" in lines[1]

    def test_dataframe_to_hec_format(self):
        """Verify pathname construction from DataFrame."""
        df = _make_dataframe(n=3)
        records = dataframe_to_hec_format(df, watershed="BASIN1")

        assert len(records) == 1
        rec = records[0]
        assert rec.pathname.startswith("/BASIN1/")
        assert "/Discharge/" in rec.pathname
        assert "/AQUASCOPE/" in rec.pathname
        assert len(rec.values) == 3

    def test_write_hec_ras_flow(self, tmp_path: Path):
        """Verify unsteady flow file has required header lines and data."""
        discharge = np.array([100.0, 120.0, 115.0])
        timestamps = pd.DatetimeIndex(
            [datetime(2024, 1, 1, h, tzinfo=timezone.utc) for h in range(3)]
        )
        out = tmp_path / "flow.u01"
        write_hec_ras_flow(discharge, timestamps, "TestRiver", "Reach1", "100", out)

        text = out.read_text()
        assert "Flow Title=AquaScope Generated Flow Data" in text
        assert "River Rch & Prof=TestRiver,Reach1,100" in text
        assert "Flow Hydrograph= 3" in text
        assert "100.00" in text

    def test_hec_pathname_format(self):
        """Verify ``/A/B/C/D/E/F/`` structure."""
        df = _make_dataframe(n=2)
        records = dataframe_to_hec_format(df, watershed="W", location="L")
        rec = records[0]

        parts = rec.pathname.split("/")
        # pathname: /A/B/C/D/E/F/ → split gives ['', A, B, C, D, E, F, '']
        assert parts[0] == ""
        assert parts[-1] == ""
        assert len(parts) == 8  # 6 parts + 2 empty


# ── SWMM tests ───────────────────────────────────────────────────────────────
class TestSWMMFormat:
    def test_write_swmm_timeseries(self, tmp_path: Path):
        """Verify SWMM timeseries output format with header."""
        df = _make_dataframe(n=3)
        out = tmp_path / "ts.dat"
        write_swmm_timeseries(df, "Rain1", out)

        text = out.read_text()
        assert "[TIMESERIES]" in text
        assert ";;Name" in text
        assert "Rain1" in text
        lines = [ln for ln in text.splitlines() if ln.startswith("Rain1")]
        assert len(lines) == 3

    def test_write_swmm_rainfall(self, tmp_path: Path):
        """Verify .dat format: StationID Year Month Day Hour Minute Value."""
        df = _make_dataframe(n=3)
        out = tmp_path / "rain.dat"
        write_swmm_rainfall(df, "RG01", out)

        text = out.read_text()
        lines = [ln for ln in text.strip().splitlines() if ln]
        assert len(lines) == 3
        first = lines[0]
        assert first.startswith("RG01")
        assert "2024" in first

    def test_swmm_date_formatting(self, tmp_path: Path):
        """Verify MM/DD/YYYY HH:MM format in SWMM timeseries output."""
        df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-03-15 14:30", tz="UTC")],
                "value": [5.5],
            }
        )
        out = tmp_path / "fmt.dat"
        write_swmm_timeseries(df, "TS1", out)

        text = out.read_text()
        assert "03/15/2024" in text
        assert "14:30" in text


# ── Integration test ─────────────────────────────────────────────────────────
class TestIntegration:
    def test_waterml_from_aquascope_data(self, tmp_path: Path):
        """Convert a list of WaterQualitySample objects → WaterML → back."""
        samples = [
            WaterQualitySample(
                source=DataSource.USGS,
                station_id="USGS-01",
                station_name="Colorado River",
                location=GeoLocation(latitude=36.0, longitude=-111.0),
                sample_datetime=datetime(2024, 6, 1, 12, tzinfo=timezone.utc),
                parameter="pH",
                value=7.4,
                unit="pH",
            ),
            WaterQualitySample(
                source=DataSource.USGS,
                station_id="USGS-01",
                station_name="Colorado River",
                location=GeoLocation(latitude=36.0, longitude=-111.0),
                sample_datetime=datetime(2024, 6, 2, 12, tzinfo=timezone.utc),
                parameter="pH",
                value=7.6,
                unit="pH",
            ),
        ]

        rows = []
        for s in samples:
            rows.append(
                {
                    "timestamp": s.sample_datetime,
                    "station_id": s.station_id,
                    "station_name": s.station_name,
                    "parameter": s.parameter,
                    "value": s.value,
                    "unit": s.unit,
                    "quality_code": "",
                    "latitude": s.location.latitude if s.location else None,
                    "longitude": s.location.longitude if s.location else None,
                }
            )
        df = pd.DataFrame(rows)

        ts_list = dataframe_to_waterml(df)
        assert len(ts_list) == 1

        out = tmp_path / "integration.xml"
        write_waterml(ts_list, out)
        result = read_waterml(out)

        assert len(result) == 1
        r = result[0]
        assert r.station_id == "USGS-01"
        assert r.parameter == "pH"
        assert len(r.values) == 2
        assert r.values[0] == 7.4
        assert r.latitude == 36.0
