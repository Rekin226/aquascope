"""Tests for the CAMELS-BR (Brazil) streamflow collector."""

from __future__ import annotations

import io
import zipfile

from aquascope.collectors.camels_br import CAMELS_BR_ATTRS_URL, CAMELS_BR_FLOW_URL, CAMELSBRCollector
from aquascope.schemas.water_data import DataSource

SAMPLE_RAW = [
    {
        "station_id": "10500000",
        "date": "1980-09-21",
        "discharge": 12.5,
        "gauge_name": "estirao_do_repouso",
        "gauge_lat": -4.3408,
        "gauge_lon": -70.9056,
        "area_km2": 61581.51,
    },
    {
        "station_id": "11400000",
        "date": "1980-09-22",
        "discharge": 43.4,
        "gauge_name": "test_station",
        "gauge_lat": -5.1389,
        "gauge_lon": -72.8136,
        "area_km2": 16558.04,
    },
]

SAMPLE_RAW_MALFORMED = [
    # Missing discharge key
    {
        "station_id": "BAD01",
        "date": "1980-09-21",
        "gauge_name": "Bad Station",
    },
    # Non-numeric discharge
    {
        "station_id": "BAD02",
        "date": "1980-09-21",
        "discharge": "not_a_number",
    },
]


def _make_fake_attrs_zip() -> bytes:
    """Build an in-memory ZIP containing fake CAMELS-BR locations and topography."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "01_CAMELS_BR_attributes/camels_br_location.txt",
            "gauge_id gauge_name gauge_lat gauge_lon area_ana area_gsim area_gsim_quality\n"
            "10500000 estirao_do_repouso -4.3408 -70.9056 61400 61581.51 high\n"
            "11400000 test_station -5.1389 -72.8136 16500 16558.04 high\n",
        )
        zf.writestr(
            "01_CAMELS_BR_attributes/camels_br_topography.txt",
            "gauge_id elev_gauge elev_mean slope_mean area\n"
            "10500000 74.00000 154.40630 7.96140 61581.51000\n"
            "11400000 61.00000 1146.48310 89.92080 16558.04000\n",
        )
    buf.seek(0)
    return buf.read()


def _make_fake_flow_zip() -> bytes:
    """Build an in-memory ZIP containing fake streamflow files per gauge."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "03_CAMELS_BR_streamflow_selected_catchments/10500000_streamflow.txt",
            "year month day streamflow_m3s streamflow_mm qual_control_by_ana qual_flag\n"
            "1980 9 21 12.5 nan 1.0 1.0\n"
            "1980 9 22 nan nan nan nan\n",
        )
        zf.writestr(
            "03_CAMELS_BR_streamflow_selected_catchments/11400000_streamflow.txt",
            "year month day streamflow_m3s streamflow_mm qual_control_by_ana qual_flag\n"
            "1980 9 21 nan nan nan nan\n"
            "1980 9 22 43.4 nan 1.0 1.0\n",
        )
    buf.seek(0)
    return buf.read()


class TestCAMELSBRInit:
    def setup_method(self):
        self.collector = CAMELSBRCollector()

    def test_collector_name(self):
        assert self.collector.name == "camels_br"


class TestCAMELSBRNormalise:
    def setup_method(self):
        self.collector = CAMELSBRCollector()

    def test_normalise_produces_correct_count(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert len(records) == 2

    def test_normalise_sets_correct_source(self):
        records = self.collector.normalise(SAMPLE_RAW)
        for r in records:
            assert r.source == DataSource.CAMELS_BR

    def test_normalise_tags_in_situ(self):
        records = self.collector.normalise(SAMPLE_RAW)
        for r in records:
            assert r.source_type == "in_situ"

    def test_normalise_parses_location(self):
        records = self.collector.normalise(SAMPLE_RAW)
        rec = records[0]
        assert rec.location is not None
        assert abs(rec.location.latitude - (-4.3408)) < 0.001
        assert abs(rec.location.longitude - (-70.9056)) < 0.001

    def test_normalise_discharge_value(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[0].discharge_cms == 12.5
        assert records[0].unit == "m3/s"

    def test_normalise_preserves_station_id(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[0].station_id == "10500000"

    def test_normalise_carries_catchment_area(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[0].catchment_area_km2 == 61581.51
        assert records[1].catchment_area_km2 == 16558.04

    def test_normalise_carries_station_name(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[0].station_name == "estirao_do_repouso"

    def test_normalise_skips_malformed_rows(self):
        records = self.collector.normalise(SAMPLE_RAW_MALFORMED)
        assert records == []

    def test_normalise_empty_input(self):
        records = self.collector.normalise([])
        assert records == []

    def test_normalise_nan_attrs_record_survives(self):
        nan = float("nan")
        raw = [
            {
                "station_id": "77777777",
                "date": "1980-09-21",
                "discharge": 5.0,
                "gauge_name": nan,
                "gauge_lat": nan,
                "gauge_lon": nan,
                "area_km2": nan,
            }
        ]
        records = self.collector.normalise(raw)
        assert len(records) == 1
        rec = records[0]
        assert rec.discharge_cms == 5.0
        assert rec.location is None
        assert rec.station_name is None
        assert rec.catchment_area_km2 is None


class TestCAMELSBRFetchRaw:
    def setup_method(self, method=None):
        self.attrs_zip = _make_fake_attrs_zip()
        self.flow_zip = _make_fake_flow_zip()

    def _patch_httpx_and_cache(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)

        class FakeResponse:
            def __init__(self, data):
                self.data = data

            def raise_for_status(self):
                pass

            def iter_bytes(self, chunk_size=65_536):
                yield self.data

        class FakeStream:
            def __init__(self, data):
                self.resp = FakeResponse(data)

            def __enter__(self):
                return self.resp

            def __exit__(self, *a):
                return False

        def mock_stream(method, url, **kwargs):
            if CAMELS_BR_ATTRS_URL in url:
                return FakeStream(self.attrs_zip)
            elif CAMELS_BR_FLOW_URL in url:
                return FakeStream(self.flow_zip)
            raise ValueError(f"Unexpected stream URL: {url}")

        monkeypatch.setattr("httpx.stream", mock_stream)

    def test_fetch_raw_returns_all_stations_by_default(self, monkeypatch, tmp_path):
        self._patch_httpx_and_cache(monkeypatch, tmp_path)
        collector = CAMELSBRCollector()
        raw = collector.fetch_raw()
        station_ids = {row["station_id"] for row in raw}
        assert station_ids == {"10500000", "11400000"}

    def test_fetch_raw_drops_na_values(self, monkeypatch, tmp_path):
        self._patch_httpx_and_cache(monkeypatch, tmp_path)
        collector = CAMELSBRCollector()
        raw = collector.fetch_raw()
        assert len(raw) == 2

    def test_fetch_raw_filters_by_station_ids(self, monkeypatch, tmp_path):
        self._patch_httpx_and_cache(monkeypatch, tmp_path)
        collector = CAMELSBRCollector()
        raw = collector.fetch_raw(station_ids=["10500000"])
        assert all(row["station_id"] == "10500000" for row in raw)
        assert len(raw) == 1

    def test_fetch_raw_filters_by_date_range(self, monkeypatch, tmp_path):
        self._patch_httpx_and_cache(monkeypatch, tmp_path)
        collector = CAMELSBRCollector()
        raw = collector.fetch_raw(start="1980-09-22", end="1980-09-22")
        assert len(raw) == 1
        assert raw[0]["station_id"] == "11400000"

    def test_fetch_raw_joins_catchment_attributes(self, monkeypatch, tmp_path):
        self._patch_httpx_and_cache(monkeypatch, tmp_path)
        collector = CAMELSBRCollector()
        raw = collector.fetch_raw(station_ids=["10500000"])
        row = raw[0]
        assert row["gauge_name"] == "estirao_do_repouso"
        assert row["area_km2"] == 61581.51

    def test_fetch_raw_warns_on_unknown_station_id(self, monkeypatch, tmp_path, caplog):
        self._patch_httpx_and_cache(monkeypatch, tmp_path)
        collector = CAMELSBRCollector()
        collector.fetch_raw(station_ids=["99999999"])
        assert any("99999999" in rec.message for rec in caplog.records)
