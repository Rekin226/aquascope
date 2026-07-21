"""Tests for the CAMELS-CL (CR2, Chile) streamflow collector."""

from __future__ import annotations

import io
import zipfile

from aquascope.collectors.camels_cl import CAMELSCLCollector
from aquascope.schemas.water_data import DataSource

SAMPLE_RAW = [
    {
        "station_id": "1001001",
        "date": "1990-01-01",
        "discharge": 12.5,
        "gauge_name": "Rio Test En Nacimiento", # fake test river
        "gauge_lat": -18.5,
        "gauge_lon": -69.5,
        "area_km2": 250.0,
    },
    {
        "station_id": "12825002",
        "date": "2011-06-13",
        "discharge": 43.4,
        "gauge_name": "Rio Azopardo En Desembocadura", # Azopardo River at the Mouth
        "gauge_lat": -54.5028,
        "gauge_lon": -68.8244,
        "area_km2": 3524.5,
    },
]

SAMPLE_RAW_MALFORMED = [
    # Missing required "discharge" key
    {
        "station_id": "BAD01",
        "date": "1990-01-01",
        "gauge_name": "Bad Station",
    },
    # Non-numeric discharge value
    {
        "station_id": "BAD02",
        "date": "1990-01-01",
        "discharge": "not_a_number",
    },
]


def _make_fake_zip() -> bytes:
    """Build an in-memory ZIP mirroring CAMELS-CL's real archive layout:
    a wide-format q_m3s_day.csv (date + one column per gauge, "NA" for
    missing values) and a catchment_attributes.csv with gauge metadata.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "CAMELS_CL_v202201/q_m3s_day.csv",
            '"date","year","month","day","1001001","12825002"\n'
            "1990-01-01,1990,1,1,12.5,NA\n"
            "1990-01-02,1990,1,2,NA,NA\n"
            "2011-06-13,2011,6,13,NA,43.4\n",
        )
        zf.writestr(
            "CAMELS_CL_v202201/catchment_attributes.csv",
            '"gauge_id","gauge_name","gauge_lat","gauge_lon","area_km2"\n'
            '"1001001","Rio Test En Nacimiento","-18.5","-69.5","250.0"\n'
            '"12825002","Rio Azopardo En Desembocadura","-54.5028","-68.8244","3524.5"\n',
        )
    buf.seek(0)
    return buf.read()


class TestCAMELSCLInit:
    def setup_method(self):
        self.collector = CAMELSCLCollector()

    def test_collector_name(self):
        assert self.collector.name == "camels_cl"


class TestCAMELSCLNormalise:
    def setup_method(self):
        self.collector = CAMELSCLCollector()

    def test_normalise_produces_correct_count(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert len(records) == 2

    def test_normalise_sets_correct_source(self):
        records = self.collector.normalise(SAMPLE_RAW)
        for r in records:
            assert r.source == DataSource.CAMELS_CL

    def test_normalise_tags_in_situ(self):
        records = self.collector.normalise(SAMPLE_RAW)
        for r in records:
            assert r.source_type == "in_situ"

    def test_normalise_parses_location(self):
        records = self.collector.normalise(SAMPLE_RAW)
        rec = records[0]
        assert rec.location is not None
        assert abs(rec.location.latitude - (-18.5)) < 0.001
        assert abs(rec.location.longitude - (-69.5)) < 0.001

    def test_normalise_discharge_value(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[0].discharge_cms == 12.5
        assert records[0].unit == "m3/s"

    def test_normalise_preserves_station_id(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[0].station_id == "1001001"

    def test_normalise_carries_catchment_area(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[0].catchment_area_km2 == 250.0
        assert records[1].catchment_area_km2 == 3524.5

    def test_normalise_carries_station_name(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[0].station_name == "Rio Test En Nacimiento"

    def test_normalise_skips_malformed_rows(self):
        records = self.collector.normalise(SAMPLE_RAW_MALFORMED)
        assert records == []

    def test_normalise_empty_input(self):
        records = self.collector.normalise([])
        assert records == []

    def test_normalise_nan_attrs_record_survives(self):
        # A gauge missing from catchment_attributes.csv comes out of the
        # left merge with NaN in every attrs column; the discharge record
        # itself must survive with the optional fields set to None.
        nan = float("nan")
        raw = [
            {
                "station_id": "7777777",
                "date": "2000-01-01",
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

    def test_normalise_missing_location_is_none(self):
        raw = [
            {
                "station_id": "1001001",
                "date": "1990-01-01",
                "discharge": 12.5,
                "gauge_lat": None,
                "gauge_lon": None,
            }
        ]
        records = self.collector.normalise(raw)
        assert records[0].location is None


class TestCAMELSCLFetchRaw:
    def setup_method(self, method=None):
        self.zip_bytes = _make_fake_zip()

    def _patch_httpx_and_cache(self, monkeypatch, tmp_path):
        """Route the (mocked) download straight into tmp_path's cache dir,
        and short-circuit httpx.stream so no real network call happens."""
        monkeypatch.chdir(tmp_path)

        class FakeResponse:
            def raise_for_status(self):
                pass

            def iter_bytes(self, chunk_size=65_536):
                yield self.zip_bytes

        class FakeStream:
            def __init__(self, zip_bytes):
                self.resp = FakeResponse()
                self.resp.zip_bytes = zip_bytes

            def __enter__(self):
                return self.resp

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(
            "httpx.stream", lambda *a, **kw: FakeStream(self.zip_bytes)
        )

    def test_fetch_raw_returns_all_stations_by_default(self, monkeypatch, tmp_path):
        self._patch_httpx_and_cache(monkeypatch, tmp_path)
        collector = CAMELSCLCollector()
        raw = collector.fetch_raw()
        station_ids = {row["station_id"] for row in raw}
        assert station_ids == {"1001001", "12825002"}

    def test_fetch_raw_drops_na_values(self, monkeypatch, tmp_path):
        self._patch_httpx_and_cache(monkeypatch, tmp_path)
        collector = CAMELSCLCollector()
        raw = collector.fetch_raw()
        # 3 dates x 2 stations = 6 cells, but 4 are "NA" -> only 2 real rows
        assert len(raw) == 2

    def test_fetch_raw_filters_by_station_ids(self, monkeypatch, tmp_path):
        self._patch_httpx_and_cache(monkeypatch, tmp_path)
        collector = CAMELSCLCollector()
        raw = collector.fetch_raw(station_ids=["1001001"])
        assert all(row["station_id"] == "1001001" for row in raw)
        assert len(raw) == 1

    def test_fetch_raw_filters_by_date_range(self, monkeypatch, tmp_path):
        self._patch_httpx_and_cache(monkeypatch, tmp_path)
        collector = CAMELSCLCollector()
        raw = collector.fetch_raw(start="2000-01-01", end="2020-01-01")
        assert len(raw) == 1
        assert raw[0]["station_id"] == "12825002"

    def test_fetch_raw_joins_catchment_attributes(self, monkeypatch, tmp_path):
        self._patch_httpx_and_cache(monkeypatch, tmp_path)
        collector = CAMELSCLCollector()
        raw = collector.fetch_raw(station_ids=["1001001"])
        row = raw[0]
        assert row["gauge_name"] == "Rio Test En Nacimiento"
        assert row["area_km2"] == 250.0

    def test_fetch_raw_warns_on_unknown_station_id(self, monkeypatch, tmp_path, caplog):
        self._patch_httpx_and_cache(monkeypatch, tmp_path)
        collector = CAMELSCLCollector()
        collector.fetch_raw(station_ids=["9999999"])
        assert any("9999999" in rec.message for rec in caplog.records)

    def test_fetch_raw_uses_cached_archive_on_second_call(self, monkeypatch, tmp_path):
        """After the first call populates data/cache/, a second call should
        not invoke httpx.stream again."""
        self._patch_httpx_and_cache(monkeypatch, tmp_path)
        collector = CAMELSCLCollector()
        collector.fetch_raw()

        calls = {"count": 0}
        original = __import__("httpx").stream

        def counting_stream(*a, **kw):
            calls["count"] += 1
            return original(*a, **kw)

        monkeypatch.setattr("httpx.stream", counting_stream)
        collector.fetch_raw()
        assert calls["count"] == 0
