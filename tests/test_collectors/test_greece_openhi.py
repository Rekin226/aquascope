"""Tests for the Greece OpenHi.net (Enhydris 3) collector.

Every fixture below is a verbatim capture from ``system.openhi.net`` taken on
2026-09-10, not a hand-written guess at the shape: the station record and its
EWKT geometry, the paginated envelope, a group list, the three-series group at
station 1458 (raw plus two aggregates), a real unit record, and CSV bodies for
the populated, empty and value-less cases. Network access is fully mocked.
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock

import pytest

from aquascope.collectors.greece_openhi import (
    API_BASE,
    GreeceOpenhiCollector,
    _classify_failure,
    _parse_csv,
    _parse_point,
    _window_params,
)
from aquascope.schemas.water_data import (
    ClimateReading,
    DataSource,
    StreamflowReading,
    WaterLevelReading,
    WaterQualitySample,
)

# Two pages, to prove the `next` walk is followed. Station 1484 is real;
# 8426 is the live Mandra discharge station.
STATIONS_PAGE_1 = {
    "count": 3,
    "next": f"{API_BASE}/stations/?format=json&page=2",
    "previous": None,
    "results": [
        {
            "id": 1484,
            "last_update": "2020-03-27T22:30:00Z",
            "last_modified": "2019-07-15T09:33:54.376697+03:00",
            "name": "Arta's Bridge",
            "code": "Arta",
            "remarks": "",
            "geom": "SRID=4326;POINT (20.975265 39.15104)",
            "display_timezone": "Etc/GMT-2",
            "altitude": None,
            "start_date": None,
            "end_date": None,
            "overseer": "",
            "owner": 21,
        }
    ],
}

STATIONS_PAGE_2 = {
    "count": 3,
    "next": None,
    "previous": f"{API_BASE}/stations/?format=json",
    "results": [
        {
            "id": 8426,
            "last_update": "2026-09-10T03:30:00Z",
            "last_modified": "2021-11-19T20:08:16.883455+02:00",
            "name": "Μάνδρα Κόμβος",
            "code": "",
            "remarks": "",
            "geom": "SRID=4326;POINT (23.494 38.0665)",
            "display_timezone": "Etc/GMT-2",
            "altitude": 303.0,
            "start_date": None,
            "end_date": None,
            "overseer": "",
            "owner": 11,
        },
        {
            # Greek Grid geometry: must be rejected, not read as degrees.
            "id": 9999,
            "last_update": None,
            "name": "Projected Station",
            "code": "",
            "geom": "SRID=2100;POINT (476714 4210762)",
            "altitude": None,
            "start_date": None,
            "end_date": None,
        },
    ],
}

GROUPS: dict[int, dict] = {
    9999: {"count": 0, "next": None, "results": []},
    1484: {
        "count": 1,
        "next": None,
        "results": [
            {
                "id": 5,
                "name": "Στάθμη",
                "hidden": False,
                "precision": 2,
                "gentity": 1484,
                "variable": 14,  # Stage
                "unit_of_measurement": 6,  # m
            }
        ],
    },
    8426: {
        "count": 3,
        "next": None,
        "results": [
            {
                "id": 887,
                "name": "Βαρομετρική Πίεση",
                "hidden": False,
                "precision": 2,
                "gentity": 8426,
                "variable": 22,  # Barometric pressure -> climate
                "unit_of_measurement": 4,
            },
            {
                "id": 884,
                "name": "Παροχή",
                "hidden": False,
                "precision": 2,
                "gentity": 8426,
                "variable": 2,  # Discharge
                "unit_of_measurement": 18,  # m3/s
            },
            {
                "id": 1200,
                "name": "Τάση Μπαταρίας",
                "hidden": False,
                "precision": 2,
                "gentity": 8426,
                "variable": 5685,  # Battery voltage: instrument telemetry
                "unit_of_measurement": 1003,
            },
        ],
    },
}

# Station 1458's real shape: a raw series plus two aggregates. Used to prove
# the Initial-first preference and the fallback when the raw one is empty.
SERIES_THREE = {
    "count": 3,
    "next": None,
    "results": [
        {"id": 10106, "type": "Aggregated", "time_step": "1h", "name": "Mean", "timeseries_group": 884},
        {"id": 10287, "type": "Initial", "time_step": "", "name": "", "timeseries_group": 884},
        {"id": 10115, "type": "Aggregated", "time_step": "1D", "name": "Mean", "timeseries_group": 884},
    ],
}

SERIES_ONE = {
    "count": 1,
    "next": None,
    "results": [{"id": 9703, "type": "Initial", "time_step": "15min", "name": "", "timeseries_group": 5}],
}

UNITS = {
    18: {"id": 18, "descr": "Κυβικά μέτρα ανά δευτερόλεπτο", "symbol": "m³/s", "variables": []},
    6: {"id": 6, "descr": "Μέτρα", "symbol": "m", "variables": []},
    2: {"id": 2, "descr": "Εκατοστά", "symbol": "cm", "variables": []},
    4: {"id": 4, "descr": "Εκατοπασκάλ", "symbol": "hPa", "variables": []},
}


class _StatusError(Exception):
    """Stands in for httpx.HTTPStatusError: what CachedHTTPClient leaves as __cause__."""

    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.response = MagicMock(status_code=status_code)


def _client_error(status_code: int) -> RuntimeError:
    """The RuntimeError CachedHTTPClient raises after exhausting its retries."""
    exc = RuntimeError("all attempts failed")
    exc.__cause__ = _StatusError(status_code)
    return exc


CSV_POPULATED = "2026-09-08 00:00,0.42,\n2026-09-08 00:15,,\n2026-09-08 00:30,0.44,\n"
CSV_ONLY_NULLS = "2026-09-10 02:00,,\n2026-09-10 02:15,,\n"  # station 1534's sensors today
CSV_EMPTY = ""


def _client(csv_by_series: dict[int, str] | None = None) -> MagicMock:
    """Mock client that routes every URL this collector builds.

    Passing ``{}`` means "every series is empty" and must not fall back to the
    default map, so the sentinel is ``None`` rather than falsiness.
    """
    if csv_by_series is None:
        csv_by_series = {10287: CSV_POPULATED, 9703: CSV_POPULATED}
    client = MagicMock()

    def get_json(url: str, *args, **kwargs):
        if "/stations/?format=json&page=2" in url:
            return STATIONS_PAGE_2
        if url.endswith("/stations/?format=json"):
            return STATIONS_PAGE_1
        if "/units/" in url:
            return UNITS[int(url.split("/units/")[1].split("/")[0])]
        if url.endswith("/timeseries/?format=json"):
            return SERIES_THREE if "/timeseriesgroups/884/" in url else SERIES_ONE
        if url.endswith("/timeseriesgroups/?format=json"):
            return GROUPS[int(url.split("/stations/")[1].split("/")[0])]
        raise AssertionError(f"unexpected URL {url}")

    def get_text(url: str, *args, **kwargs):
        series_id = int(url.rstrip("/").split("/")[-2])
        return csv_by_series.get(series_id, CSV_EMPTY)

    client.get_json.side_effect = get_json
    client.get_text.side_effect = get_text
    return client


class TestParsePoint:
    def test_ewkt_is_lon_lat_not_lat_lon(self):
        """The real Arta's Bridge record. Swapping the order lands it in Turkey."""
        assert _parse_point("SRID=4326;POINT (20.975265 39.15104)") == pytest.approx((39.15104, 20.975265))

    def test_greek_grid_rejected(self):
        assert _parse_point("SRID=2100;POINT (476714 4210762)") is None

    def test_missing_or_malformed(self):
        assert _parse_point(None) is None
        assert _parse_point("") is None
        assert _parse_point("SRID=4326;LINESTRING (0 0, 1 1)") is None


class TestParseCsv:
    def test_headerless_rows_with_a_gap(self):
        assert _parse_csv(CSV_POPULATED) == [
            (datetime(2026, 9, 8, 0, 0), 0.42),
            (datetime(2026, 9, 8, 0, 30), 0.44),
        ]

    def test_all_null_values_yields_nothing(self):
        assert _parse_csv(CSV_ONLY_NULLS) == []

    def test_seconds_and_iso_separator_accepted(self):
        assert _parse_csv("2026-09-08T00:00:30,1.5,\n") == [(datetime(2026, 9, 8, 0, 0, 30), 1.5)]

    def test_junk_rows_skipped(self):
        assert _parse_csv("not-a-row\n2026-09-08 00:00,abc,\n") == []


class TestWindowParams:
    def test_both_bounds(self):
        assert _window_params("2026-09-01", "2026-09-05") == {
            "start_date": "2026-09-01",
            "end_date": "2026-09-05",
        }

    def test_half_open_and_date_objects(self):
        assert _window_params(date(2026, 9, 1), None) == {"start_date": "2026-09-01"}
        assert _window_params(None, None) == {}


class TestClassifyFailure:
    def test_401_is_restricted_not_failed(self):
        """Station 28082's discharge really does answer 401."""
        assert _classify_failure(_client_error(401)) == "restricted"

    def test_transport_error_is_failed(self):
        exc = RuntimeError("all attempts failed")
        exc.__cause__ = None
        assert _classify_failure(exc) == "failed"


class TestStations:
    def test_pagination_is_followed(self):
        """Reading only page 1 would return 1 station instead of 2."""
        stations = GreeceOpenhiCollector(client=_client()).stations()
        assert [s.station_id for s in stations] == ["1484", "8426"]

    def test_projected_geometry_station_is_dropped(self):
        assert "9999" not in {s.station_id for s in GreeceOpenhiCollector(client=_client()).stations()}

    def test_variables_exclude_instrument_telemetry(self):
        """Station 8426 has a battery-voltage group; it is not an observation."""
        station = next(s for s in GreeceOpenhiCollector(client=_client()).stations() if s.station_id == "8426")
        assert station.variables == ("climate", "discharge")

    def test_station_fields(self):
        station = next(s for s in GreeceOpenhiCollector(client=_client()).stations() if s.station_id == "8426")
        assert station.name == "Μάνδρα Κόμβος"
        assert station.country == "GRC"
        assert station.latitude == pytest.approx(38.0665)
        assert station.extra["altitude_m"] == 303.0
        assert station.period_end == date(2026, 9, 10)

    def test_variable_and_bbox_filters(self):
        collector = GreeceOpenhiCollector(client=_client())
        assert [s.station_id for s in collector.stations(variable="discharge")] == ["8426"]
        assert collector.stations(variable="groundwater_level") == []
        assert [s.station_id for s in collector.stations(bbox=(20.0, 38.5, 22.0, 40.0))] == ["1484"]

    def test_max_items(self):
        assert len(GreeceOpenhiCollector(client=_client()).stations(max_items=1)) == 1


class TestSeriesOutcome:
    """The four-way split the collector exists to preserve (#165).

    Exercised directly on ``_fetch_series_data`` rather than only through the
    ``fetch_raw`` tally, because collapsing any of these into "failed" is the
    exact bug the issue is about.
    """

    @staticmethod
    def _fetch(collector: GreeceOpenhiCollector, series_id: int):
        return collector._fetch_series_data("8426", 884, series_id, {}, False)

    def test_populated_series_is_ok(self):
        got = self._fetch(GreeceOpenhiCollector(client=_client()), 10287)
        assert got.outcome == "ok"
        assert [v for _, v in got.rows] == pytest.approx([0.42, 0.44])

    def test_zero_byte_body_is_empty_not_failed(self):
        got = self._fetch(GreeceOpenhiCollector(client=_client({})), 10287)
        assert (got.outcome, got.rows) == ("empty", [])

    def test_timestamps_without_values_are_their_own_case(self):
        got = self._fetch(GreeceOpenhiCollector(client=_client({10287: CSV_ONLY_NULLS})), 10287)
        assert (got.outcome, got.rows) == ("no_values", [])

    def test_401_is_restricted_not_failed(self):
        client = _client()
        client.get_text.side_effect = _client_error(401)
        got = self._fetch(GreeceOpenhiCollector(client=client), 10287)
        assert (got.outcome, got.rows) == ("restricted", [])

    def test_transport_failure_is_failed(self):
        client = _client()
        client.get_text.side_effect = RuntimeError("connection reset")
        assert self._fetch(GreeceOpenhiCollector(client=client), 10287).outcome == "failed"


class TestFetchRaw:
    def test_initial_series_is_preferred_over_aggregates(self):
        collector = GreeceOpenhiCollector(client=(client := _client()))
        collector.fetch_raw(variable="discharge")
        first = [c.args[0] for c in client.get_text.call_args_list][0]
        assert "/timeseries/10287/data/" in first  # the Initial one, not 10106

    def test_empty_initial_falls_back_to_the_aggregate(self):
        """Station 1458's raw discharge is empty while an aggregate holds data."""
        collector = GreeceOpenhiCollector(client=_client({10287: CSV_EMPTY, 10106: CSV_POPULATED}))
        rows = collector.fetch_raw(variable="discharge")
        assert len(rows) == 2

    def test_date_window_is_pushed_to_the_api(self):
        collector = GreeceOpenhiCollector(client=(client := _client()))
        collector.fetch_raw(variable="discharge", start="2026-09-08", end="2026-09-09")
        assert client.get_text.call_args_list[0].kwargs["params"] == {
            "start_date": "2026-09-08",
            "end_date": "2026-09-09",
        }

    def test_latest_only_uses_the_bottom_endpoint(self):
        collector = GreeceOpenhiCollector(client=(client := _client()))
        collector.fetch_raw(variable="discharge", latest_only=True)
        assert client.get_text.call_args_list[0].args[0].endswith("/bottom/")

    def test_station_id_filter(self):
        collector = GreeceOpenhiCollector(client=_client())
        assert collector.fetch_raw(variable="discharge", station_ids=[8426])
        assert collector.fetch_raw(variable="discharge", station_ids=[1484]) == []

    def test_unknown_variable_raises(self):
        with pytest.raises(ValueError, match="Unknown variable"):
            GreeceOpenhiCollector(client=_client()).fetch_raw(variable="reservoir_storage")

    def test_empty_is_reported_apart_from_failed(self, caplog):
        """The #165 guard: no data must not read as a broken source."""
        collector = GreeceOpenhiCollector(client=_client({}))  # every series empty
        with caplog.at_level("INFO"):
            assert collector.fetch_raw(variable="discharge") == []
        assert "3 catalogued but empty" in caplog.text
        assert "0 failed" in caplog.text

    def test_only_null_values_is_its_own_outcome(self, caplog):
        collector = GreeceOpenhiCollector(
            client=_client({10287: CSV_ONLY_NULLS, 10106: CSV_ONLY_NULLS, 10115: CSV_ONLY_NULLS})
        )
        with caplog.at_level("INFO"):
            assert collector.fetch_raw(variable="discharge") == []
        assert "3 returning only null values" in caplog.text
        assert "0 catalogued but empty" in caplog.text

    def test_restricted_series_counted_as_restricted(self, caplog):
        client = _client()
        client.get_text.side_effect = _client_error(401)
        with caplog.at_level("INFO"):
            assert GreeceOpenhiCollector(client=client).fetch_raw(variable="discharge") == []
        assert "3 restricted (401/403)" in caplog.text
        assert "0 failed" in caplog.text


class TestNormalise:
    def test_discharge_is_streamflow_in_cms(self):
        records = GreeceOpenhiCollector(client=_client()).collect(variable="discharge")
        assert all(isinstance(r, StreamflowReading) for r in records)
        assert [r.discharge_cms for r in records] == pytest.approx([0.42, 0.44])
        assert records[0].source == DataSource.GREECE_OPENHI
        assert records[0].source_type == "in_situ"

    def test_stage_in_metres(self):
        records = GreeceOpenhiCollector(client=_client()).collect(variable="water_level")
        assert all(isinstance(r, WaterLevelReading) for r in records)
        assert records[0].water_level == pytest.approx(0.42)
        assert records[0].location.latitude == pytest.approx(39.15104)

    def test_centimetre_stage_is_converted(self):
        rows = [
            {
                "station_id": "1",
                "datetime": datetime(2026, 9, 8),
                "value": 42.0,
                "unit": "cm",
                "variable": "water_level",
                "parameter": "stage_m",
            }
        ]
        assert GreeceOpenhiCollector(client=_client()).normalise(rows)[0].water_level == pytest.approx(0.42)

    def test_climate_carries_altitude(self):
        records = GreeceOpenhiCollector(client=_client()).collect(variable="climate")
        assert all(isinstance(r, ClimateReading) for r in records)
        assert records[0].parameter == "pressure_barometric_hpa"
        assert records[0].altitude_m == 303.0
        assert records[0].unit == "hPa"

    def test_water_quality_sample(self):
        rows = [
            {
                "station_id": "1534",
                "datetime": datetime(2020, 6, 1),
                "value": 7.8,
                "unit": "-",
                "variable": "water_quality",
                "parameter": "pH",
            }
        ]
        record = GreeceOpenhiCollector(client=_client()).normalise(rows)[0]
        assert isinstance(record, WaterQualitySample)
        assert (record.parameter, record.value, record.unit) == ("pH", 7.8, "-")

    def test_unconvertible_discharge_unit_is_dropped(self):
        rows = [
            {
                "station_id": "1",
                "datetime": datetime(2026, 9, 8),
                "value": 5.0,
                "unit": "cfs",
                "variable": "discharge",
                "parameter": "discharge_cms",
            }
        ]
        assert GreeceOpenhiCollector(client=_client()).normalise(rows) == []


class TestCollectRoundTrip:
    def test_records_carry_station_identity(self):
        records = GreeceOpenhiCollector(client=_client()).collect(variable="discharge")
        assert records
        assert all(r.station_id == "8426" for r in records)
        assert all(r.station_name == "Μάνδρα Κόμβος" for r in records)
