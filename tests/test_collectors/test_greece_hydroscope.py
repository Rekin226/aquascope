"""Tests for the Greece Hydroscope (Enhydris) collector.

Catalog and download calls go through a mocked client; the ``.hts`` parser,
the EWKT point parser and ``normalise()`` are exercised directly with fixture
payloads shaped like real hydroscope.gr responses - no network calls.
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock

import pytest

from aquascope.collectors.greece_hydroscope import (
    GreeceHydroscopeCollector,
    _date_suffix,
    _parse_hts,
    _parse_point,
    _pick_series,
)
from aquascope.schemas.water_data import (
    ClimateReading,
    DataSource,
    StreamflowReading,
    WaterLevelReading,
)

# Two stations on the Ministry of Environment instance (db=1): one gauged, one
# without coordinates so the catalog has to drop it.
FAKE_STATIONS = [
    {
        "id": 200045,
        "name": "ΓΕΦ. ΚΟΜΜΑ (ΣΠΕΡΧΕΙΟΣ)",
        "point": "SRID=4326;POINT (22.44051968326435 38.84953975650789)",
        "srid": 2100,
    },
    {"id": 200046, "name": "ΧΩΡΙΣ ΣΥΝΤΕΤΑΓΜΕΝΕΣ", "point": None, "srid": None},
]

FAKE_TIMESERIES = [
    # Discharge with data.
    {
        "id": 2225,
        "gentity": 200045,
        "variable": 101,
        "unit_of_measurement": 9,
        "start_date_utc": "2004-04-01T00:00:00+03:00",
        "end_date_utc": "2010-05-01T00:00:00+03:00",
    },
    # Daily stage with data, same station.
    {
        "id": 1319,
        "gentity": 200045,
        "variable": 88,
        "unit_of_measurement": 8,
        "start_date_utc": "1973-10-14T08:00:00+02:00",
        "end_date_utc": "2013-12-31T10:00:00+02:00",
    },
    # Catalogued but empty (Count=0 upstream): no start date, must be skipped
    # without ever issuing a download.
    {
        "id": 1367,
        "gentity": 200045,
        "variable": 101,
        "unit_of_measurement": 9,
        "start_date_utc": None,
        "end_date_utc": None,
    },
    # A variable outside the mapped set (ΑΝΕΜΟΣ / wind direction).
    {
        "id": 914,
        "gentity": 200045,
        "variable": 2,
        "unit_of_measurement": 1,
        "start_date_utc": "1967-01-01T08:00:00+02:00",
        "end_date_utc": "1997-03-31T09:00:00+03:00",
    },
    # Belongs to the station that has no coordinates.
    {
        "id": 1400,
        "gentity": 200046,
        "variable": 8,
        "unit_of_measurement": 3,
        "start_date_utc": "1960-01-01T08:00:00+02:00",
        "end_date_utc": "2011-04-30T08:00:00+03:00",
    },
]

# The unit here (l/s) contradicts the catalog's m3/s on purpose: the file wins.
HTS_DISCHARGE = (
    "Version=2\n"
    "Unit=l/s\n"
    "Count=3\n"
    "Title=\n"
    "Comment=ΑΓ. ΓΕΩΡΓΙΟΣ\n"
    "Timezone=EET (UTC+0200)\n"
    "Time_step=0,0\n"
    "Variable=ΠΑΡΟΧΗ\n"
    "\n"
    "2004-04-01 00:00,7508.500000,\n"
    "2004-05-01 00:00,,\n"  # gap in the record - must be skipped
    "2004-06-01 00:00,3860.000000,\n"
)

HTS_STAGE = (
    "Version=2\n"
    "Unit=m\n"
    "Count=2\n"
    "Comment=ΓΕΦ. ΜΙΚΡ. ΔΕΡΕΙΟΥ\n"
    "Timezone=EET (UTC+0200)\n"
    "Time_step=1440,0\n"
    "Variable=ΣΤΑΘΜΗ\n"
    "\n"
    "1973-10-14 08:00,0.010000,\n"
    "1973-10-15 08:00,0.380000,\n"
)


def _client(hts: str = HTS_DISCHARGE) -> MagicMock:
    """A mock client serving the fixture catalog for db=1 and nothing elsewhere."""
    client = MagicMock()

    def get_json(url: str, *args, **kwargs):
        if not url.startswith("http://kyy.hydroscope.gr"):
            raise RuntimeError(f"instance unavailable: {url}")
        return FAKE_STATIONS if url.endswith("/api/Station/") else FAKE_TIMESERIES

    client.get_json.side_effect = get_json
    client.get_text.return_value = hts
    return client


class TestParsePoint:
    def test_wgs84_point_yields_lat_lon(self):
        assert _parse_point("SRID=4326;POINT (22.4 38.8)") == pytest.approx((38.8, 22.4))

    def test_greek_grid_is_rejected_not_misread(self):
        """SRID 2100 coordinates are metres; reading them as degrees would land in the Atlantic."""
        assert _parse_point("SRID=2100;POINT (312345.0 4210762.0)") is None

    def test_missing_or_malformed_point(self):
        assert _parse_point(None) is None
        assert _parse_point("") is None
        assert _parse_point("SRID=4326;POLYGON ((0 0, 1 1, 2 2, 0 0))") is None

    def test_out_of_range_coordinates_rejected(self):
        assert _parse_point("SRID=4326;POINT (476714 4210762)") is None


class TestParseHts:
    def test_headers_and_values(self):
        headers, values = _parse_hts(HTS_DISCHARGE)
        assert headers["Unit"] == "l/s"
        assert headers["Variable"] == "ΠΑΡΟΧΗ"
        assert headers["Title"] == ""
        assert values == [
            (datetime(2004, 4, 1, 0, 0), 7508.5),
            (datetime(2004, 6, 1, 0, 0), 3860.0),
        ]

    def test_empty_value_rows_are_dropped(self):
        _, values = _parse_hts(HTS_DISCHARGE)
        assert all(v is not None for _, v in values)
        assert len(values) == 2  # three rows upstream, one is a gap

    def test_payload_without_body(self):
        headers, values = _parse_hts("Version=2\nUnit=m\nCount=0\n\n")
        assert headers["Count"] == "0"
        assert values == []

    def test_empty_payload(self):
        assert _parse_hts("") == ({}, [])


class TestDateSuffix:
    def test_two_sided_range_becomes_a_path(self):
        assert _date_suffix("2010-01-01", "2010-12-31") == "2010-01-01/2010-12-31/"

    def test_date_objects_accepted(self):
        assert _date_suffix(date(2010, 1, 1), date(2010, 12, 31)) == "2010-01-01/2010-12-31/"

    def test_half_open_range_is_filtered_client_side(self):
        assert _date_suffix("2010-01-01", None) == ""
        assert _date_suffix(None, "2010-12-31") == ""
        assert _date_suffix(None, None) == ""


class TestStations:
    def test_catalog_skips_stations_without_coordinates(self):
        stations = GreeceHydroscopeCollector(client=_client()).stations()
        assert [s.station_id for s in stations] == ["100200045"]

    def test_composite_id_matches_national_catalog(self):
        station = GreeceHydroscopeCollector(client=_client()).stations()[0]
        assert station.station_id == "100200045"  # db 1 * 100_000_000 + 200045
        assert station.url == "http://main.hydroscope.gr/stations/d/100200045/"
        assert station.country == "GRC"

    def test_variables_and_period_span_every_kept_series(self):
        station = GreeceHydroscopeCollector(client=_client()).stations()[0]
        assert station.variables == ("discharge", "water_level")  # wind is not mapped
        assert station.period_start == date(1973, 10, 14)
        assert station.period_end == date(2013, 12, 31)

    def test_variable_filter(self):
        collector = GreeceHydroscopeCollector(client=_client())
        assert len(collector.stations(variable="discharge")) == 1
        assert collector.stations(variable="groundwater_level") == []

    def test_bbox_filter(self):
        collector = GreeceHydroscopeCollector(client=_client())
        assert len(collector.stations(bbox=(20.0, 34.0, 27.0, 42.0))) == 1
        assert collector.stations(bbox=(0.0, 0.0, 1.0, 1.0)) == []

    def test_max_items(self):
        assert len(GreeceHydroscopeCollector(client=_client()).stations(max_items=1)) == 1

    def test_unreachable_instance_does_not_empty_the_catalog(self):
        """Two of the three instances raise here; the catalog still returns db=1."""
        assert len(GreeceHydroscopeCollector(client=_client()).stations()) == 1


class TestFetchRaw:
    def test_empty_series_is_never_downloaded(self):
        collector = GreeceHydroscopeCollector(client=(client := _client()))
        collector.fetch_raw(variable="discharge")
        urls = [call.args[0] for call in client.get_text.call_args_list]
        assert any("/timeseries/d/2225/download/" in u for u in urls)
        assert not any("/timeseries/d/1367/" in u for u in urls)

    def test_date_window_is_pushed_to_the_server(self):
        collector = GreeceHydroscopeCollector(client=(client := _client()))
        collector.fetch_raw(variable="discharge", start="2004-01-01", end="2004-12-31")
        urls = [call.args[0] for call in client.get_text.call_args_list]
        assert urls == ["http://kyy.hydroscope.gr/timeseries/d/2225/download/2004-01-01/2004-12-31/"]

    def test_half_open_window_filters_client_side(self):
        collector = GreeceHydroscopeCollector(client=_client())
        rows = collector.fetch_raw(variable="discharge", start="2004-05-01")
        assert [r["datetime"] for r in rows] == [datetime(2004, 6, 1)]

    def test_station_id_filter(self):
        collector = GreeceHydroscopeCollector(client=_client())
        assert collector.fetch_raw(variable="discharge", station_ids=["100200045"])
        assert collector.fetch_raw(variable="discharge", station_ids=["999999999"]) == []

    def test_unknown_variable_raises(self):
        with pytest.raises(ValueError, match="Unknown variable"):
            GreeceHydroscopeCollector(client=_client()).fetch_raw(variable="soil_moisture")

    def test_download_failure_skips_the_series(self):
        client = _client()
        client.get_text.side_effect = RuntimeError("502 Bad Gateway")
        assert GreeceHydroscopeCollector(client=client).fetch_raw(variable="discharge") == []


class TestNormalise:
    def test_discharge_converted_from_the_file_unit(self):
        """The catalog says m3/s and the file says l/s; the file wins."""
        records = GreeceHydroscopeCollector(client=_client()).collect(variable="discharge")
        assert all(isinstance(r, StreamflowReading) for r in records)
        assert [r.discharge_cms for r in records] == pytest.approx([7.5085, 3.86])
        assert records[0].source == DataSource.GREECE_HYDROSCOPE
        assert records[0].source_type == "in_situ"
        assert records[0].unit == "m3/s"

    def test_water_level_in_metres(self):
        records = GreeceHydroscopeCollector(client=_client(HTS_STAGE)).collect(variable="water_level")
        assert all(isinstance(r, WaterLevelReading) for r in records)
        assert [r.water_level for r in records] == pytest.approx([0.01, 0.38])
        assert records[0].location.latitude == pytest.approx(38.84953975650789)

    def test_precipitation_becomes_a_climate_reading(self):
        rain = HTS_STAGE.replace("Unit=m\n", "Unit=mm\n").replace("Variable=ΣΤΑΘΜΗ", "Variable=ΒΡΟΧΟΠΤΩΣΗ")
        collector = GreeceHydroscopeCollector(client=_client(rain))
        # The only precipitation series belongs to the station with no point,
        # so ask for it by feeding the parser directly.
        rows = [
            {
                "station_id": "100200046",
                "station_name": "ΧΩΡΙΣ ΣΥΝΤΕΤΑΓΜΕΝΕΣ",
                "latitude": None,
                "longitude": None,
                "datetime": datetime(1960, 1, 1, 8, 0),
                "value": 12.4,
                "unit": "mm",
                "variable": "precipitation",
                "agency": "Ministry of Environment and Energy (ΥΠΕΝ)",
            }
        ]
        records = collector.normalise(rows)
        assert isinstance(records[0], ClimateReading)
        assert records[0].parameter == "rainfall_mm"
        assert records[0].value == 12.4
        assert records[0].location is None

    def test_unconvertible_unit_is_dropped_not_assumed(self):
        """A discharge series declaring an unexpected unit must not be emitted as-is."""
        odd = HTS_DISCHARGE.replace("Unit=l/s", "Unit=cfs")
        assert GreeceHydroscopeCollector(client=_client(odd)).collect(variable="discharge") == []

    def test_litres_per_second_alias(self):
        rows = [
            {
                "station_id": "1",
                "datetime": datetime(2004, 4, 1),
                "value": 2000.0,
                "unit": "lt/s",
                "variable": "discharge",
            }
        ]
        assert GreeceHydroscopeCollector(client=_client()).normalise(rows)[0].discharge_cms == pytest.approx(2.0)


class TestCollectRoundTrip:
    def test_records_carry_station_identity(self):
        records = GreeceHydroscopeCollector(client=_client()).collect(variable="discharge")
        assert records
        assert all(r.station_id == "100200045" for r in records)
        assert all(r.station_name == "ΓΕΦ. ΚΟΜΜΑ (ΣΠΕΡΧΕΙΟΣ)" for r in records)
        assert all(r.source == DataSource.GREECE_HYDROSCOPE for r in records)


class TestPickSeries:
    """One series per station and variable.

    Station 200082 really does hold three water-level series: two ΣΤΑΘΜΗ
    covering 1950-1983 and 1950-1982, plus a ΣΤΑΘΜΗ (ΠΛΗΜΜΥΡΑ). Concatenating
    them produced 515,650 rows with 11,219 duplicate timestamps and silently
    interleaved flood stage into the stage record.
    """

    @staticmethod
    def _series(series_id, variable, start, end):
        return {
            "id": series_id,
            "variable": variable,
            "start_date_utc": f"{start}T00:00:00+02:00",
            "end_date_utc": f"{end}T00:00:00+02:00",
        }

    def test_longest_record_wins_among_equals(self):
        short = self._series(23, 88, "1950-10-15", "1982-05-30")
        long = self._series(1233, 88, "1950-10-15", "1983-07-31")
        assert [s["id"] for s in _pick_series([short, long], "water_level")] == [1233]

    def test_ordinary_stage_beats_flood_stage_even_when_shorter(self):
        stage = self._series(1233, 88, "1970-01-01", "1975-01-01")
        flood = self._series(1234, 103, "1953-11-01", "1983-02-13")
        assert [s["id"] for s in _pick_series([stage, flood], "water_level")] == [1233]

    def test_flood_stage_is_used_when_it_is_all_there_is(self):
        flood = self._series(1234, 103, "1953-11-01", "1983-02-13")
        assert [s["id"] for s in _pick_series([flood], "water_level")] == [1234]

    def test_continuous_discharge_beats_individual_gaugings(self):
        gaugings = self._series(1367, 101, "1960-01-01", "2010-01-01")  # ΥΔΡΟΜΕΤΡΗΣΗ
        continuous = self._series(2225, 85, "2004-04-01", "2010-05-01")  # ΠΑΡΟΧΗ
        assert [s["id"] for s in _pick_series([gaugings, continuous], "discharge")] == [2225]

    def test_single_and_empty_candidates_pass_through(self):
        one = self._series(1, 88, "1950-01-01", "1960-01-01")
        assert _pick_series([one], "water_level") == [one]
        assert _pick_series([], "water_level") == []

    def test_fetch_raw_emits_one_series_per_station(self):
        """The whole point: no timestamp appears twice."""
        collector = GreeceHydroscopeCollector(client=_client(HTS_STAGE))
        rows = collector.fetch_raw(variable="water_level")
        stamps = [r["datetime"] for r in rows]
        assert len(stamps) == len(set(stamps))
        assert len({r["series_id"] for r in rows}) == 1
