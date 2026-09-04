"""Station catalog (``stations()``) tests for the six sources that expose one (#187).

Every test injects a MagicMock HTTP client whose payloads are shaped like the
live responses observed on 2026-08-16, so nothing here touches the network.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from aquascope.collectors.bom import PARAMETER_VARIABLE_MAP, BOMCollector
from aquascope.collectors.france_hubeau import HubeauHydrometrieCollector
from aquascope.collectors.ireland_opw import IrelandOPWCollector
from aquascope.collectors.pegelonline import PegelonlineCollector
from aquascope.collectors.taiwan_cwa import TaiwanCWACollector
from aquascope.collectors.uk_ea import UKEACollector
from aquascope.collectors.usgs import USGSCollector
from aquascope.schemas.station import Station

# ─── USGS ────────────────────────────────────────────────────────────────────

USGS_TS_PAGE = {
    "type": "FeatureCollection",
    "features": [
        {
            "geometry": {"type": "Point", "coordinates": [-77.2458, 38.9759]},
            "properties": {
                "monitoring_location_id": "USGS-01646000",
                "parameter_code": "00060",
                "begin": "1935-04-01T00:00:00.000001",
                "end": "2026-08-14T00:00:00.000001",
                "computation_identifier": "Mean",
            },
        },
        {
            "geometry": {"type": "Point", "coordinates": [-77.2458, 38.9759]},
            "properties": {
                "monitoring_location_id": "USGS-01646000",
                "parameter_code": "00010",
                "begin": "2001-01-01T00:00:00.000001",
                "end": "2010-01-01T00:00:00.000001",
                "computation_identifier": "Mean",
            },
        },
        {
            # no geometry: must be skipped
            "geometry": None,
            "properties": {"monitoring_location_id": "USGS-99999999", "parameter_code": "00060"},
        },
        {
            # unmapped parameter: skipped
            "geometry": {"type": "Point", "coordinates": [-77.1, 38.9]},
            "properties": {"monitoring_location_id": "USGS-01646001", "parameter_code": "72019"},
        },
    ],
    "links": [{"rel": "self", "href": "x"}],
}
USGS_LOC_PAGE = {
    "features": [
        {"properties": {"id": "USGS-01646000", "monitoring_location_name": "DIFFICULT RUN NEAR GREAT FALLS, VA"}},
        {"properties": {"id": "USGS-00000000", "monitoring_location_name": "not requested"}},
    ],
    "links": [],
}


class TestUSGSStations:
    def _collector(self):
        client = MagicMock()

        def get_json(path, params=None, **kw):
            if "time-series-metadata" in path:
                return USGS_TS_PAGE
            if "monitoring-locations" in path:
                return USGS_LOC_PAGE
            raise AssertionError(path)

        client.get_json.side_effect = get_json
        return USGSCollector(api_key="DEMO_KEY", client=client), client

    def test_joins_series_and_names(self):
        collector, client = self._collector()
        stations = collector.stations(bbox=(-77.3, 38.8, -77.0, 39.0))
        assert len(stations) == 1
        st = stations[0]
        assert isinstance(st, Station)
        assert st.station_id == "USGS-01646000"
        assert st.name == "DIFFICULT RUN NEAR GREAT FALLS, VA"
        assert st.variables == ("discharge", "water_quality")
        assert st.period_start == date(1935, 4, 1)
        assert st.period_end == date(2026, 8, 14)
        assert st.url.endswith("/monitoring-location/01646000/")
        # bbox is forwarded to the API
        first_call = client.get_json.call_args_list[0]
        assert first_call.kwargs["params"]["bbox"] == "-77.3,38.8,-77.0,39.0"

    def test_variable_filter_sets_parameter_code(self):
        collector, client = self._collector()
        stations = collector.stations(variable="discharge")
        assert stations[0].variables == ("discharge",)
        assert client.get_json.call_args_list[0].kwargs["params"]["parameter_code"] == "00060"

    def test_unknown_variable_returns_empty_without_calls(self):
        collector, client = self._collector()
        assert collector.stations(variable="reservoir_storage") == []
        client.get_json.assert_not_called()

    def test_pagination_follows_next_and_caps(self):
        page1 = {
            "features": [
                {
                    "geometry": {"type": "Point", "coordinates": [-100.0 - i, 40.0]},
                    "properties": {"monitoring_location_id": f"USGS-{i:08d}", "parameter_code": "00060"},
                }
                for i in range(3)
            ],
            "links": [{"rel": "next", "href": "https://example/next"}],
        }
        page2 = {
            "features": [
                {
                    "geometry": {"type": "Point", "coordinates": [-90.0, 40.0]},
                    "properties": {"monitoring_location_id": "USGS-00000099", "parameter_code": "00060"},
                }
            ],
            "links": [],
        }
        client = MagicMock()
        calls = []

        def get_json(path, params=None, **kw):
            calls.append((path, params))
            if "monitoring-locations" in path:
                return {"features": [], "links": []}
            return page2 if path.startswith("https://example/next") else page1

        client.get_json.side_effect = get_json
        collector = USGSCollector(api_key="DEMO_KEY", client=client)
        assert len(collector.stations()) == 4
        # the absolute next link is fetched with params=None so its query string survives
        assert ("https://example/next", None) in calls
        assert len(collector.stations(max_items=2)) == 2

    def test_nwis_state_sweep_uses_aq_not_as(self):
        client = MagicMock()
        client.get_text.return_value = ""
        collector = USGSCollector(api_key="DEMO_KEY", client=client)
        collector._nwis_site_names(None)
        state_codes = {
            call.kwargs["params"]["stateCd"]
            for call in client.get_text.call_args_list
        }
        assert "AQ" in state_codes
        assert "AS" not in state_codes


# ─── UK EA ───────────────────────────────────────────────────────────────────

UKEA_PAGE = {
    "items": [
        {
            "@id": "http://environment.data.gov.uk/hydrology/id/stations/abc",
            "label": "Crayford",
            "notation": "abc",
            "lat": 51.44984,
            "long": 0.172527,
            "riverName": ["River Cray", "Cray"],
            "wiskiID": "451220001",
            "dateOpened": "1969-07-01",
            "observedProperty": [
                {"@id": "http://environment.data.gov.uk/reference/def/op/waterFlow"},
                {"@id": "http://environment.data.gov.uk/reference/def/op/waterLevel"},
            ],
        },
        {
            "@id": "http://environment.data.gov.uk/hydrology/id/stations/outside",
            "label": "Far away",
            "notation": "outside",
            "lat": 55.0,
            "long": -3.0,
            "observedProperty": {"@id": "http://environment.data.gov.uk/reference/def/op/rainfall"},
        },
        {"label": "no coords", "notation": "nc"},
    ]
}


class TestUKEAStations:
    def test_parses_and_filters_bbox(self):
        client = MagicMock()
        client.get_json.return_value = UKEA_PAGE
        stations = UKEACollector(client=client).stations(bbox=(-0.2, 51.4, 0.2, 51.6))
        assert [s.station_id for s in stations] == ["abc"]
        st = stations[0]
        assert st.variables == ("discharge", "water_level")
        assert st.river == "River Cray"
        assert st.period_start == date(1969, 7, 1)
        assert st.extra["wiskiID"] == "451220001"
        params = client.get_json.call_args.kwargs["params"]
        assert params["mineq-lat"] == 51.4 and params["maxeq-long"] == 0.2

    def test_variable_maps_to_observed_property(self):
        client = MagicMock()
        client.get_json.return_value = {"items": []}
        UKEACollector(client=client).stations(variable="groundwater_level")
        assert client.get_json.call_args.kwargs["params"]["observedProperty"] == "groundwaterLevel"
        assert UKEACollector(client=client).stations(variable="reservoir_storage") == []

    def test_no_bbox_keeps_everything_with_coords(self):
        client = MagicMock()
        client.get_json.return_value = UKEA_PAGE
        stations = UKEACollector(client=client).stations()
        assert {s.station_id for s in stations} == {"abc", "outside"}
        assert next(s for s in stations if s.station_id == "outside").variables == ("precipitation",)


# ─── Hub'Eau ─────────────────────────────────────────────────────────────────

HUBEAU_PAGE1 = {
    "count": 2,
    "next": "https://hubeau.eaufrance.fr/api/v2/hydrometrie/referentiel/stations?page=2",
    "data": [
        {
            "code_station": "F664000301",
            "libelle_station": "La Marne à Joinville-le-Pont",
            "code_site": "F6640003",
            "longitude_station": 2.476,
            "latitude_station": 48.815,
            "libelle_cours_eau": "La Marne",
            "date_ouverture_station": "1930-01-01T00:00:00Z",
            "date_fermeture_station": None,
            "type_station": "STD",
        },
    ],
}
HUBEAU_PAGE2 = {
    "count": 2,
    "next": None,
    "data": [
        {"code_station": "X1", "longitude_station": 5.0, "latitude_station": 45.0, "libelle_station": "Elsewhere"},
        {"code_station": "X2", "longitude_station": None, "latitude_station": None},
    ],
}


class TestHubeauStations:
    def test_paginates_and_filters(self):
        client = MagicMock()
        client.get_json.side_effect = [HUBEAU_PAGE1, HUBEAU_PAGE2]
        stations = HubeauHydrometrieCollector(client=client).stations()
        assert [s.station_id for s in stations] == ["F664000301", "X1"]
        st = stations[0]
        assert st.variables == ("discharge", "water_level")
        assert st.period_start == date(1930, 1, 1) and st.period_end is None
        assert st.url == "https://www.hydro.eaufrance.fr/sitehydro/F6640003/fiche"
        # page 2 was fetched via its absolute URL with no params
        second = client.get_json.call_args_list[1]
        assert second.args[0].startswith("https://hubeau") and second.kwargs["params"] is None

    def test_bbox_forwarded_and_rechecked(self):
        client = MagicMock()
        client.get_json.side_effect = [HUBEAU_PAGE1, HUBEAU_PAGE2]
        stations = HubeauHydrometrieCollector(client=client).stations(bbox=(2.2, 48.8, 2.5, 48.9))
        assert [s.station_id for s in stations] == ["F664000301"]
        assert client.get_json.call_args_list[0].kwargs["params"]["bbox"] == "2.2,48.8,2.5,48.9"

    def test_unknown_variable(self):
        client = MagicMock()
        assert HubeauHydrometrieCollector(client=client).stations(variable="climate") == []
        client.get_json.assert_not_called()


# ─── PEGELONLINE ─────────────────────────────────────────────────────────────

PEGEL_PAGE = [
    {
        "uuid": "u1",
        "number": "48300105",
        "shortname": "CELLE",
        "longname": "CELLE",
        "km": 1.74,
        "agency": "VERDEN",
        "longitude": 10.062,
        "latitude": 52.622,
        "water": {"shortname": "ALLER", "longname": "ALLER"},
        "timeseries": [{"shortname": "W", "unit": "cm"}, {"shortname": "Q", "unit": "m³/s"}],
    },
    {
        "uuid": "u2",
        "number": "1",
        "longname": "ONLY W",
        "longitude": 8.0,
        "latitude": 50.0,
        "water": {"longname": "RHEIN"},
        "timeseries": [{"shortname": "W"}],
    },
    {"uuid": "u3", "longname": "no coords"},
]


class TestPegelonlineStations:
    def test_variables_from_timeseries(self):
        client = MagicMock()
        client.get_json.return_value = PEGEL_PAGE
        stations = PegelonlineCollector(client=client).stations()
        assert [s.station_id for s in stations] == ["u1", "u2"]
        assert stations[0].variables == ("discharge", "water_level")
        assert stations[0].river == "ALLER"
        assert stations[0].extra["number"] == "48300105"
        assert stations[1].variables == ("water_level",)

    def test_variable_and_bbox_filters(self):
        client = MagicMock()
        client.get_json.return_value = PEGEL_PAGE
        collector = PegelonlineCollector(client=client)
        assert [s.station_id for s in collector.stations(variable="discharge")] == ["u1"]
        assert [s.station_id for s in collector.stations(bbox=(7.0, 49.0, 9.0, 51.0))] == ["u2"]
        assert collector.stations(variable="climate") == []


# ─── Ireland OPW ─────────────────────────────────────────────────────────────

OPW_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-7.5757, 54.8383]},
            "properties": {"name": "Sandy Mills", "ref": "0000001041"},
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-6.2, 53.3]},
            "properties": {"name": "No ref"},
        },
    ],
}


class TestIrelandStations:
    def test_geojson_to_stations(self):
        client = MagicMock()
        client.get_json.return_value = OPW_GEOJSON
        stations = IrelandOPWCollector(client=client).stations()
        assert len(stations) == 1
        st = stations[0]
        assert st.station_id == "0000001041" and st.name == "Sandy Mills"
        assert (st.latitude, st.longitude) == (54.8383, -7.5757)
        assert st.variables == ("water_level",)
        assert st.url == "https://waterlevel.ie/0001/1041/"

    def test_filters(self):
        client = MagicMock()
        client.get_json.return_value = OPW_GEOJSON
        collector = IrelandOPWCollector(client=client)
        assert collector.stations(variable="discharge") == []
        assert collector.stations(bbox=(-6.5, 53.0, -6.0, 53.5)) == []


# ─── Taiwan CWA ──────────────────────────────────────────────────────────────

CWA_STATION_LIST = {
    "data": [
        {
            "stationAttribute": "cwb",
            "item": [
                {
                    "stationID": "466920",
                    "stationName": "臺北",
                    "altitude": 5.3,
                    "longitude": 121.5149,
                    "latitude": 25.0377,
                    "countryName": "臺北市",
                    "stationStartDate": "1896-01-01",
                    "stationEndDate": "",
                },
            ],
        },
        {
            "stationAttribute": "auto",
            "item": [
                {
                    "stationID": "C0A9F0",
                    "stationName": "自動站",
                    "longitude": 121.6,
                    "latitude": 24.9,
                    "stationStartDate": "2010-05-01",
                    "stationEndDate": "2020-01-31",
                },
                {"stationID": "NOCOORD", "stationName": "x"},
            ],
        },
    ]
}


class TestTaiwanCWAStations:
    def test_station_list_to_stations(self):
        client = MagicMock()
        client.post_json.return_value = CWA_STATION_LIST
        stations = TaiwanCWACollector(client=client).stations()
        assert [s.station_id for s in stations] == ["466920", "C0A9F0"]
        st = stations[0]
        assert st.name == "臺北" and st.country == "TWN"
        assert st.variables == ("climate", "precipitation")
        assert st.period_start == date(1896, 1, 1) and st.period_end is None
        assert st.extra == {"attribute": "cwb", "altitude_m": 5.3, "county": "臺北市"}
        assert stations[1].period_end == date(2020, 1, 31)
        client.post_json.assert_called_once_with("station_list", form={})

    def test_filters(self):
        client = MagicMock()
        client.post_json.return_value = CWA_STATION_LIST
        collector = TaiwanCWACollector(client=client)
        assert [s.station_id for s in collector.stations(bbox=(121.5, 25.0, 121.6, 25.1))] == ["466920"]
        assert collector.stations(variable="discharge") == []
        assert len(collector.stations(max_items=1)) == 1


# ─── BOM (Australia) ────────────────────────────────────────────────────────
#
# BOM's KiWIS instance only fills in station_latitude/station_longitude/
# parametertype_name when getStationList is filtered by a single
# parametertype_name (confirmed live 2026-08-25); an unfiltered request
# returns those fields blank for every row. So stations() issues one
# request per BOM parameter type and merges by station_no -- these mocks
# key a payload per parametertype_name, shaped like the live responses.

_BOM_HEADER = ["station_no", "station_name", "station_latitude", "station_longitude", "parametertype_name"]
BOM_NO_MATCHES = ["No matches."]

BOM_BY_PARAMETER_TYPE = {
    "Water Course Discharge": [
        _BOM_HEADER,
        ["410001", "M/BIDGEE R @ WAGGA", "-35.1082", "147.3598", "Water Course Discharge"],
    ],
    "Water Course Level": [
        _BOM_HEADER,
        ["410001", "M/BIDGEE R @ WAGGA", "-35.1082", "147.3598", "Water Course Level"],
    ],
    "Storage Level": [
        _BOM_HEADER,
        ["410130", "BURRINJUCK DAM", "-35.0", "148.6", "Storage Level"],
    ],
    "Storage Volume": [
        _BOM_HEADER,
        ["410130", "BURRINJUCK DAM", "-35.0", "148.6", "Storage Volume"],
    ],
    "Rainfall": [
        _BOM_HEADER,
        # KiWIS sometimes still returns blank coordinates for a given row
        # even inside a filtered response -- must be dropped, not raise.
        ["9999999", "NO COORDS", "", "", "Rainfall"],
    ],
    # BOM spells this parameter type with an "@", not the word "At" --
    # PARAMETER_VARIABLE_MAP must use the same spelling or every EC row
    # silently maps to nothing.
    "Electrical Conductivity @ 25C": [
        _BOM_HEADER,
        ["410001", "M/BIDGEE R @ WAGGA", "-35.1082", "147.3598", "Electrical Conductivity @ 25C"],
    ],
    "Ground Water Level": [
        _BOM_HEADER,
        # Real "unset location" garbage seen live from BOM's API 2026-08-25
        # -- clusters around lat=-85.5 regardless of the station's actual
        # location. Must be dropped, not returned as a real station.
        ["GW041027", "MAULES CK Thornfield", "-85.5275", "56.7670", "Ground Water Level"],
        # A legitimate external territory station (Christmas Island) sits
        # well outside the mainland+Tasmania bbox but must survive the
        # Australia-wide sanity filter.
        ["CHRISTMAS01", "Christmas Island Bore", "-10.4900", "105.6300", "Ground Water Level"],
    ],
}


def _bom_client(by_parameter_type: dict = BOM_BY_PARAMETER_TYPE) -> MagicMock:
    """MagicMock client whose ``getStationList`` response depends on the
    single ``parametertype_name`` in the call's params, defaulting to
    ``"No matches."`` for any type not given an explicit payload."""
    client = MagicMock()

    def get_json(url, params=None, **kw):
        return by_parameter_type.get(params["parametertype_name"], BOM_NO_MATCHES)

    client.get_json.side_effect = get_json
    return client


class TestBOMStations:
    def test_groups_parameter_types_per_station(self):
        client = _bom_client()
        stations = BOMCollector(client=client).stations()
        assert [s.station_id for s in stations] == ["410001", "410130", "CHRISTMAS01"]

        wagga = stations[0]
        assert isinstance(wagga, Station)
        assert wagga.name == "M/BIDGEE R @ WAGGA"
        assert (wagga.latitude, wagga.longitude) == (-35.1082, 147.3598)
        assert wagga.variables == ("discharge", "water_level", "water_quality")
        assert wagga.country == "AUS"
        assert wagga.url.endswith("/410001")

        dam = stations[1]
        assert dam.variables == ("reservoir_storage",)

        # row with blank coordinates is dropped rather than raising
        assert "9999999" not in [s.station_id for s in stations]
        # row with implausible (sentinel) coordinates is dropped too
        assert "GW041027" not in [s.station_id for s in stations]

        # one getStationList call per BOM parameter type, each filtered
        assert client.get_json.call_count == len(PARAMETER_VARIABLE_MAP)
        requested_types = {c.kwargs["params"]["parametertype_name"] for c in client.get_json.call_args_list}
        assert requested_types == set(PARAMETER_VARIABLE_MAP)

    def test_drops_implausible_coordinates_but_keeps_external_territories(self, caplog):
        client = _bom_client()
        stations = BOMCollector(client=client).stations(variable="groundwater_level")
        station_ids = [s.station_id for s in stations]

        # the -85.5-latitude sentinel is dropped...
        assert "GW041027" not in station_ids
        # ...but a real external-territory station (outside the mainland
        # bbox, inside the Australia-wide sanity bbox) is kept.
        assert "CHRISTMAS01" in station_ids
        christmas = next(s for s in stations if s.station_id == "CHRISTMAS01")
        assert (christmas.latitude, christmas.longitude) == (-10.49, 105.63)

    def test_variable_filter_narrows_requests_and_result(self):
        client = _bom_client()
        collector = BOMCollector(client=client)
        stations = collector.stations(variable="reservoir_storage")
        assert [s.station_id for s in stations] == ["410130"]

        # only the two reservoir_storage parameter types are requested,
        # each in its own call (not comma-joined into one) -- "Storage
        # Percentage Full" was removed: it's not a real BOM parameter type
        # (0 live rows, absent from getParameterList), and Storage Level +
        # Storage Volume already cover reservoir_storage.
        requested_types = {c.kwargs["params"]["parametertype_name"] for c in client.get_json.call_args_list}
        assert requested_types == {"Storage Level", "Storage Volume"}
        assert client.get_json.call_count == 2

    def test_unknown_variable_returns_empty_without_calls(self):
        client = _bom_client()
        collector = BOMCollector(client=client)
        assert collector.stations(variable="climate") == []
        client.get_json.assert_not_called()

    def test_bbox_forwarded_and_rechecked(self):
        client = _bom_client()
        collector = BOMCollector(client=client)
        stations = collector.stations(bbox=(147.0, -35.2, 148.0, -35.0))
        # 410130 sits at lon 148.6, outside the bbox's east=148.0 -- dropped
        # client-side even though the mock doesn't filter server-side.
        assert [s.station_id for s in stations] == ["410001"]
        for call in client.get_json.call_args_list:
            assert call.kwargs["params"]["bbox"] == "147.0,-35.2,148.0,-35.0"

    def test_max_items_caps_result(self):
        client = _bom_client()
        stations = BOMCollector(client=client).stations(max_items=1)
        assert len(stations) == 1

    def test_no_matches_returns_empty(self):
        client = _bom_client(by_parameter_type={})
        assert BOMCollector(client=client).stations() == []

    def test_electrical_conductivity_maps_to_water_quality(self):
        # Regression test for the "At" vs "@" spelling mismatch: BOM's
        # KiWIS instance uses "Electrical Conductivity @ 25C", and a wrong
        # spelling in PARAMETER_VARIABLE_MAP means this request would
        # return 0 rows (a live check found 3,599 real rows for the
        # correct spelling vs. 0 for "At").
        client = _bom_client()
        stations = BOMCollector(client=client).stations(variable="water_quality")
        requested_types = {c.kwargs["params"]["parametertype_name"] for c in client.get_json.call_args_list}
        assert "Electrical Conductivity @ 25C" in requested_types
        wagga = next(s for s in stations if s.station_id == "410001")
        assert "water_quality" in wagga.variables

    def test_one_parameter_type_failing_does_not_fail_the_whole_catalog(self, caplog):
        client = MagicMock()

        def get_json(url, params=None, **kw):
            if params["parametertype_name"] == "Water Course Discharge":
                raise RuntimeError("boom")
            return BOM_BY_PARAMETER_TYPE.get(params["parametertype_name"], BOM_NO_MATCHES)

        client.get_json.side_effect = get_json
        with caplog.at_level("WARNING"):
            stations = BOMCollector(client=client).stations()
        # the failing type is skipped, but stations from other types still come back
        assert "410130" in [s.station_id for s in stations]
        # summary warning reports the lost parameter type
        assert "BOM getStationList failed for 1 of" in caplog.text
        assert "Water Course Discharge" in caplog.text

    def test_all_parameter_types_failing_raises(self):
        client = MagicMock()
        client.get_json.side_effect = RuntimeError("boom")
        with pytest.raises(
            RuntimeError,
            match=rf"BOM getStationList failed for all {len(PARAMETER_VARIABLE_MAP)} parameter type\(s\)",
        ):
            BOMCollector(client=client).stations()

    def test_single_variable_failing_raises(self):
        client = MagicMock()
        client.get_json.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match=r"BOM getStationList failed for all 1 parameter type\(s\)"):
            BOMCollector(client=client).stations(variable="discharge")
