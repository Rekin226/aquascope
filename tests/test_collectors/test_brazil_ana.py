"""Tests for the Brazil ANA Hidroweb collector.

fetch_raw/stations are tested with mocked HTTP; normalise() and the
QC/datetime parsing helpers are tested directly with hand-built fixture
data shaped like ANA's documented API responses - no network calls.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from unittest.mock import MagicMock

import pytest

from aquascope.collectors.brazil_ana import (
    BrazilANACollector,
    _parse_ana_datetime,
    _parse_historical_xml,
    _qc_remark,
)
from aquascope.schemas.water_data import ClimateReading, DataSource, StreamflowReading, WaterLevelReading

FAKE_STATIONS_PAGE = {
    "features": [
        {
            "attributes": {
                "Codigo": 15400000,
                "Nome": "Boa Esperanca",
                "TipoEstacao": "Fluviométrica",
                "Latitude": -13.5,
                "Longitude": -47.9,
                "Rio": "Rio Tocantins",
                "UF": "GO",
                "Municipio": "Some City",
                "Bacia": "Tocantins",
                "Operando": "Sim",
            }
        },
        {
            "attributes": {
                "Codigo": 15400001,
                "Nome": "Chuva Station",
                "TipoEstacao": "Pluviométrica",
                "Latitude": -14.0,
                "Longitude": -48.2,
                "Rio": None,
                "UF": "GO",
                "Municipio": "Other City",
                "Bacia": "Tocantins",
                "Operando": "Sim",
            }
        },
        {
            # Missing coordinates - must be skipped
            "attributes": {
                "Codigo": 99999999,
                "Nome": "No Coords",
                "TipoEstacao": "Fluviométrica",
                "Latitude": None,
                "Longitude": None,
            }
        },
    ]
}

FAKE_AUTH_RESPONSE = {
    "status": "OK",
    "code": 200,
    "message": "Sucesso",
    "items": {"tokenautenticacao": "fake-token-abc"},
}

FAKE_SERIES_RESPONSE = {
    "status": "OK",
    "code": 200,
    "message": "Sucesso",
    "items": [
        {
            "Chuva_Adotada": "0.00",
            "Chuva_Adotada_Status": "0",
            "Cota_Adotada": "781.00",
            "Cota_Adotada_Status": "0",
            "Data_Atualizacao": "2026-01-02 00:28:03.307",
            "Data_Hora_Medicao": "2026-01-01 23:00:00.0",
            "Vazao_Adotada": "13225.42",
            "Vazao_Adotada_Status": "1",
            "codigoestacao": "15400000",
        },
        {
            # No discharge/level at all, only rainfall - all three readings possible
            "Chuva_Adotada": "5.20",
            "Chuva_Adotada_Status": "2",
            "Cota_Adotada": None,
            "Cota_Adotada_Status": None,
            "Data_Hora_Medicao": "2026-01-02 00:00:00.0",
            "Vazao_Adotada": None,
            "Vazao_Adotada_Status": None,
            "codigoestacao": "15400000",
        },
        {
            # Completely empty row - must be skipped without raising
            "Chuva_Adotada": None,
            "Cota_Adotada": "",
            "Vazao_Adotada": "null",
            "Data_Hora_Medicao": "2026-01-03 00:00:00.0",
            "codigoestacao": "15400000",
        },
        {
            # Missing timestamp entirely - must be skipped
            "Chuva_Adotada": "1.0",
            "codigoestacao": "15400000",
        },
    ],
}


class TestBrazilANAStations:
    def test_stations_parses_page_and_filters_missing_coords(self):
        mock_client = MagicMock()
        mock_client.get_json.side_effect = [FAKE_STATIONS_PAGE, {"features": []}]
        collector = BrazilANACollector(client=mock_client)

        stations = collector.stations()

        assert len(stations) == 2
        river_station = next(s for s in stations if s.station_id == "15400000")
        assert river_station.name == "Boa Esperanca"
        assert river_station.variables == ("discharge", "water_level")
        assert river_station.latitude == -13.5
        assert river_station.country == "BRA"

        rain_station = next(s for s in stations if s.station_id == "15400001")
        assert rain_station.variables == ("precipitation",)

    def test_stations_no_auth_required(self):
        """Station catalog must not touch the OAuth endpoint."""
        mock_client = MagicMock()
        mock_client.get_json.side_effect = [FAKE_STATIONS_PAGE, {"features": []}]
        collector = BrazilANACollector(client=mock_client)  # no credentials passed

        collector.stations()

        called_urls = [call.args[0] for call in mock_client.get_json.call_args_list]
        assert all("OAUth" not in url for url in called_urls)

    def test_stations_respects_variable_filter(self):
        mock_client = MagicMock()
        mock_client.get_json.side_effect = [FAKE_STATIONS_PAGE, {"features": []}]
        collector = BrazilANACollector(client=mock_client)

        stations = collector.stations(variable="precipitation")
        assert [s.station_id for s in stations] == ["15400001"]

    def test_stations_respects_max_items(self):
        mock_client = MagicMock()
        mock_client.get_json.return_value = FAKE_STATIONS_PAGE
        collector = BrazilANACollector(client=mock_client)

        stations = collector.stations(max_items=1)
        assert len(stations) == 1


class TestBrazilANAFetchRaw:
    def test_fetch_raw_requires_station_ids(self):
        collector = BrazilANACollector(client=MagicMock())
        with pytest.raises(ValueError):
            collector.fetch_raw()

    def test_fetch_raw_requires_credentials(self):
        collector = BrazilANACollector(client=MagicMock())
        with pytest.raises(RuntimeError):
            collector.fetch_raw(station_ids=["15400000"])

    def test_fetch_raw_authenticates_and_fetches_series(self):
        mock_client = MagicMock()
        mock_client.get_json.side_effect = [FAKE_AUTH_RESPONSE, FAKE_SERIES_RESPONSE]
        collector = BrazilANACollector(identificador="12345678900", senha="secret", client=mock_client)

        rows = collector.fetch_raw(station_ids="15400000")

        assert len(rows) == 4
        auth_call, series_call = mock_client.get_json.call_args_list
        assert "OAUth" in auth_call.args[0]
        assert auth_call.kwargs["headers"] == {"Identificador": "12345678900", "Senha": "secret"}
        assert series_call.kwargs["headers"] == {"Authorization": "Bearer fake-token-abc"}
        assert series_call.kwargs["params"]["codEstacao"] == "15400000"

    def test_fetch_raw_reuses_cached_token_across_stations(self):
        mock_client = MagicMock()
        mock_client.get_json.side_effect = [
            FAKE_AUTH_RESPONSE,
            FAKE_SERIES_RESPONSE,
            {"items": []},
        ]
        collector = BrazilANACollector(identificador="x", senha="y", client=mock_client)

        collector.fetch_raw(station_ids=["15400000", "15400001"])

        auth_calls = [c for c in mock_client.get_json.call_args_list if "OAUth" in c.args[0]]
        assert len(auth_calls) == 1

    def test_fetch_raw_skips_station_on_error(self):
        mock_client = MagicMock()
        mock_client.get_json.side_effect = [
            FAKE_AUTH_RESPONSE,
            RuntimeError("boom"),
            FAKE_SERIES_RESPONSE,
        ]
        collector = BrazilANACollector(identificador="x", senha="y", client=mock_client)

        rows = collector.fetch_raw(station_ids=["bad_station", "15400000"])
        assert len(rows) == 4  # only the second (working) station's rows

    def test_fetch_raw_raises_when_every_station_fails(self):
        """A shape drift/dead endpoint must surface as an error, not silently look like '[] records'."""
        mock_client = MagicMock()
        mock_client.get_json.side_effect = [
            FAKE_AUTH_RESPONSE,
            RuntimeError("boom 1"),
            RuntimeError("boom 2"),
        ]
        collector = BrazilANACollector(identificador="x", senha="y", client=mock_client)

        with pytest.raises(RuntimeError, match="all 2 station"):
            collector.fetch_raw(station_ids=["bad_1", "bad_2"])

    def test_fetch_raw_raises_when_response_missing_items_key(self):
        """Missing 'items' entirely (vs. present-but-empty) signals a shape drift, not 'no data'."""
        mock_client = MagicMock()
        mock_client.get_json.side_effect = [FAKE_AUTH_RESPONSE, {"status": "OK"}]  # no "items" key
        collector = BrazilANACollector(identificador="x", senha="y", client=mock_client)

        with pytest.raises(RuntimeError, match="unexpected response shape|all 1 station"):
            collector.fetch_raw(station_ids=["15400000"])

    def test_fetch_raw_genuinely_empty_items_does_not_raise(self):
        """A station with zero readings this period ('items': []) is legitimate, not a failure."""
        mock_client = MagicMock()
        mock_client.get_json.side_effect = [FAKE_AUTH_RESPONSE, {"status": "OK", "items": []}]
        collector = BrazilANACollector(identificador="x", senha="y", client=mock_client)

        rows = collector.fetch_raw(station_ids=["15400000"])
        assert rows == []

    def test_fetch_raw_auth_failure_raises(self):
        mock_client = MagicMock()
        mock_client.get_json.return_value = {"status": "ERROR", "items": {}}
        collector = BrazilANACollector(identificador="x", senha="y", client=mock_client)

        with pytest.raises(RuntimeError):
            collector.fetch_raw(station_ids=["15400000"])


class TestBrazilANANormalise:
    def test_normalise_splits_row_into_three_reading_types(self):
        mock_client = MagicMock()
        mock_client.get_json.side_effect = [FAKE_STATIONS_PAGE, {"features": []}]
        collector = BrazilANACollector(client=mock_client)

        readings = collector.normalise(FAKE_SERIES_RESPONSE["items"])

        # Row 1: discharge + level + (zero) rainfall = 3 readings.
        # Row 2: rainfall only = 1 reading.
        # Row 3 (all blank/null): 0 readings.
        # Row 4 (no timestamp): 0 readings.
        assert len(readings) == 4

        streamflow = [r for r in readings if isinstance(r, StreamflowReading)]
        assert len(streamflow) == 1
        assert streamflow[0].discharge_cms == 13225.42
        assert streamflow[0].source == DataSource.BRAZIL_ANA
        assert streamflow[0].station_id == "15400000"
        assert streamflow[0].source_type == "in_situ"
        assert streamflow[0].remark == _qc_remark("1")

        levels = [r for r in readings if isinstance(r, WaterLevelReading)]
        assert len(levels) == 1
        assert levels[0].water_level == pytest.approx(7.81)  # 781 cm -> 7.81 m

        rainfall = [r for r in readings if isinstance(r, ClimateReading)]
        assert len(rainfall) == 2
        assert {r.value for r in rainfall} == {0.0, 5.2}
        assert rainfall[0].parameter == "rainfall_mm"

    def test_normalise_enriches_with_station_metadata(self):
        mock_client = MagicMock()
        mock_client.get_json.side_effect = [FAKE_STATIONS_PAGE, {"features": []}]
        collector = BrazilANACollector(client=mock_client)

        readings = collector.normalise([FAKE_SERIES_RESPONSE["items"][0]])
        streamflow = next(r for r in readings if isinstance(r, StreamflowReading))

        assert streamflow.station_name == "Boa Esperanca"
        assert streamflow.location is not None
        assert streamflow.location.latitude == -13.5
        assert streamflow.location.longitude == -47.9

    def test_normalise_survives_station_catalog_failure(self):
        """If the enrichment lookup fails, normalise() still returns readings (no crash)."""
        mock_client = MagicMock()
        mock_client.get_json.side_effect = ConnectionError("offline")
        collector = BrazilANACollector(client=mock_client)

        readings = collector.normalise([FAKE_SERIES_RESPONSE["items"][0]])
        assert len(readings) == 3
        streamflow = next(r for r in readings if isinstance(r, StreamflowReading))
        assert streamflow.location is None
        assert streamflow.station_name is None

    def test_normalise_skips_malformed_rows(self):
        collector = BrazilANACollector(client=MagicMock())
        bad_rows = [
            {"codigoestacao": "1", "Data_Hora_Medicao": "not-a-date", "Vazao_Adotada": "1.0"},
            {"codigoestacao": "", "Data_Hora_Medicao": "2026-01-01 00:00:00.0", "Vazao_Adotada": "1.0"},
        ]
        readings = collector.normalise(bad_rows)
        assert readings == []


class TestBrazilANAHelpers:
    def test_parse_ana_datetime_with_fractional_seconds(self):
        dt = _parse_ana_datetime("2026-01-01 23:00:00.0")
        assert dt.year == 2026 and dt.hour == 23

    def test_parse_ana_datetime_without_fractional_seconds(self):
        dt = _parse_ana_datetime("2026-01-01 23:00:00")
        assert dt.hour == 23

    def test_parse_ana_datetime_date_only(self):
        dt = _parse_ana_datetime("2026-01-01")
        assert dt.hour == 0

    def test_parse_ana_datetime_rejects_garbage(self):
        with pytest.raises(ValueError):
            _parse_ana_datetime("not-a-date")

    def test_qc_remark_maps_known_flags(self):
        assert _qc_remark("0") is None
        assert _qc_remark(None) is None
        assert "suspect" in _qc_remark("1")
        assert "poor" in _qc_remark("2")
        assert _qc_remark("9") is None


# ── historical (conventional-network) path ──────────────────────────────
# Field names/structure below are transcribed from community-documented
# HidroSerieHistorica usage, not a live response (see module docstring).

FAKE_HISTORICAL_XML_TWO_LEVELS = """<?xml version="1.0" encoding="utf-8"?>
<DocumentElement>
  <SerieHistorica>
    <EstacaoCodigo>15400000</EstacaoCodigo>
    <NivelConsistencia>1</NivelConsistencia>
    <Data>2020-01-01T00:00:00</Data>
    <Vazao01>100.0</Vazao01>
    <Vazao02>101.0</Vazao02>
  </SerieHistorica>
  <SerieHistorica>
    <EstacaoCodigo>15400000</EstacaoCodigo>
    <NivelConsistencia>2</NivelConsistencia>
    <Data>2020-01-01T00:00:00</Data>
    <Vazao01>102.5</Vazao01>
    <Vazao02>103.5</Vazao02>
    <Vazao31>999.0</Vazao31>
  </SerieHistorica>
  <SerieHistorica>
    <!-- ANA's "relisted month" quirk: same month, day != 01 -->
    <EstacaoCodigo>15400000</EstacaoCodigo>
    <NivelConsistencia>2</NivelConsistencia>
    <Data>2020-01-31T00:00:00</Data>
    <Vazao01>-1.0</Vazao01>
  </SerieHistorica>
</DocumentElement>
"""

FAKE_HISTORICAL_XML_SINGLE = """<?xml version="1.0" encoding="utf-8"?>
<DocumentElement>
  <SerieHistorica>
    <EstacaoCodigo>15400000</EstacaoCodigo>
    <NivelConsistencia>1</NivelConsistencia>
    <Data>2020-02-01T00:00:00</Data>
    <Cota01>500</Cota01>
  </SerieHistorica>
</DocumentElement>
"""


class TestParseHistoricalXML:
    def test_parses_rows_into_field_dicts(self):
        rows = _parse_historical_xml(FAKE_HISTORICAL_XML_TWO_LEVELS)
        assert len(rows) == 3
        assert rows[0]["EstacaoCodigo"] == "15400000"
        assert rows[0]["NivelConsistencia"] == "1"
        assert rows[0]["Vazao01"] == "100.0"

    def test_malformed_xml_raises_parse_error(self):
        """Genuinely broken/unparseable XML must not look identical to 'no data'."""
        with pytest.raises(ET.ParseError):
            _parse_historical_xml("<not><valid")

    def test_empty_document_returns_empty_list(self):
        """A well-formed document with zero row elements is ANA's legitimate 'no data' shape."""
        assert _parse_historical_xml("<DocumentElement></DocumentElement>") == []


class TestBrazilANAFetchHistoricalRaw:
    def test_rejects_unknown_variable(self):
        collector = BrazilANACollector(client=MagicMock(), legacy_client=MagicMock())
        with pytest.raises(ValueError):
            collector.fetch_raw(station_ids="15400000", mode="historical", variables=("not_a_variable",))

    def test_rejects_unknown_mode(self):
        collector = BrazilANACollector(client=MagicMock(), legacy_client=MagicMock())
        with pytest.raises(ValueError):
            collector.fetch_raw(station_ids="15400000", mode="bogus")

    def test_requires_no_credentials(self):
        """Historical mode must not touch the OAuth-credentialed client at all."""
        mock_legacy = MagicMock()
        mock_legacy.get_text.return_value = FAKE_HISTORICAL_XML_SINGLE
        mock_client = MagicMock()  # would raise/track if touched
        collector = BrazilANACollector(client=mock_client, legacy_client=mock_legacy)  # no identificador/senha

        collector.fetch_raw(station_ids="15400000", mode="historical", variables=("water_level",))

        mock_client.get_json.assert_not_called()

    def test_sends_correct_params_per_variable(self):
        mock_legacy = MagicMock()
        mock_legacy.get_text.return_value = FAKE_HISTORICAL_XML_SINGLE
        collector = BrazilANACollector(client=MagicMock(), legacy_client=mock_legacy)

        collector.fetch_raw(
            station_ids="15400000",
            mode="historical",
            variables=("water_level",),
            start_date="2020-01-01",
            end_date="2020-12-31",
        )

        mock_legacy.get_text.assert_called_once()
        call = mock_legacy.get_text.call_args
        assert "HidroSerieHistorica" in call.args[0]
        assert call.kwargs["params"] == {
            "codEstacao": "15400000",
            "tipoDados": "1",  # water_level
            "nivelConsistencia": "",
            "dataInicio": "2020-01-01",
            "dataFim": "2020-12-31",
        }

    def test_tags_rows_with_mode_station_and_variable(self):
        mock_legacy = MagicMock()
        mock_legacy.get_text.return_value = FAKE_HISTORICAL_XML_SINGLE
        collector = BrazilANACollector(client=MagicMock(), legacy_client=mock_legacy)

        rows = collector.fetch_raw(station_ids="15400000", mode="historical", variables=("water_level",))

        assert len(rows) == 1
        assert rows[0]["_mode"] == "historical"
        assert rows[0]["_station_id"] == "15400000"
        assert rows[0]["_variable"] == "water_level"

    def test_isolates_per_station_variable_failures(self):
        mock_legacy = MagicMock()
        mock_legacy.get_text.side_effect = [RuntimeError("boom"), FAKE_HISTORICAL_XML_SINGLE]
        collector = BrazilANACollector(client=MagicMock(), legacy_client=mock_legacy)

        rows = collector.fetch_raw(
            station_ids=["bad_station", "15400000"], mode="historical", variables=("water_level",)
        )
        assert len(rows) == 1  # only the working station's row

    def test_raises_when_every_request_fails(self):
        """A dead endpoint (network errors on every request) must surface as an error."""
        mock_legacy = MagicMock()
        mock_legacy.get_text.side_effect = [RuntimeError("boom 1"), RuntimeError("boom 2")]
        collector = BrazilANACollector(client=MagicMock(), legacy_client=mock_legacy)

        with pytest.raises(RuntimeError, match="all 2 request"):
            collector.fetch_raw(station_ids=["bad_1", "bad_2"], mode="historical", variables=("water_level",))

    def test_raises_when_every_response_is_unparseable(self):
        """A shape drift (endpoint now returns an HTML error page, say) must not look like 'no data'."""
        mock_legacy = MagicMock()
        mock_legacy.get_text.return_value = "<html>not the expected XML</html"  # malformed
        collector = BrazilANACollector(client=MagicMock(), legacy_client=mock_legacy)

        with pytest.raises(RuntimeError, match="all 1 request"):
            collector.fetch_raw(station_ids=["15400000"], mode="historical", variables=("water_level",))

    def test_genuinely_empty_document_does_not_raise(self):
        """A station with no historical records ('<DocumentElement/>') is legitimate, not a failure."""
        mock_legacy = MagicMock()
        mock_legacy.get_text.return_value = "<DocumentElement></DocumentElement>"
        collector = BrazilANACollector(client=MagicMock(), legacy_client=mock_legacy)

        rows = collector.fetch_raw(station_ids=["15400000"], mode="historical", variables=("water_level",))
        assert rows == []

    def test_partial_failure_does_not_raise(self):
        """Only some requests failing (mix of good stations/data and bad ones) must not raise."""
        mock_legacy = MagicMock()
        mock_legacy.get_text.side_effect = [RuntimeError("boom"), FAKE_HISTORICAL_XML_SINGLE]
        collector = BrazilANACollector(client=MagicMock(), legacy_client=mock_legacy)

        rows = collector.fetch_raw(
            station_ids=["bad_station", "15400000"], mode="historical", variables=("water_level",)
        )
        assert len(rows) == 1


class TestBrazilANANormaliseHistorical:
    def _tagged_rows(self, xml_text: str, variable: str = "discharge"):
        rows = _parse_historical_xml(xml_text)
        for row in rows:
            row["_mode"] = "historical"
            row["_station_id"] = "15400000"
            row["_variable"] = variable
        return rows

    def test_prefers_consistido_over_bruto_for_same_month(self):
        collector = BrazilANACollector(client=MagicMock())
        rows = self._tagged_rows(FAKE_HISTORICAL_XML_TWO_LEVELS)

        readings = collector.normalise(rows)
        streamflow = [r for r in readings if isinstance(r, StreamflowReading)]

        # Two distinct days (01, 02) survive from the Consistido row, not Bruto.
        by_day = {r.reading_datetime.day: r.discharge_cms for r in streamflow if r.reading_datetime.month == 1}
        assert by_day[1] == 102.5  # Consistido value, not Bruto's 100.0
        assert by_day[2] == 103.5

    def test_drops_relisted_month_rows(self):
        collector = BrazilANACollector(client=MagicMock())
        rows = self._tagged_rows(FAKE_HISTORICAL_XML_TWO_LEVELS)

        readings = collector.normalise(rows)
        # The day!=01 relisted row (value -1.0) must never appear.
        assert all(r.discharge_cms != -1.0 for r in readings if isinstance(r, StreamflowReading))

    def test_skips_calendar_invalid_days(self):
        """Vazao31 in a 31-day-valid month is fine, but this fixture only has a Consistido
        row's Vazao31=999.0 for January (31 days) - valid. Swap to February to test invalid day."""
        collector = BrazilANACollector(client=MagicMock())
        xml = """<?xml version="1.0"?>
        <DocumentElement>
          <SerieHistorica>
            <EstacaoCodigo>15400000</EstacaoCodigo>
            <NivelConsistencia>2</NivelConsistencia>
            <Data>2021-02-01T00:00:00</Data>
            <Vazao01>10.0</Vazao01>
            <Vazao30>20.0</Vazao30>
          </SerieHistorica>
        </DocumentElement>"""
        rows = self._tagged_rows(xml)
        readings = collector.normalise(rows)
        streamflow = [r for r in readings if isinstance(r, StreamflowReading)]
        # Feb 2021 has 28 days - day 30 must be silently skipped, not raise.
        assert len(streamflow) == 1
        assert streamflow[0].reading_datetime.day == 1

    def test_water_level_converts_cm_to_m(self):
        collector = BrazilANACollector(client=MagicMock())
        rows = self._tagged_rows(FAKE_HISTORICAL_XML_SINGLE, variable="water_level")

        readings = collector.normalise(rows)
        levels = [r for r in readings if isinstance(r, WaterLevelReading)]
        assert len(levels) == 1
        assert levels[0].water_level == pytest.approx(5.0)  # 500 cm -> 5.0 m
        assert levels[0].unit == "m"

    def test_precipitation_maps_to_climate_reading(self):
        collector = BrazilANACollector(client=MagicMock())
        xml = """<?xml version="1.0"?>
        <DocumentElement>
          <SerieHistorica>
            <EstacaoCodigo>15400000</EstacaoCodigo>
            <NivelConsistencia>2</NivelConsistencia>
            <Data>2020-03-01T00:00:00</Data>
            <Chuva01>12.4</Chuva01>
          </SerieHistorica>
        </DocumentElement>"""
        rows = self._tagged_rows(xml, variable="precipitation")

        readings = collector.normalise(rows)
        rainfall = [r for r in readings if isinstance(r, ClimateReading)]
        assert len(rainfall) == 1
        assert rainfall[0].parameter == "rainfall_mm"
        assert rainfall[0].value == 12.4

    def test_remark_reflects_qa_level(self):
        collector = BrazilANACollector(client=MagicMock())
        rows = self._tagged_rows(FAKE_HISTORICAL_XML_SINGLE, variable="water_level")

        readings = collector.normalise(rows)
        levels = [r for r in readings if isinstance(r, WaterLevelReading)]
        assert "Bruto" in levels[0].remark

    def test_mixed_telemetric_and_historical_batch_dispatches_correctly(self):
        mock_client = MagicMock()
        mock_client.get_json.side_effect = [FAKE_STATIONS_PAGE, {"features": []}]
        collector = BrazilANACollector(client=mock_client)

        telemetric_row = dict(FAKE_SERIES_RESPONSE["items"][0])
        telemetric_row["_mode"] = "telemetric"
        historical_rows = self._tagged_rows(FAKE_HISTORICAL_XML_SINGLE, variable="water_level")

        readings = collector.normalise([telemetric_row, *historical_rows])

        assert any(isinstance(r, StreamflowReading) for r in readings)  # from telemetric row
        assert any(isinstance(r, WaterLevelReading) and r.water_level == 5.0 for r in readings)  # from historical row

    def test_skips_rows_missing_month(self):
        collector = BrazilANACollector(client=MagicMock())
        rows = [{"_mode": "historical", "_station_id": "1", "_variable": "discharge", "Vazao01": "5.0"}]
        assert collector.normalise(rows) == []
