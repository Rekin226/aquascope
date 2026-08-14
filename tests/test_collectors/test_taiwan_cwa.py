"""Tests for the Taiwan CWA (CODIS) climate collector (#177).

Fixtures are trimmed copies of real CODIS responses captured 2026-08-13
(station 466920, Taipei, 2024-03-01..03)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aquascope.collectors.taiwan_cwa import PARAMETER_MAP, TaiwanCWACollector
from aquascope.schemas.water_data import ClimateReading, DataSource

FIXTURES = Path(__file__).parent / "fixtures"
STATION_DATA = json.loads((FIXTURES / "cwa_station_data.json").read_text())
STATION_LIST = json.loads((FIXTURES / "cwa_station_list.json").read_text())


def _collector() -> TaiwanCWACollector:
    col = TaiwanCWACollector()
    col.client = MagicMock()
    return col


def _post_json(url, form=None, **kw):
    return STATION_LIST if url == "station_list" else STATION_DATA


class TestFetchRaw:
    def test_windows_are_calendar_years(self):
        col = _collector()
        col.client.post_json.side_effect = _post_json
        col.fetch_raw(station_ids=["466920"], start="2019-06-01", end="2021-02-01")
        forms = [c.kwargs.get("form") or c.args[1] for c in col.client.post_json.call_args_list]
        windows = [(f["start"][:10], f["end"][:10]) for f in forms]
        assert windows == [
            ("2019-06-01", "2019-12-31"),
            ("2020-01-01", "2020-12-31"),
            ("2021-01-01", "2021-02-01"),
        ]

    def test_single_string_station_accepted(self):
        col = _collector()
        col.client.post_json.side_effect = _post_json
        raw = col.fetch_raw(station_ids="466920", start="2024-03-01", end="2024-03-03")
        assert len(raw) == 3 and raw[0]["station_id"] == "466920"

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError):
            _collector().fetch_raw(start="2024-05-01", end="2024-01-01")

    def test_empty_answer_yields_no_rows(self):
        col = _collector()
        col.client.post_json.return_value = None
        assert col.fetch_raw(start="2024-03-01", end="2024-03-02") == []


class TestNormalise:
    def _records(self) -> list[ClimateReading]:
        col = _collector()
        col.client.post_json.side_effect = _post_json
        raw = col.fetch_raw(station_ids=["466920"], start="2024-03-01", end="2024-03-03")
        return list(col.normalise(raw))

    def test_maps_real_payload(self):
        recs = self._records()
        assert recs and all(isinstance(r, ClimateReading) for r in recs)
        assert all(r.source == DataSource.TAIWAN_CWA for r in recs)
        by_param = {}
        for r in recs:
            by_param.setdefault(r.parameter, []).append(r)
        # Real values from the 2024-03-01 fixture day
        rain = [r for r in by_param["rainfall_mm"] if r.sample_datetime.day == 1]
        assert rain[0].value == 11.5 and rain[0].unit == "mm"
        tmean = [r for r in by_param["temperature_mean_c"] if r.sample_datetime.day == 1]
        assert tmean[0].value == 12.3

    def test_station_metadata_joined(self):
        recs = self._records()
        r = recs[0]
        assert r.station_name == "臺北"
        assert r.location is not None and abs(r.location.latitude - 25.037658) < 1e-6
        assert r.altitude_m is not None

    def test_negative_accumulations_dropped(self):
        recs = self._records()
        for r in recs:
            if r.parameter in ("rainfall_mm", "solar_radiation_mj_m2", "pan_evaporation_mm"):
                assert r.value >= 0
        # The fixture's 2024-03-01 pan evaporation is -0.9 and must be absent
        pan_day1 = [
            r for r in recs
            if r.parameter == "pan_evaporation_mm" and r.sample_datetime.day == 1
        ]
        assert pan_day1 == []

    def test_station_list_failure_degrades_gracefully(self):
        col = _collector()

        def post_json(url, form=None, **kw):
            if url == "station_list":
                raise RuntimeError("All 3 attempts failed")
            return STATION_DATA

        col.client.post_json.side_effect = post_json
        raw = col.fetch_raw(station_ids=["466920"], start="2024-03-01", end="2024-03-03")
        recs = col.normalise(raw)
        assert recs and recs[0].location is None and recs[0].station_name is None

    def test_parameter_map_is_curated(self):
        # Guard against silent growth without unit thought
        assert set(PARAMETER_MAP) == {
            "rainfall_mm", "temperature_mean_c", "temperature_max_c",
            "temperature_min_c", "relative_humidity_pct",
            "solar_radiation_mj_m2", "wind_speed_ms", "pan_evaporation_mm",
        }
