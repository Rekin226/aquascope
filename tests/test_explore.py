"""``aquascope.explore`` (#189): the (source, station) -> answer entry point behind the Explorer and MCP.

Collectors are replaced by fakes; the JSON contract the browser and the MCP
tools rely on is checked in CPython, exactly the code Pyodide runs.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from aquascope import explore as analysis
from aquascope.archive import catalog as catalog_mod
from aquascope.schemas.water_data import DataSource, StreamflowReading, WaterLevelReading


def _daily_flow(years: int = 30, seed: int = 7) -> pd.Series:
    idx = pd.date_range("1990-01-01", periods=int(365.25 * years), freq="D")
    rng = np.random.default_rng(seed)
    base = 50 + 30 * np.sin(np.arange(len(idx)) / 58.1)
    return pd.Series(np.exp(rng.normal(0, 0.5, len(idx))) * base, index=idx)


def test_analyze_series_full_contract():
    s = _daily_flow(30)
    out = analysis.analyze_series(s, "discharge", "m3/s")
    json.dumps(out)  # must be JSON-safe
    assert out["variable"] == "discharge" and out["unit"] == "m3/s"
    assert out["n"] == len(s) and out["years"] == pytest.approx(30, abs=0.1)
    assert out["stats"]["max"] >= out["stats"]["mean"] >= out["stats"]["min"]
    assert len(out["series"]["t"]) == len(out["series"]["v"]) == len(s)
    assert out["series"]["t"][0] == "1990-01-01"
    assert 28 <= len(out["annual_max"]["year"]) <= 30
    assert out["fdc"]["q95"] < out["fdc"]["q50"] < out["fdc"]["q10"]
    ffa = out["ffa"]
    assert ffa["return_periods"] == [2, 5, 10, 25, 50, 100]
    g, lp3 = ffa["fits"]["gev_lmoments"], ffa["fits"]["lp3"]
    assert g["q"] == sorted(g["q"])  # return levels increase with T
    assert lp3["q"] == sorted(lp3["q"])
    assert all(lo <= q <= hi for q, (lo, hi) in zip(lp3["q"], lp3["ci"]))
    names = {m["name"] for m in out["methods"]}
    assert names >= {"Flow-duration curve", "GEV fitted by L-moments", "Log-Pearson III (Bulletin 17C style)"}
    assert out["trend"]["n_years"] >= 28 and out["trend"]["trend"] in ("increasing", "decreasing", "no trend")


def test_short_record_has_no_ffa_but_says_why():
    s = _daily_flow(3)
    out = analysis.analyze_series(s, "discharge", "m3/s")
    assert "ffa" not in out
    assert any("at least 10 complete years" in n for n in out["notes"])
    assert "fdc" in out  # FDC still fine on 3 years


def test_water_level_skips_flow_only_analytics():
    s = _daily_flow(20)
    out = analysis.analyze_series(s, "water_level", "m")
    assert "fdc" not in out and "ffa" not in out
    assert "trend" in out


def test_partial_years_are_dropped_from_annual_maxima():
    s = _daily_flow(12)
    s = s[s.index >= "1990-11-01"]  # first year has only two months
    am = analysis._annual_max(s)
    assert 1990 not in list(am.index.year)


def test_records_to_series_handles_models():
    now = datetime(2026, 1, 1)
    recs = [
        StreamflowReading(source=DataSource.USGS, station_id="x", reading_datetime=now + timedelta(days=i),
                          discharge_cms=float(i + 1), source_type="in_situ")
        for i in range(3)
    ]
    s, var, unit = analysis._records_to_series(recs)
    assert var == "discharge" and unit == "m3/s" and list(s.values) == [1.0, 2.0, 3.0]
    lv = [WaterLevelReading(source=DataSource.UK_EA, station_id="y", reading_datetime=now, water_level=2.5)]
    s2, var2, unit2 = analysis._records_to_series(lv)
    assert var2 == "water_level" and unit2 == "m" and s2.iloc[0] == 2.5
    assert analysis._records_to_series([]) == (None, "", "")


class _FakeUSGS:
    def __init__(self, series):
        self._s = series
        self.calls = []

    def collect(self, **kw):
        self.calls.append(kw)
        if kw.get("parameter") != "00060":
            return []
        return [
            StreamflowReading(source=DataSource.USGS, station_id="USGS-1", reading_datetime=t.to_pydatetime(),
                              discharge_cms=float(v), source_type="in_situ")
            for t, v in self._s.items()
        ]


def test_analyze_station_usgs_end_to_end_with_fake_collector():
    fake = _FakeUSGS(_daily_flow(15))
    with patch.object(analysis, "build_collector", return_value=fake):
        store = {}
        out = analysis.analyze_station("usgs", "USGS-01646500", years=15, store=store)
    assert fake.calls[0]["station_id"] == "USGS-01646500"  # the collector maps the id for NWIS / OGC itself
    assert out["source"] == "usgs" and out["agency"].startswith("U.S. Geological")
    assert out["license"] == "US-PD" and "public domain" in out["attribution"]
    assert out["n"] > 5000 and "ffa" in out
    assert isinstance(store["series"], pd.Series)
    csv = analysis.to_csv(out)
    assert csv.splitlines()[0] == "date,discharge_m3_per_s"
    assert csv.count("\n") == out["n"] + 1
    ci = analysis.flood_ci(store["series"])
    assert len(ci["q"]) == 6 and all(lo <= q <= hi for q, (lo, hi) in zip(ci["q"], ci["ci"]))
    assert ci["n_bootstrap"] == 1000
    assert isinstance(ci["n_bootstrap_discarded"], int) and ci["n_bootstrap_discarded"] >= 0
    assert ci["method"]["name"].startswith("GEV")


def test_analyze_station_reports_empty_source():
    class Empty:
        def collect(self, **kw):
            return []

    with patch.object(analysis, "build_collector", return_value=Empty()):
        out = analysis.analyze_station("usgs", "USGS-0", years=5)
    assert out["n"] == 0 and "no observations" in out["error"]


def test_uk_ea_measure_preference():
    class FakeClient:
        def get_json(self, path, params=None):
            return {"items": [{"measures": [
                {"@id": "http://x/measures/S-flow-min-86400-m3s-qualified", "parameter": "flow", "period": 86400,
                 "valueStatistic": {"@id": "http://x/def/core/minimum"}},
                {"@id": "http://x/measures/S-flow-m-86400-m3s-qualified", "parameter": "flow", "period": 86400,
                 "valueStatistic": {"@id": "http://x/def/core/mean"}},
                {"@id": "http://x/measures/S-level-i-900-m-qualified", "parameter": "level", "period": 900,
                 "valueStatistic": {"@id": "http://x/def/core/instantaneous"}},
            ]}]}

    class Fake:
        client = FakeClient()

    assert analysis._uk_ea_pick_measure(Fake(), "S") == ("S-flow-m-86400-m3s-qualified", "discharge")
    assert analysis._uk_ea_pick_measure(Fake(), "S", variable="water_level") == ("S-level-i-900-m-qualified",
                                                                                 "water_level")
    assert analysis._uk_ea_pick_measure(Fake(), "S", variable="precipitation") == (None, None)


def test_uk_ea_measure_variables_level_rain_and_boreholes():
    """Real EA shapes: level sites publish daily max/min + 15-min; rain gauges daily totals;
    boreholes 'level' in mAOD with manual dips (period None) and sometimes a logger."""
    measures = [
        {"@id": "http://x/measures/L-level-max-86400-m-qualified", "parameter": "level", "period": 86400,
         "valueStatistic": {"@id": "http://x/def/core/maximum"}, "unitName": "m"},
        {"@id": "http://x/measures/L-level-min-86400-m-qualified", "parameter": "level", "period": 86400,
         "valueStatistic": {"@id": "http://x/def/core/minimum"}, "unitName": "m"},
        {"@id": "http://x/measures/L-level-i-900-m-qualified", "parameter": "level", "period": 900,
         "valueStatistic": {"@id": "http://x/def/core/instantaneous"}, "unitName": "m"},
        {"@id": "http://x/measures/L-rainfall-t-86400-mm-qualified", "parameter": "rainfall", "period": 86400,
         "valueStatistic": {"@id": "http://x/def/core/total"}, "unitName": "mm"},
        {"@id": "http://x/measures/L-gw-dipped-i-mAOD-qualified", "parameter": "level", "period": None,
         "valueStatistic": {"@id": "http://x/def/core/instantaneous"}, "unitName": "mAOD"},
        {"@id": "http://x/measures/L-gw-logged-i-900-mAOD-qualified", "parameter": "level", "period": 900,
         "valueStatistic": {"@id": "http://x/def/core/instantaneous"}, "unitName": "mAOD"},
    ]

    class FakeClient:
        def get_json(self, path, params=None):
            return {"items": [{"measures": measures}]}

    class Fake:
        client = FakeClient()

    pick = analysis._uk_ea_pick_measure
    assert pick(Fake(), "L") == ("L-level-max-86400-m-qualified", "water_level")  # no flow: level, daily max
    assert pick(Fake(), "L", variable="precipitation") == ("L-rainfall-t-86400-mm-qualified", "precipitation")
    assert pick(Fake(), "L", variable="groundwater_level") == ("L-gw-dipped-i-mAOD-qualified", "groundwater_level")
    assert pick(Fake(), "L", variable="discharge") == (None, None)


def test_fetch_series_variable_selection():
    """USGS: the requested variable maps to the parameter code and feet become metres; asking for a
    variable the station lacks yields an empty result rather than another variable."""
    calls = []

    class FakeUSGS:
        def collect(self, **kw):
            calls.append(kw["parameter"])
            if kw["parameter"] == "00065":
                return [WaterLevelReading(source=DataSource.USGS, station_id="1", reading_datetime=datetime(2020, 1, 1)
                                          + timedelta(days=i), water_level=10.0, unit="ft") for i in range(20)]
            return []

    with patch.object(analysis, "build_collector", return_value=FakeUSGS()):
        out = analysis.fetch_series("usgs", "USGS-1", years=1, prefer_archive=False)
        assert calls == ["00060", "00065"] and out["variable"] == "water_level"
        assert out["unit"] == "m" and out["series"].iloc[0] == pytest.approx(3.048)
        calls.clear()
        out = analysis.fetch_series("usgs", "USGS-1", years=1, prefer_archive=False, variable="water_level")
        assert calls == ["00065"] and out["variable"] == "water_level"
        calls.clear()
        out = analysis.fetch_series("usgs", "USGS-1", years=1, prefer_archive=False, variable="discharge")
        assert calls == ["00060"] and out["series"] is None and out["variable"] == ""


def test_trend_uses_the_records_own_sampling():
    """A 40-year monthly borehole record gets a trend; the annual maxima stay daily-coverage gated."""
    idx = pd.date_range("1985-01-15", periods=12 * 40, freq="MS")
    s = pd.Series(np.linspace(30, 32, len(idx)) + np.sin(np.arange(len(idx)) / 2), index=idx)
    out = analysis.analyze_series(s, "groundwater_level", "m")
    assert out["annual_max"]["year"] == []
    assert out["trend"]["n_years"] == 40 and out["trend"]["trend"] == "increasing"
    assert "ffa" not in out and not any("Flood" in n for n in out["notes"])


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        analysis.fetch_series("nope", "1")


# ── the record window (#270) ────────────────────────────────────────────────
#
# The Thames at Kingston is catalogued from 1883 and was served from 1986,
# because every face asked for the last 40 years while the note said "full
# period requested". The default now asks for the whole record, an explicit
# ``years`` is a cap, and the note says what was asked for and what came back.


def _today():
    return datetime.now(timezone.utc).date()


class _RecordingUKEA:
    """A uk_ea fake with one daily flow measure; records the window it was asked for."""

    def __init__(self, series):
        self._s = series
        self.calls = []
        self.client = self  # _uk_ea_pick_measure reads collector.client.get_json

    def get_json(self, path, params=None):
        return {"items": [{"measures": [
            {"@id": "http://x/measures/K-flow-m-86400-m3s-qualified", "parameter": "flow", "period": 86400,
             "valueStatistic": {"@id": "http://x/def/core/mean"}},
        ]}]}

    def collect(self, **kw):
        self.calls.append(kw)
        s = self._s
        if kw.get("min_date"):
            s = s[s.index >= kw["min_date"]]
        return [
            StreamflowReading(source=DataSource.UK_EA, station_id="K", reading_datetime=t.to_pydatetime(),
                              discharge_cms=float(v), source_type="in_situ")
            for t, v in s.items()
        ]


def test_the_default_window_is_the_full_catalog_record():
    fake = _RecordingUKEA(_daily_flow(30))  # served from 1990
    with patch.object(analysis, "build_collector", return_value=fake):
        out = analysis.fetch_series("uk_ea", "K", prefer_archive=False, period_start="1883-10-01")
    assert fake.calls[0]["min_date"] == "1883-10-01" and fake.calls[0]["max_items"] is None
    assert out["requested"] == {"start": "1883-10-01", "end": _today().isoformat(), "years": None,
                                "catalog_start": "1883-10-01"}
    assert "full record requested (from 1883-10-01" in out["note"]
    assert "the served record starts 1990-01-01" in out["note"]  # honest about what came back


def test_the_catalog_at_hand_supplies_the_start_when_the_caller_does_not(monkeypatch, tmp_path):
    monkeypatch.setenv("AQUASCOPE_CACHE_DIR", str(tmp_path))  # no cached copy on disk
    catalog_mod.set_catalog([{"source": "uk_ea", "station_id": "K", "period_start": "1883-10-01T00:00:00",
                              "period_end": None}])
    try:
        fake = _RecordingUKEA(_daily_flow(5))
        with patch.object(analysis, "build_collector", return_value=fake):
            out = analysis.fetch_series("uk_ea", "K", prefer_archive=False)
    finally:
        catalog_mod.set_catalog(None)
    assert fake.calls[0]["min_date"] == "1883-10-01" and out["requested"]["catalog_start"] == "1883-10-01"


def test_without_a_catalog_the_window_is_generous_and_the_note_says_why(monkeypatch, tmp_path):
    monkeypatch.setenv("AQUASCOPE_CACHE_DIR", str(tmp_path))
    catalog_mod.set_catalog(None)
    fake = _RecordingUKEA(_daily_flow(5))
    with patch.object(analysis, "build_collector", return_value=fake):
        out = analysis.fetch_series("uk_ea", "K", prefer_archive=False)
    asked = date.fromisoformat(fake.calls[0]["min_date"])
    assert (_today() - asked).days >= (analysis.FULL_RECORD_YEARS - 1) * 365
    assert "the catalog has no start date" in out["note"] and out["requested"]["catalog_start"] is None


def test_an_explicit_years_cap_still_limits_the_window():
    fake = _RecordingUKEA(_daily_flow(30))
    with patch.object(analysis, "build_collector", return_value=fake):
        out = analysis.fetch_series("uk_ea", "K", years=10, prefer_archive=False, period_start="1883-10-01")
    asked = date.fromisoformat(fake.calls[0]["min_date"])
    assert abs((_today() - asked).days - 10 * 365.25) < 2
    assert out["requested"]["years"] == 10 and "last 10 years requested" in out["note"]
    assert "only the last 10 years were requested" in out["note"]


def test_analyze_station_asks_usgs_for_the_whole_catalog_span():
    fake = _FakeUSGS(_daily_flow(15))
    with patch.object(analysis, "build_collector", return_value=fake), \
            patch("aquascope.archive.observations.fetch_archived_series", return_value=None):
        out = analysis.analyze_station("usgs", "USGS-1", period_start="1930-01-01")
    assert fake.calls[0]["days"] == (_today() - date(1930, 1, 1)).days
    assert out["requested"]["years"] is None and out["requested"]["start"] == "1930-01-01"
    assert "full record requested" in out["fetch_note"] and "served record starts 1990-01-01" in out["fetch_note"]
    assert out["ffa"]["n_years"] >= 13


def test_the_archive_copy_is_served_whole_by_default_and_capped_on_request():
    idx = pd.date_range(end=_today(), periods=int(365.25 * 30), freq="D")
    hit = pd.Series(np.linspace(1, 2, len(idx)), index=idx)
    with patch("aquascope.archive.observations.fetch_archived_series", return_value=hit):
        whole = analysis.fetch_series("usgs", "USGS-1", period_start="1930-01-01")
        capped = analysis.fetch_series("usgs", "USGS-1", years=5, period_start="1930-01-01")
    assert len(whole["series"]) == len(hit) and "full record requested" in whole["note"]
    assert f"the served record starts {idx[0].date().isoformat()}" in whole["note"]
    assert (_today() - capped["series"].index.min().date()).days <= 5 * 365.25 + 1
    assert "last 5 years requested" in capped["note"]


def test_cwa_is_capped_and_says_it_did_not_ask_for_the_full_record():
    class FakeCWA:
        def __init__(self):
            self.calls = []

        def collect(self, **kw):
            self.calls.append(kw)
            return []

    fake = FakeCWA()
    with patch.object(analysis, "build_collector", return_value=fake):
        out = analysis.fetch_series("taiwan_cwa", "466920", prefer_archive=False, period_start="1896-01-01")
    asked = date.fromisoformat(fake.calls[0]["start"])
    assert abs((_today() - asked).days - analysis.CWA_MAX_YEARS * 365.25) < 2
    assert "not the full record" in out["note"] and out["requested"]["years"] == analysis.CWA_MAX_YEARS


def test_a_long_record_analyses_quickly():
    """150 years of daily flow (about 55k rows) must not make the click-to-chart wait noticeable."""
    idx = pd.date_range("1876-01-01", periods=int(365.25 * 150), freq="D")
    rng = np.random.default_rng(1)
    s = pd.Series(np.exp(rng.normal(0, 0.5, len(idx))) * 50, index=idx)
    analysis.analyze_series(s[:400], "discharge", "m3/s")  # warm the lazy imports
    t0 = time.perf_counter()
    out = analysis.analyze_series(s, "discharge", "m3/s")
    assert time.perf_counter() - t0 < 5.0
    assert out["ffa"]["n_years"] == 150 and len(out["series"]["t"]) <= 25_000


def test_catalog_period_reads_the_rows_at_hand_and_never_downloads(monkeypatch, tmp_path):
    monkeypatch.setenv("AQUASCOPE_CACHE_DIR", str(tmp_path))
    catalog_mod.set_catalog(None)
    with patch("aquascope.archive.catalog._download", side_effect=AssertionError("must not download")):
        assert catalog_mod.catalog_period("uk_ea", "K") == (None, None)
        catalog_mod.set_catalog([{"source": "uk_ea", "station_id": "K", "period_start": "1883-10-01",
                                  "period_end": "2026-08-19"}])
        try:
            assert catalog_mod.catalog_period("uk_ea", "K") == ("1883-10-01", "2026-08-19")
            assert catalog_mod.catalog_period("uk_ea", "nope") == (None, None)
        finally:
            catalog_mod.set_catalog(None)


def test_catalog_period_uses_a_cached_copy_whatever_its_age(monkeypatch, tmp_path):
    monkeypatch.setenv("AQUASCOPE_CACHE_DIR", str(tmp_path))
    catalog_mod.set_catalog(None)
    gj = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "geometry": {"type": "Point", "coordinates": [-0.31, 51.41]},
        "properties": {"source": "uk_ea", "station_id": "K", "period_start": "1883-10-01", "period_end": None},
    }]}
    dest = tmp_path / "Rekin226__aquascope-gauges.geojson"
    dest.write_text(json.dumps(gj), encoding="utf-8")
    old = time.time() - 30 * 86400  # older than the download TTL: still read, never refreshed here
    os.utime(dest, (old, old))
    with patch("aquascope.archive.catalog._download", side_effect=AssertionError("must not download")):
        assert catalog_mod.catalog_period("uk_ea", "K") == ("1883-10-01", None)
