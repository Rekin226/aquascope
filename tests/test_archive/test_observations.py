"""Archive Phases 1 and 2 (#188): budgeted, incremental per-station daily observations, several variables."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from aquascope.archive import observations as obs

CATALOG = [
    {"source": "hubeau_hydrometrie", "station_id": "A1", "variables": ["discharge", "water_level"]},
    {"source": "hubeau_hydrometrie", "station_id": "A2", "variables": ["discharge", "water_level"]},
    {"source": "hubeau_hydrometrie", "station_id": "A3", "variables": ["discharge", "water_level"]},
    {"source": "uk_ea", "station_id": "B1", "variables": ["water_level"]},  # no discharge: skipped
    {"source": "pegelonline", "station_id": "P1", "variables": ["discharge"]},  # not harvestable
]


def _series(n=800, start="2020-01-01"):
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.Series(np.linspace(1, 2, n), index=idx)


def test_csv_gz_roundtrip():
    s = _series(10)
    payload = obs.series_to_csv_gz(s)
    assert gzip.decompress(payload).decode().startswith("date,value\n2020-01-01,1\n")
    back = obs.read_csv_gz(payload)
    assert len(back) == 10 and back.iloc[-1] == pytest.approx(2.0)
    # sub-daily input is averaged to daily
    sub = pd.Series([1.0, 3.0], index=pd.to_datetime(["2020-01-01 06:00", "2020-01-01 18:00"]))
    assert obs.read_csv_gz(obs.series_to_csv_gz(sub)).iloc[0] == 2.0


def test_harvest_writes_files_manifest_and_report(tmp_path):
    calls = []

    def fake_fetch(source, sid, *, years, prefer_archive, variable=None):
        calls.append((source, sid, prefer_archive, variable))
        if sid == "A2":
            return {"series": None, "variable": "", "unit": "", "note": ""}
        return {"series": _series(), "variable": "discharge", "unit": "m3/s", "note": "fake"}

    with patch("aquascope.explore.fetch_series", side_effect=fake_fetch):
        report = obs.harvest_observations(tmp_path, sources=["hubeau_hydrometrie"], catalog=CATALOG, max_stations=10)

    assert [c[2:] for c in calls] == [(False, "discharge")] * 3  # never reads the archive back; asks per variable
    h = report.sources[0]
    assert (h.attempted, h.harvested, h.empty, h.failed) == (3, 2, 1, 0)
    f = tmp_path / "obs" / "discharge" / "hubeau_hydrometrie" / "A1.csv.gz"
    assert f.exists() and obs.read_csv_gz(f.read_bytes()).shape[0] == 800
    manifest = json.loads((tmp_path / "obs" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == 2
    entry = manifest["sources"]["hubeau_hydrometrie/discharge"]
    assert entry["variable"] == "discharge" and entry["license"] == "etalab-2.0" and entry["n_stations"] == 2
    assert entry["source"] == "hubeau_hydrometrie" and entry["unit"] == "m3/s"
    assert entry["stations"]["A1"]["n"] == 800 and entry["stations"]["A1"]["file"].endswith("A1.csv.gz")
    assert entry["stations"]["A1"]["note"] == "fake"
    assert entry["stations"]["A2"]["empty"] is True
    assert (tmp_path / "obs" / "last_run.json").exists()


def test_every_harvestable_variable_gets_its_own_budget_and_cursor(tmp_path):
    catalog = [
        {"source": "uk_ea", "station_id": "F1", "variables": ["discharge", "water_level"]},
        {"source": "uk_ea", "station_id": "R1", "variables": ["precipitation"]},
        {"source": "uk_ea", "station_id": "G1", "variables": ["groundwater_level"]},
    ]

    def fake_fetch(source, sid, *, years, prefer_archive, variable=None):
        unit = {"discharge": "m3/s", "water_level": "m", "precipitation": "mm", "groundwater_level": "m"}[variable]
        return {"series": _series(50), "variable": variable, "unit": unit, "note": f"measure {variable}"}

    with patch("aquascope.explore.fetch_series", side_effect=fake_fetch) as fake:
        report = obs.harvest_observations(tmp_path, sources=["uk_ea"], catalog=catalog, max_stations=5)
    assert [(h.variable, h.harvested) for h in report.sources] == [
        ("discharge", 1), ("water_level", 1), ("precipitation", 1), ("groundwater_level", 1),
    ]
    assert fake.call_count == 4
    assert (tmp_path / "obs" / "groundwater_level" / "uk_ea" / "G1.csv.gz").exists()
    manifest = obs.load_manifest(tmp_path)
    assert set(manifest["sources"]) == {"uk_ea/discharge", "uk_ea/water_level", "uk_ea/precipitation",
                                        "uk_ea/groundwater_level"}
    # a single-variable run touches only that cursor
    with patch("aquascope.explore.fetch_series", side_effect=fake_fetch) as fake:
        r2 = obs.harvest_observations(tmp_path, sources=["uk_ea"], variable="precipitation", catalog=catalog)
    assert [h.variable for h in r2.sources] == ["precipitation"] and fake.call_count == 0  # fresh already
    with pytest.raises(ValueError, match="does not harvest"):
        obs.harvest_observations(tmp_path, sources=["hubeau_hydrometrie"], variable="precipitation", catalog=catalog)


def test_manifest_v1_is_migrated_in_place(tmp_path):
    # Derived from now(), never hard-coded: refresh_days is 30, so a fixed date silently stops being "fresh"
    # thirty days after it is written and the freshness half of this test starts failing on a clock tick.
    harvested = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(timespec="seconds")
    station = {"n": 10, "harvested_at": harvested}
    v1 = {"version": 1, "sources": {"usgs": {"variable": "discharge", "license": "US-PD",
                                             "stations": {"USGS-1": station}}}}
    (tmp_path / "obs").mkdir()
    (tmp_path / "obs" / "manifest.json").write_text(json.dumps(v1), encoding="utf-8")
    m = obs.load_manifest(tmp_path)
    assert m["version"] == 2 and "usgs" not in m["sources"]
    assert m["sources"]["usgs/discharge"]["stations"]["USGS-1"]["n"] == 10
    assert m["sources"]["usgs/discharge"]["source"] == "usgs" and m["bundles"] == {}
    # and the cursor survives: USGS-1 is fresh, not re-harvested
    catalog = [{"source": "usgs", "station_id": "USGS-1", "variables": ["discharge"]}]
    with patch("aquascope.explore.fetch_series") as fake:
        obs.harvest_observations(tmp_path, sources=["usgs"], variable="discharge", catalog=catalog)
    assert fake.call_count == 0


def test_harvest_is_incremental_and_budgeted(tmp_path):
    def fake_fetch(source, sid, *, years, prefer_archive, variable=None):
        return {"series": _series(), "variable": "discharge", "unit": "m3/s", "note": "fake"}

    with patch("aquascope.explore.fetch_series", side_effect=fake_fetch) as fake:
        r1 = obs.harvest_observations(tmp_path, sources=["hubeau_hydrometrie"], catalog=CATALOG, max_stations=2)
        assert r1.sources[0].attempted == 2 and fake.call_count == 2
        r2 = obs.harvest_observations(tmp_path, sources=["hubeau_hydrometrie"], catalog=CATALOG, max_stations=2)
        assert r2.sources[0].attempted == 1 and fake.call_count == 3  # only the remaining station
        r3 = obs.harvest_observations(tmp_path, sources=["hubeau_hydrometrie"], catalog=CATALOG, max_stations=2)
        assert r3.sources[0].attempted == 0  # everything fresh, nothing to do

    # make one station stale: it gets picked again
    manifest = obs.load_manifest(tmp_path)
    old = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat(timespec="seconds")
    manifest["sources"]["hubeau_hydrometrie/discharge"]["stations"]["A1"]["harvested_at"] = old
    obs.save_manifest(tmp_path, manifest)
    with patch("aquascope.explore.fetch_series", side_effect=fake_fetch) as fake:
        r4 = obs.harvest_observations(tmp_path, sources=["hubeau_hydrometrie"], catalog=CATALOG, max_stations=5)
    assert r4.sources[0].attempted == 1 and fake.call_args.args[1] == "A1"


def test_failures_are_recorded_not_raised(tmp_path):
    def boom(source, sid, *, years, prefer_archive, variable=None):
        raise RuntimeError("503 from agency")

    with patch("aquascope.explore.fetch_series", side_effect=boom):
        report = obs.harvest_observations(tmp_path, sources=["hubeau_hydrometrie"], catalog=CATALOG, max_stations=2)
    h = report.sources[0]
    assert h.failed == 2 and h.harvested == 0 and "503" in h.errors[0]


def test_refuses_non_harvestable_and_non_redistributable(tmp_path):
    with pytest.raises(ValueError):
        obs.harvest_observations(tmp_path, sources=["pegelonline"], catalog=CATALOG)
    with patch.dict(obs.HARVESTABLE, {"grdc": ("discharge",)}):
        with pytest.raises(ValueError, match="redistributable"):
            obs.harvest_observations(tmp_path, sources=["grdc"], catalog=CATALOG)


def test_fetch_archived_series_404_and_hit(monkeypatch):
    import urllib.error

    class Resp:
        def __init__(self, data):
            self._d = data

        def read(self):
            return self._d

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    payload = obs.series_to_csv_gz(_series(5))

    def urlopen(url, timeout=30):
        if "A404" in url:
            raise urllib.error.HTTPError(url, 404, "nf", None, None)
        return Resp(payload)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert obs.fetch_archived_series("hubeau_hydrometrie", "A404", "discharge") is None
    s = obs.fetch_archived_series("hubeau_hydrometrie", "A1", "discharge")
    assert len(s) == 5
    assert obs.archive_series_url("usgs", "USGS-1", "discharge").endswith("/obs/discharge/usgs/USGS-1.csv.gz")


def test_explore_prefers_archive(monkeypatch):
    from aquascope import explore

    hit = _series(30)
    with patch("aquascope.archive.observations.fetch_archived_series", return_value=hit) as fa:
        out = explore.fetch_series("usgs", "USGS-1", years=40)
    assert out["variable"] == "discharge" and out["unit"] == "m3/s" and "archive" in out["note"]
    assert out["series"].equals(hit[hit.index >= out["series"].index.min()])
    assert fa.call_args.args == ("usgs", "USGS-1", "discharge")

    # a station mirrored only for water level: the discharge file is a miss, the level file a hit
    def archived(source, sid, variable):
        return hit if variable == "water_level" else None

    with patch("aquascope.archive.observations.fetch_archived_series", side_effect=archived) as fa:
        out = explore.fetch_series("usgs", "USGS-2", years=40)
    assert out["variable"] == "water_level" and out["unit"] == "m"
    assert [c.args[2] for c in fa.call_args_list] == ["discharge", "water_level"]
    # asking for a variable the archive does not mirror for this source skips the archive entirely
    with patch("aquascope.archive.observations.fetch_archived_series") as fa, \
            patch("aquascope.explore.build_collector") as bc:
        bc.return_value.collect.return_value = []
        out = explore.fetch_series("hubeau_hydrometrie", "H1", years=5, variable="water_level")
    assert fa.call_count == 0 and out["series"] is None


def test_a_full_record_harvest_keeps_closed_stations_and_a_capped_one_skips_them():
    """#270: the harvest asked for 40 years and skipped anything closed before that window."""
    rows = [
        {"source": "usgs", "station_id": "OLD", "variables": ["discharge"], "period_end": "1950-12-31"},
        {"source": "usgs", "station_id": "NEW", "variables": ["discharge"], "period_end": None},
    ]
    manifest = {"sources": {}}
    picked = obs._pick_stations(rows, manifest, "usgs", "discharge", 10, 30, None)
    assert [r["station_id"] for r in picked] == ["OLD", "NEW"]
    picked = obs._pick_stations(rows, manifest, "usgs", "discharge", 10, 30, None, years=40)
    assert [r["station_id"] for r in picked] == ["NEW"]


def test_sub_daily_rainfall_folds_to_daily_totals_and_flow_to_means():
    """OpenHi telemetry is 15-minute: a day of rainfall is its sum, a day of flow its mean (#408)."""
    idx = pd.date_range("2024-01-01", periods=96 * 2, freq="15min")
    rain = pd.Series(np.full(len(idx), 0.5), index=idx)
    assert obs._daily_agg("precipitation", rain) == "sum"
    daily = obs.read_csv_gz(obs.series_to_csv_gz(rain, agg="sum"))
    assert list(daily.round(6)) == [48.0, 48.0]

    flow = pd.Series(np.full(len(idx), 3.0), index=idx)
    assert obs._daily_agg("discharge", flow) == "mean"
    daily = obs.read_csv_gz(obs.series_to_csv_gz(flow, agg=obs._daily_agg("discharge", flow)))
    assert list(daily) == [3.0, 3.0]

    # Rainfall that already arrives daily keeps the mean path: a missing day
    # stays missing rather than becoming a zero total.
    daily_rain = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-01", "2024-01-03"]))
    assert obs._daily_agg("precipitation", daily_rain) == "mean"
    assert list(obs.read_csv_gz(obs.series_to_csv_gz(daily_rain))) == [1.0, 2.0]


def test_harvest_sums_sub_daily_rainfall(tmp_path):
    idx = pd.date_range("2024-01-01", periods=96, freq="15min")
    fifteen_minute = pd.Series(np.full(96, 0.25), index=idx)

    def fetch(source, sid, **kw):
        return {"series": fifteen_minute, "variable": "precipitation", "unit": "mm", "note": "test"}

    catalog = [{"source": "greece_openhi", "station_id": "8425", "variables": ["precipitation"], "period_end": None}]
    with patch("aquascope.explore.fetch_series", side_effect=fetch):
        report = obs.harvest_observations(
            tmp_path, sources=["greece_openhi"], variable="precipitation", catalog=catalog, max_stations=1,
        )
    assert report.sources[0].harvested == 1
    stored = obs.read_csv_gz(obs.obs_path(tmp_path, "precipitation", "greece_openhi", "8425").read_bytes())
    assert list(stored) == [24.0]


def test_openhi_is_harvestable_because_the_browser_cannot_call_it():
    from aquascope.registry import SOURCES

    meta = SOURCES["greece_openhi"]
    assert "greece_openhi" in obs.HARVESTABLE
    assert meta.redistributable and not meta.browser_reachable
    assert set(obs.HARVESTABLE["greece_openhi"]) <= set(meta.variables)
    assert all(SOURCES[k].redistributable for k in obs.HARVESTABLE)


def test_csv_gz_roundtrip_with_quality():
    s = _series(5)
    quality = pd.Series(
        ["approved", "approved", "provisional", "suspect", "unknown"],
        index=s.resample("D").mean().dropna().index,
    )
    payload = obs.series_to_csv_gz(s, quality=quality)
    assert gzip.decompress(payload).decode().startswith("date,value,quality\n")
    value, back_quality = obs.read_csv_gz(payload, include_quality=True)
    assert list(back_quality) == ["approved", "approved", "provisional", "suspect", "unknown"]
    assert len(value) == 5


def test_read_csv_gz_without_quality_arg_is_unchanged():
    # every pre-existing call site must keep behaving exactly like today
    s = _series(5)
    payload = obs.series_to_csv_gz(s)  # no quality passed
    assert obs.read_csv_gz(payload).shape[0] == 5  # plain Series, as before


def test_read_csv_gz_old_file_without_quality_column_defaults_to_unknown():
    s = _series(5)
    payload = obs.series_to_csv_gz(s)  # a file written before this feature existed
    value, quality = obs.read_csv_gz(payload, include_quality=True)
    assert list(quality) == ["unknown"] * 5


def _rate_limited_error():
    from aquascope.utils.http_client import RateLimitedError

    try:
        raise RateLimitedError("https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items")
    except RateLimitedError as inner:
        try:
            raise RuntimeError("collector wrapped it") from inner
        except RuntimeError as outer:
            return outer


def test_a_rate_limited_source_stops_for_the_run_and_leaves_the_rest_for_next_time(tmp_path):
    """The 2026-09-14 and 09-21 harvests spent two hours retrying a spent USGS quota, station after station."""
    calls = []

    def fake_fetch(source, sid, *, years, prefer_archive, variable=None):
        calls.append(sid)
        if sid == "A2":
            raise _rate_limited_error()
        return {"series": _series(), "variable": "discharge", "unit": "m3/s", "note": ""}

    with patch("aquascope.explore.fetch_series", side_effect=fake_fetch):
        report = obs.harvest_observations(tmp_path, sources=["hubeau_hydrometrie"], variable="discharge",
                                          catalog=CATALOG, max_stations=10)
    h = report.sources[0]
    assert calls == ["A1", "A2"]  # A3 never asked
    assert (h.harvested, h.failed, h.stopped) == (1, 1, "rate limited")
    stations = json.loads((tmp_path / "obs" / "manifest.json").read_text())["sources"][
        "hubeau_hydrometrie/discharge"]["stations"]
    assert set(stations) == {"A1"}  # A2 and A3 stay fresh, so the next run picks them first

    with patch("aquascope.explore.fetch_series", side_effect=lambda *a, **k: {
            "series": _series(), "variable": "discharge", "unit": "m3/s", "note": ""}) as again:
        obs.harvest_observations(tmp_path, sources=["hubeau_hydrometrie"], variable="discharge",
                                 catalog=CATALOG, max_stations=2)
    assert [c.args[1] for c in again.call_args_list] == ["A2", "A3"]


def test_an_ordinary_failure_does_not_stop_the_source(tmp_path):
    def fake_fetch(source, sid, *, years, prefer_archive, variable=None):
        if sid == "A1":
            raise RuntimeError("All 3 attempts failed for https://example.test")
        return {"series": _series(), "variable": "discharge", "unit": "m3/s", "note": ""}

    with patch("aquascope.explore.fetch_series", side_effect=fake_fetch):
        report = obs.harvest_observations(tmp_path, sources=["hubeau_hydrometrie"], variable="discharge",
                                          catalog=CATALOG, max_stations=10)
    h = report.sources[0]
    assert (h.attempted, h.harvested, h.failed, h.stopped) == (3, 2, 1, "")


def test_the_time_budget_stops_a_source(tmp_path):
    with patch("aquascope.explore.fetch_series") as fake:
        fake.return_value = {"series": _series(), "variable": "discharge", "unit": "m3/s", "note": ""}
        report = obs.harvest_observations(tmp_path, sources=["hubeau_hydrometrie"], variable="discharge",
                                          catalog=CATALOG, max_stations=10, max_seconds=0)
    h = report.sources[0]
    assert fake.call_count == 0 and h.stopped == "time budget"  # a spent budget asks nothing more
