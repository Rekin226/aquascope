"""The issued-forecast archive and the status snapshot (#517): which gauges, what is written, what is published."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pq = pytest.importorskip("pyarrow.parquet", reason="the archive extra")

from aquascope.archive import forecasts as fa  # noqa: E402

TODAY = date(2026, 10, 8)
WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "forecast-archive.yml"


def _series(end: str = "2026-10-08", start: str = "1990-01-01") -> pd.Series:
    idx = pd.date_range(start, end, freq="D")
    return pd.Series(50 + 30 * np.sin(2 * np.pi * idx.dayofyear.to_numpy() / 365.25), index=idx)


MANIFEST = {"sources": {
    "usgs/discharge": {"source": "usgs", "variable": "discharge", "stations": {
        "USGS-1": {"first": "1950-01-01", "last": "2026-10-05"},
        "USGS-2": {"first": "2020-01-01", "last": "2026-10-05"},   # too short
        "USGS-3": {"first": "1950-01-01", "last": "2026-08-01"},   # stale mirror
    }},
    "usgs/water_level": {"source": "usgs", "variable": "water_level", "stations": {
        "USGS-9": {"first": "1950-01-01", "last": "2026-10-05"}}},
    "uk_ea/discharge": {"source": "uk_ea", "variable": "discharge", "stations": {
        "a": {"first": "1960-01-01", "last": "2026-10-03"}, "b": {"first": "1900-01-01", "last": "2026-10-03"}}},
}}


def test_candidates_are_long_recent_discharge_mirrors():
    got = fa.candidates(MANIFEST, today=TODAY)
    assert [(g["source"], g["station_id"]) for g in got] == [("uk_ea", "b"), ("uk_ea", "a"), ("usgs", "USGS-1")]


def test_round_robin_keeps_every_source_under_a_cap():
    rows = [{"source": s, "station_id": str(i)} for s in ("a", "b") for i in range(5)] + [{"source": "c",
                                                                                         "station_id": "0"}]
    kept, dropped = fa.round_robin(rows, 4)
    assert [r["source"] for r in kept] == ["a", "b", "c", "a"] and len(dropped) == 7
    assert fa.round_robin(rows, None) == (rows, [])


def test_status_rows_only_for_fresh_records():
    rows, meta = fa.status_rows({("usgs", "1"): _series(), ("usgs", "2"): _series(end="2026-09-20"),
                                 ("uk_ea", "3"): None}, today=TODAY)
    assert [(r["source"], r["station_id"]) for r in rows] == [("usgs", "1")]
    assert set(rows[0]) == {"source", "station_id", "value_date", "value", "percentile", "class", "n_years"}
    assert meta["sources"] == ["usgs"] and meta["n"] == 1 and sum(meta["counts"].values()) == 1
    assert meta["date"] == "2026-10-08" and len(meta["classes"]) == 5


def test_snap_reaches_snaps_once_and_retries_a_miss_later():
    calls = []

    def snap(lat, lon):
        calls.append((lat, lon))
        return {"snapped": lat > 0, "river_id": 760021611 if lat > 0 else None, "distance_m": 12.0}

    gauges = [{"source": "usgs", "station_id": "1"}, {"source": "usgs", "station_id": "2"},
              {"source": "usgs", "station_id": "3"}]
    known = {("usgs", "3"): {"river_id": None, "snapped_at": "2026-10-01"}}
    out = fa.snap_reaches(gauges, known, {("usgs", "1"): (40.0, -75.0), ("usgs", "2"): (-1.0, 1.0),
                                          ("usgs", "3"): (40.0, -75.0)}, today=TODAY, snapper=snap)
    assert out[("usgs", "1")]["river_id"] == 760021611 and out[("usgs", "2")]["river_id"] is None
    assert out[("usgs", "3")]["river_id"] is None and len(calls) == 2  # a miss a week ago is not retried yet
    fa.snap_reaches(gauges, out, {}, today=TODAY, snapper=snap)
    assert len(calls) == 2


def _fake_forecast(**kw):
    return {"river_id": kw["river_id"],
            "geoglows": {"date": ["2026-10-07", "2026-10-08"], "mean": [1.0, 2.0], "generated": "2026-10-08T08:00Z",
                         "initialized": "2026-10-07T00:00Z"},
            "reach_check": {"ratio": 0.9, "matches": True},
            "glofas": {"date": ["2026-10-08"], "mean": [3.0]},
            "correction": {"forecast": {"date": ["2026-10-07", "2026-10-08"], "mean": [0.5, 1.0]},
                           "skill": {"by": "month", "raw": {"kge": 0.2}, "corrected": {"kge": 0.6}}},
            "gauge_thresholds": {"return_periods": [2, 5, 10, 25, 50, 100], "q": [9.0, 12.0, 15.0, 18.0, None, 22.0]}}


def test_issue_one_writes_raw_and_corrected_rows_and_picks_the_glofas_cell_once(monkeypatch):
    from aquascope import explore, nownext

    seen = {}
    monkeypatch.setattr(explore, "snap_glofas_cell", lambda lat, lon, ref, window=2: {
        "lat": 40.05, "lon": -75.0, "flow_magnitude_matches": True})

    def fake(**kw):
        seen.update(kw)
        return _fake_forecast(**kw)

    monkeypatch.setattr(nownext, "forecast", fake)
    reach = {"source": "usgs", "station_id": "1", "river_id": 760021611, "lat": 40.0, "lon": -75.0,
             "glofas_lat": None, "glofas_lon": None}
    rows, new = fa.issue_one({"source": "usgs", "station_id": "1"}, _series(), reach, today=TODAY)
    assert seen["glofas_at"] == (40.05, -75.0) and new["glofas_lat"] == 40.05
    geo = [r for r in rows if r["model"] == "geoglows"]
    assert [r["lead_day"] for r in geo] == [0, 1] and [r["mean_c"] for r in geo] == [0.5, 1.0]  # from the run's start
    assert {r["init_date"] for r in geo} == {"2026-10-07"} and geo[0]["reach_mean_ratio"] == 0.9
    glo = [r for r in rows if r["model"] == "glofas"]
    assert glo[0]["mean"] == 3.0 and glo[0]["mean_c"] is None and glo[0]["kge_corrected"] == 0.6
    assert glo[0]["init_date"] is None and glo[0]["lead_day"] == 0  # Open-Meteo does not say when GloFAS ran
    # the gauge's own flows travel with every row, for the watch digest and the feeds (#521)
    got = (geo[0]["gauge_q2"], geo[0]["gauge_q10"], geo[0]["gauge_q50"], glo[0]["gauge_q100"])
    assert got == (9.0, 15.0, None, 22.0)


def test_run_writes_the_snapshot_the_issue_and_the_manifest(monkeypatch, tmp_path):
    from aquascope import nownext

    monkeypatch.setattr(fa, "read_published_json", lambda path, repo_id=fa.DEFAULT_REPO: (
        {"issues": [{"date": "2026-10-07", "n_gauges": 1}]} if path.endswith("manifest.json") else {}))
    monkeypatch.setattr(fa, "read_published_rows", lambda path, repo_id=fa.DEFAULT_REPO: [])
    monkeypatch.setattr(fa, "gauge_record", lambda source, sid, today: (
        (_series(), "") if sid != "a" else (_series(end="2026-09-01"), "")))
    monkeypatch.setattr("aquascope.rivers.snap_to_river",
                        lambda lat, lon, **kw: {"snapped": True, "river_id": 760021611, "distance_m": 5.0})
    monkeypatch.setattr("aquascope.explore.snap_glofas_cell", lambda *a, **k: {})
    monkeypatch.setattr(nownext, "forecast", _fake_forecast)
    catalog = [{"source": "usgs", "station_id": "USGS-1", "latitude": 40.0, "longitude": -75.0},
               {"source": "uk_ea", "station_id": "a", "latitude": 51.0, "longitude": 0.0},
               {"source": "uk_ea", "station_id": "b", "latitude": 51.5, "longitude": 0.1}]
    info = fa.run(tmp_path, manifest=MANIFEST, catalog=catalog, today=TODAY, max_gauges=1, workers=1)
    root = tmp_path / "forecasts"
    status = pq.read_table(root / "status" / "latest.parquet").to_pylist()
    assert {r["station_id"] for r in status} == {"USGS-1", "b"}  # "a" has no fresh value
    meta = json.loads((root / "status" / "latest.json").read_text())
    assert meta["sources"] == ["uk_ea", "usgs"] and meta["n"] == 2
    issued = pq.read_table(root / "issued" / "2026-10-08.parquet").to_pylist()
    assert {r["station_id"] for r in issued} == {"b"} and len(issued) == 3
    reaches = pq.read_table(root / "reaches.parquet").to_pylist()
    assert {r["station_id"] for r in reaches} == {"USGS-1", "b"}
    man = json.loads((root / "manifest.json").read_text())
    assert [i["date"] for i in man["issues"]] == ["2026-10-07", "2026-10-08"]
    assert man["issues"][-1]["dropped"] == {"daily cap": 1, "no fresh value": 1}
    assert man["status"]["file"] == "forecasts/status/latest.parquet" and "CC BY 4.0" in man["about"]
    assert info["n_gauges"] == 1 and info["status_gauges"] == 2


def test_run_stops_reading_records_at_half_the_time_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(fa, "read_published_json", lambda path, repo_id=fa.DEFAULT_REPO: {})
    monkeypatch.setattr(fa, "read_published_rows", lambda path, repo_id=fa.DEFAULT_REPO: [])
    monkeypatch.setattr(fa, "gauge_record", lambda *a, **k: pytest.fail("no record past the budget"))
    info = fa.run(tmp_path, manifest=MANIFEST, catalog=[], today=TODAY, workers=1, time_budget_s=0)
    assert info["dropped"] == {"time budget (records)": 3} and info["status_gauges"] == 0


def test_publish_uploads_only_the_forecasts_folder(monkeypatch, tmp_path):
    root = tmp_path / "forecasts"
    (root / "status").mkdir(parents=True)
    (root / "status" / "latest.json").write_text("{}")
    (root / "manifest.json").write_text(json.dumps({"issues": [{"date": "2026-10-08"}]}))
    (tmp_path / "stations.parquet").write_text("not ours")
    seen = {}

    def fake_publish(folder, repo_id, token=None, commit_message=None):
        seen["files"] = sorted(str(p.relative_to(folder)) for p in Path(folder).rglob("*") if p.is_file())
        seen["message"] = commit_message
        return "https://hf.co/commit/1"

    monkeypatch.setattr("aquascope.archive.publish.publish_folder", fake_publish)
    assert fa.publish(tmp_path) == "https://hf.co/commit/1"
    assert seen["files"] == ["forecasts/manifest.json", "forecasts/status/latest.json"]
    assert seen["message"] == "forecasts: issue 2026-10-08"


def test_the_workflow_runs_daily_and_a_smoke_run_never_publishes():
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    on = wf.get("on", wf.get(True))
    assert on["schedule"][0]["cron"].endswith("* * *")
    inputs = on["workflow_dispatch"]["inputs"]
    assert inputs["publish"]["default"] is True and inputs["max_items"]["default"] == ""
    steps = wf["jobs"]["forecast"]["steps"]
    publish = next(s for s in steps if "forecasts publish" in s.get("run", ""))
    assert "inputs.max_items == ''" in publish["if"] and "github.event_name == 'schedule'" in publish["if"]
    assert wf["jobs"]["forecast"]["env"]["HF_TOKEN"] == "${{ secrets.HF_TOKEN }}"
    runs = " ".join(s.get("run", "") for s in steps)
    assert "harvest" not in runs and "publish_folder" not in runs  # it never touches the catalogue
