"""The issued-forecast archive and the status snapshot (#517): which gauges, what is written, what is published."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
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
    # no GloFAS record yet: raw only, and no skill borrowed from GEOGLOWS's correction
    assert glo[0]["mean"] == 3.0 and glo[0]["mean_c"] is None and glo[0]["kge_corrected"] is None
    assert geo[0]["kge_corrected"] == 0.6 and geo[0]["by"] == "month"
    assert glo[0]["init_date"] is None and glo[0]["lead_day"] == 0  # Open-Meteo does not say when GloFAS ran
    # the gauge's own flows travel with every row, for the watch digest and the feeds (#521)
    got = (geo[0]["gauge_q2"], geo[0]["gauge_q10"], geo[0]["gauge_q50"], glo[0]["gauge_q100"])
    assert got == (9.0, 15.0, None, 22.0)


def test_run_writes_the_snapshot_the_issue_and_the_manifest(monkeypatch, tmp_path):
    from aquascope import nownext

    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Response(
        200, json={"issues": [{"date": "2026-10-07", "n_gauges": 1}]}, request=httpx.Request("GET", url)))
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
    # Then the scorecard, from the files this run wrote (nothing to pair yet on the first day).
    assert man["issues"][-1]["verify"] == info["verify"] and info["verify"]["complete"] is True
    assert (tmp_path / "verify" / "manifest.json").exists()


@pytest.mark.parametrize(("status", "code", "content"), [
    (503, "", b"unavailable"),
    (None, "", b""),  # transport failure
    (404, "RevisionNotFound", b"missing revision"),
    (404, "", b"generic proxy error"),
    (200, "", b"{"),  # truncated response
    (200, "", b"{}"),
    (200, "", b'{"issues": {}}'),
    (200, "", b'{"issues": [{"n_gauges": 5}]}'),
    (200, "", b'{"issues": [{"date": "2026-02-30"}]}'),
])
def test_unreadable_issue_history_preserves_the_previous_build(monkeypatch, tmp_path, status, code, content):
    def get(url, **kwargs):
        if status is None:
            raise httpx.ConnectError("offline")
        return httpx.Response(status, headers={"X-Error-Code": code}, content=content,
                              request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(fa, "read_published_rows", lambda *a, **kw: [])
    root = tmp_path / "forecasts"
    root.mkdir()
    previous = b'{"issues":[{"date":"2026-10-07"}]}'
    (root / "manifest.json").write_bytes(previous)

    with pytest.raises((ValueError, httpx.HTTPError)):
        fa.run(tmp_path, manifest={}, catalog=[], today=TODAY, workers=1, verify_budget_s=0)

    assert (root / "manifest.json").read_bytes() == previous
    assert not (root / "status").exists()


def test_first_issue_can_bootstrap_a_missing_hub_file(monkeypatch, tmp_path):
    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Response(
        404, headers={"X-Error-Code": "EntryNotFound"}, request=httpx.Request("GET", url)))
    monkeypatch.setattr(fa, "read_published_rows", lambda *a, **kw: [])

    fa.run(tmp_path, manifest={}, catalog=[], today=TODAY, workers=1, verify_budget_s=0)

    issues = json.loads((tmp_path / "forecasts" / "manifest.json").read_text())["issues"]
    assert [entry["date"] for entry in issues] == [TODAY.isoformat()]


def test_run_can_skip_the_scorecard(monkeypatch, tmp_path):
    monkeypatch.setattr(fa, "_published_issues", lambda repo_id: [])
    monkeypatch.setattr(fa, "read_published_rows", lambda path, repo_id=fa.DEFAULT_REPO: [])
    monkeypatch.setattr(fa, "scorecard", lambda *a, **k: pytest.fail("no scorecard at a zero budget"))
    info = fa.run(tmp_path, manifest={}, catalog=[], today=TODAY, workers=1, verify_budget_s=0)
    assert "verify" not in info and not (tmp_path / "verify").exists()


def _glofas_forecast(**kw):
    out = _fake_forecast(**kw)
    out["glofas"] = {"date": ["2026-10-08", "2026-10-09"], "mean": [30.0, 60.0], "median": [28.0, 55.0],
                     "p25": [20.0, 40.0], "p75": [40.0, 80.0], "min": [10.0, 20.0], "max": [70.0, 100.0]}
    return out


def _glofas_model(gauge: pd.Series, start: str = "2016-10-07") -> pd.Series:
    """A GloFAS reanalysis at half the gauge's flow, over its last ten years."""
    return gauge[start:] / 2.0


def test_issue_one_corrects_glofas_to_the_gauge_and_leaves_geoglows_alone(monkeypatch):
    from aquascope import nownext

    monkeypatch.setattr(nownext, "forecast", _glofas_forecast)
    reach = {"source": "usgs", "station_id": "1", "river_id": 760021611, "lat": 40.0, "lon": -75.0,
             "glofas_lat": 40.05, "glofas_lon": -75.0}
    gauge = _series()
    plain, _ = fa.issue_one({"source": "usgs", "station_id": "1"}, gauge, reach, today=TODAY)
    rows, _ = fa.issue_one({"source": "usgs", "station_id": "1"}, gauge, reach, today=TODAY,
                           glofas_record=_glofas_model(gauge))
    geo = [r for r in rows if r["model"] == "geoglows"]
    assert geo == [r for r in plain if r["model"] == "geoglows"]  # GEOGLOWS rows are unchanged
    glo = [r for r in rows if r["model"] == "glofas"]
    assert [r["mean"] for r in glo] == [30.0, 60.0]
    # the model runs at half the gauge, so the same exceedance on the gauge's curve is about twice the flow
    for r in glo:
        for k in ("mean", "median", "p25", "p75"):
            assert r[f"{k}_c"] == pytest.approx(2 * r[k], rel=0.05), k
        assert r["high_res_c"] is None and r["by"] == "month"
        assert r["kge_corrected"] > r["kge_raw"]  # the GloFAS correction's own hindcast skill
    assert all(r["mean_c"] is None for r in plain if r["model"] == "glofas")


def test_a_short_glofas_record_leaves_glofas_raw(monkeypatch):
    from aquascope import nownext

    monkeypatch.setattr(nownext, "forecast", _glofas_forecast)
    reach = {"source": "usgs", "station_id": "1", "river_id": 1, "lat": 40.0, "lon": -75.0, "glofas_lat": 40.0,
             "glofas_lon": -75.0}
    rows, _ = fa.issue_one({"source": "usgs", "station_id": "1"}, _series(), reach, today=TODAY,
                           glofas_record=_glofas_model(_series(), start="2025-10-01"))
    glo = [r for r in rows if r["model"] == "glofas"]
    assert glo and all(r["mean_c"] is None and r["kge_corrected"] is None for r in glo)


def test_glofas_records_read_once_then_kept_within_a_call_budget(tmp_path):
    gauge = _series()
    cell = {"glofas_lat": 40.05, "glofas_lon": -75.0}
    known = {("usgs", "kept"): {"source": "usgs", "station_id": "kept", **cell, "glofas_record_at": "2026-09-01"},
             ("usgs", "moved"): {"source": "usgs", "station_id": "moved", **cell, "glofas_record_at": "2026-09-01"},
             ("usgs", "new"): {"source": "usgs", "station_id": "new", **cell},
             ("usgs", "late"): {"source": "usgs", "station_id": "late", **cell},
             ("usgs", "nocell"): {"source": "usgs", "station_id": "nocell", "glofas_lat": None}}
    published = {fa.glofas_record_path("usgs", "kept"): [{"date": "2020-01-01", "q": 5.0, "lat": 40.05,
                                                          "lon": -75.0}],
                 fa.glofas_record_path("usgs", "moved"): [{"date": "2020-01-01", "q": 5.0, "lat": 41.0,
                                                           "lon": -75.0}]}
    fetched, slept = [], []

    def fetcher(lat, lon, start, end):
        fetched.append((lat, lon, start, end))
        return _glofas_model(gauge, start)[:end]

    gauges = [{"source": "usgs", "station_id": s} for s in ("kept", "moved", "new", "late", "nocell")]
    cost = fa.openmeteo_calls(3653)
    out, stats = fa.glofas_records(gauges, known, tmp_path / "forecasts", today=TODAY, years=10,
                                   budget_calls=2 * cost, reader=lambda p: published.get(p, []), fetcher=fetcher,
                                   sleep=slept.append, workers=1)
    assert len(out[("usgs", "kept")]) == 1 and stats["read"] == 1  # read from the Archive, not Open-Meteo
    assert {k[1] for k in out} == {"kept", "new", "late"}  # "moved" is fetched again only within the budget
    assert stats == {"read": 1, "fetched": 2, "calls": round(2 * cost, 1), "no cell yet": 1, "left for later": 1,
                     "failed": 0}
    assert fetched[0][2:] == ("2016-10-07", "2026-10-07") and 260 < cost < 262
    assert slept == [pytest.approx(60 * cost / fa.GLOFAS_CALLS_PER_MINUTE)] * 2
    written = pq.read_table(tmp_path / fa.glofas_record_path("usgs", "new")).to_pylist()
    assert len(written) == 3653 and written[0]["lat"] == 40.05
    assert known[("usgs", "new")]["glofas_record_at"] == "2026-10-08"
    assert known[("usgs", "moved")]["glofas_record_at"] == "2026-09-01"  # not stamped: left for the next run


def test_run_issues_glofas_corrected_once_its_record_is_read(monkeypatch, tmp_path):
    from aquascope import nownext

    reach = {"source": "uk_ea", "station_id": "b", "river_id": 760021611, "distance_m": 5.0, "lat": 51.5,
             "lon": 0.1, "glofas_lat": 51.5, "glofas_lon": 0.1, "snapped_at": "2026-10-01"}
    monkeypatch.setattr(fa, "_published_issues", lambda repo_id: [])
    monkeypatch.setattr(fa, "read_published_rows", lambda path, repo_id=fa.DEFAULT_REPO: (
        [reach] if path.endswith("reaches.parquet") else []))
    monkeypatch.setattr(fa, "gauge_record", lambda source, sid, today: (_series(), ""))
    monkeypatch.setattr(fa, "_fetch_glofas", lambda lat, lon, start, end: _glofas_model(_series(), start))
    monkeypatch.setattr(fa.time, "sleep", lambda s: None)
    monkeypatch.setattr(nownext, "forecast", _glofas_forecast)
    manifest = {"sources": {"uk_ea/discharge": MANIFEST["sources"]["uk_ea/discharge"]}}
    info = fa.run(tmp_path, manifest=manifest, catalog=[], today=TODAY, workers=1)
    root = tmp_path / "forecasts"
    issued = pq.read_table(root / "issued" / "2026-10-08.parquet").to_pylist()
    glo = [r for r in issued if r["model"] == "glofas"]
    assert glo and all(r["mean_c"] == pytest.approx(2 * r["mean"], rel=0.05) for r in glo)
    assert info["glofas_corrected"] == 1 and info["glofas_record"]["fetched"] == 1
    assert (root / "glofas_record" / "uk_ea" / "b.parquet").exists()
    reaches = pq.read_table(root / "reaches.parquet").to_pylist()
    assert reaches[0]["glofas_record_at"] == "2026-10-08"


@pytest.mark.parametrize("budget", [0, -1])
def test_run_stops_reading_records_at_half_the_time_budget(monkeypatch, tmp_path, budget):
    # A coarse clock may return the same tick for the start and the first read.
    monkeypatch.setattr(fa.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(fa, "_published_issues", lambda repo_id: [])
    monkeypatch.setattr(fa, "read_published_rows", lambda path, repo_id=fa.DEFAULT_REPO: [])
    monkeypatch.setattr(fa, "gauge_record", lambda *a, **k: pytest.fail("no record past the budget"))
    info = fa.run(tmp_path, manifest=MANIFEST, catalog=[], today=TODAY, workers=1, time_budget_s=budget)
    assert info["dropped"] == {"time budget (records)": 3} and info["status_gauges"] == 0


def test_publish_rejects_a_build_missing_a_published_issue(monkeypatch, tmp_path):
    root = tmp_path / "forecasts"
    root.mkdir()
    (root / "manifest.json").write_text('{"issues": [{"date": "2026-10-08"}]}')
    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Response(
        200, json={"issues": [{"date": "2026-10-07"}, {"date": "2026-10-08"}]}, request=httpx.Request("GET", url)))
    monkeypatch.setattr("aquascope.archive.publish.publish_folder", lambda *a, **kw: pytest.fail("unexpected upload"))

    with pytest.raises(ValueError, match="2026-10-07"):
        fa.publish(tmp_path)


def test_publish_uploads_only_the_forecasts_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(fa, "_published_issues", lambda repo_id: [])
    root = tmp_path / "forecasts"
    (root / "status").mkdir(parents=True)
    (root / "status" / "latest.json").write_text("{}")
    (root / "manifest.json").write_text(json.dumps({"issues": [{"date": "2026-10-08"}]}))
    (tmp_path / "stations.parquet").write_text("not ours")
    seen = {}

    def fake_publish(folder, repo_id, token=None, commit_message=None):
        seen["files"] = sorted(p.relative_to(folder).as_posix() for p in Path(folder).rglob("*") if p.is_file())
        seen["message"] = commit_message
        return "https://hf.co/commit/1"

    monkeypatch.setattr("aquascope.archive.publish.publish_folder", fake_publish)
    assert fa.publish(tmp_path) == "https://hf.co/commit/1"
    assert seen["files"] == ["forecasts/manifest.json", "forecasts/status/latest.json"]
    assert seen["message"] == "forecasts: issue 2026-10-08"


def _scorecard_files(out, *, complete=True):
    v = out / "verify"
    v.mkdir(parents=True)
    (v / "forecast_skill.parquet").write_text("table")
    (v / "summary.json").write_text("{}")
    (v / "manifest.json").write_text(json.dumps({"complete": complete, "unread": 0 if complete else 3}))


@pytest.mark.parametrize("complete", [True, False])
def test_publish_adds_the_scorecard_only_when_complete(monkeypatch, tmp_path, complete):
    monkeypatch.setattr(fa, "_published_issues", lambda repo_id: [])
    root = tmp_path / "forecasts"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"issues": [{"date": "2026-10-08"}]}))
    _scorecard_files(tmp_path, complete=complete)
    (tmp_path / "feeds").mkdir()
    (tmp_path / "feeds" / "index.json").write_text("{}")
    seen = {}

    def fake_publish(folder, repo_id, token=None, commit_message=None):
        seen["files"] = sorted(p.relative_to(folder).as_posix() for p in Path(folder).rglob("*") if p.is_file())
        seen["message"] = commit_message
        return "https://hf.co/commit/2"

    monkeypatch.setattr("aquascope.archive.publish.publish_folder", fake_publish)
    fa.publish(tmp_path)
    scorecard = ["verify/forecast_skill.parquet", "verify/manifest.json", "verify/summary.json"]
    assert seen["files"] == ["forecasts/manifest.json"] + (scorecard if complete else [])
    assert seen["message"] == "forecasts: issue 2026-10-08" + (", scorecard" if complete else "")


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
