"""Per-gauge Atom feeds (#521): which entries, valid Atom, what is written and published."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

pq = pytest.importorskip("pyarrow.parquet", reason="the archive extra")

from aquascope.archive import feeds  # noqa: E402
from aquascope.archive import forecasts as fa  # noqa: E402
from tests.test_watch import assert_valid_atom  # noqa: E402

TODAY = date(2026, 10, 8)
MADE = datetime(2026, 10, 8, 7, 30, tzinfo=timezone.utc)
WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "forecast-archive.yml"
STATUS = {"source": "usgs", "station_id": "USGS-01646500", "value_date": "2026-10-07", "value": 140.0,
          "percentile": 93.4, "class": "much_above", "n_years": 46}


def _issued(peak: float, *, q2: float = 100.0, q10: float = 150.0, day: str = "2026-10-08") -> list[dict]:
    return [{"source": "usgs", "station_id": "USGS-01646500", "model": "geoglows", "issue_date": day,
             "valid_date": f"2026-10-{d:02d}", "mean_c": v, "gauge_q2": q2, "gauge_q5": 130.0, "gauge_q10": q10}
            for d, v in ((8, 90.0), (9, peak), (10, 80.0))] + [
        {"source": "usgs", "station_id": "USGS-01646500", "model": "glofas", "issue_date": day,
         "valid_date": "2026-10-09", "mean_c": None, "mean": 9999.0}]


def test_safe_paths():
    assert feeds.feed_path("usgs", "USGS-01646500") == "usgs/USGS-01646500.xml"
    assert feeds.feed_path("uk_ea", "a b/c?") == "uk_ea/a_b_c_.xml"
    assert feeds.explorer_link("usgs", "USGS-1").endswith("#s=usgs/USGS-1&tab=now")


def test_forecast_level_is_the_rarest_flow_passed_by_the_corrected_mean():
    assert feeds.forecast_level(_issued(140.0), today=TODAY)["return_period"] == 5
    lvl = feeds.forecast_level(_issued(160.0), today=TODAY)
    assert lvl == {"return_period": 10, "q": 150.0, "peak": 160.0, "peak_date": "2026-10-09",
                   "issue_date": "2026-10-08"}
    assert feeds.forecast_level(_issued(95.0), today=TODAY) is None
    assert feeds.forecast_level([], today=TODAY) is None


def test_entries_on_a_class_change_and_a_forecast_alert_without_repeats():
    first = feeds.gauge_entries(STATUS, _issued(160.0), [], name="Potomac", made=MADE, today=TODAY)
    assert [e["kind"] for e in first] == ["status", "forecast"] or [e["kind"] for e in first] == ["forecast", "status"]
    st = next(e for e in first if e["kind"] == "status")
    assert st["title"] == "Potomac: much above normal"
    assert "93rd percentile of 46 years" in st["summary"] and st["entry_id"].endswith("#status-2026-10-07")
    fc = next(e for e in first if e["kind"] == "forecast")
    assert fc["title"] == "Potomac: forecast above the 10-year flow" and "not a flood warning" in fc["summary"]
    # the next day: same class, same level -> nothing new
    nxt = datetime(2026, 10, 9, 7, 30, tzinfo=timezone.utc)
    again = feeds.gauge_entries({**STATUS, "value_date": "2026-10-08"}, _issued(160.0, day="2026-10-09"), first,
                                name="Potomac", made=nxt, today=date(2026, 10, 9))
    assert len(again) == 2
    # a class change adds one, and says what it was
    calmer = feeds.gauge_entries({**STATUS, "value_date": "2026-10-09", "class": "above", "percentile": 80.0},
                                 [], again, name="Potomac", made=nxt, today=date(2026, 10, 9))
    assert len(calmer) == 3 and calmer[0]["kind"] == "status" and "It was much above normal before." in \
        calmer[0]["summary"]


def test_entries_are_capped_and_a_rerun_does_not_duplicate():
    old = [{"source": "usgs", "station_id": "USGS-01646500", "kind": "status", "cls": "normal" if i % 2 else "below",
            "level": None, "entry_id": f"x#status-{i}", "updated": f"2026-09-{i + 1:02d}T07:00:00Z", "title": "t",
            "summary": "s", "link": "l"} for i in range(25)]
    got = feeds.gauge_entries(STATUS, [], old, name="P", made=MADE, today=TODAY)
    assert len(got) == feeds.MAX_ENTRIES and got[0]["cls"] == "much_above"
    twice = feeds.gauge_entries(STATUS, [], got[1:], name="P", made=MADE, today=TODAY)
    assert len({e["entry_id"] for e in twice}) == len(twice)


def test_run_writes_valid_atom_an_index_and_the_state(tmp_path):
    status = [STATUS, {**STATUS, "station_id": "odd id/2", "class": "normal", "percentile": 50.0}]
    info = feeds.run(tmp_path, today=TODAY, made=MADE, status=status, issued=_issued(160.0), state=[],
                     catalog=[{"source": "usgs", "station_id": "USGS-01646500", "name": "POTOMAC RIVER & CO"}])
    assert info == {"feeds": 2, "new_status_entries": 2, "new_forecast_entries": 1, "date": "2026-10-08"}
    root = tmp_path / "feeds"
    text = (root / "usgs" / "USGS-01646500.xml").read_text(encoding="utf-8")
    assert_valid_atom(text)
    assert "POTOMAC RIVER &amp; CO: AquaScope watch" in text and "CC BY 4.0" in text
    assert_valid_atom((root / "usgs" / "odd_id_2.xml").read_text(encoding="utf-8"))
    index = json.loads((root / "index.json").read_text())
    assert index["feeds"] == {"usgs/USGS-01646500": "usgs/USGS-01646500.xml", "usgs/odd id/2": "usgs/odd_id_2.xml"}
    assert index["base"].endswith("/resolve/main/feeds/") and "Not flood warnings" in index["about"]
    state = pq.read_table(root / "state.parquet").to_pylist()
    assert len(state) == 3 and {"entry_id", "kind", "cls", "level", "updated"} <= set(state[0])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["feeds"]  # nothing outside feeds/


def test_run_reads_what_the_forecast_step_wrote(tmp_path, monkeypatch):
    fa._write_rows([STATUS], tmp_path / "forecasts" / "status" / "latest.parquet")
    fa._write_rows(_issued(160.0), tmp_path / "forecasts" / "issued" / "2026-10-08.parquet")
    monkeypatch.setattr(fa, "read_published_rows", lambda path, repo_id=fa.DEFAULT_REPO: [])
    info = feeds.run(tmp_path, today=TODAY, made=MADE, catalog=[])
    assert info["feeds"] == 1 and info["new_forecast_entries"] == 1


def test_publish_uploads_only_the_feeds_folder(monkeypatch, tmp_path):
    (tmp_path / "feeds" / "usgs").mkdir(parents=True)
    (tmp_path / "feeds" / "index.json").write_text(json.dumps({"date": "2026-10-08"}))
    (tmp_path / "feeds" / "usgs" / "1.xml").write_text("<feed/>")
    (tmp_path / "forecasts").mkdir()
    (tmp_path / "forecasts" / "manifest.json").write_text("{}")
    seen = {}

    def fake_publish(folder, repo_id, token=None, commit_message=None, allow_patterns=None):
        seen["files"] = sorted(str(p.relative_to(folder)) for p in Path(folder).rglob("*") if p.is_file())
        seen["message"] = commit_message
        seen["patterns"] = allow_patterns
        return "https://hf.co/commit/2"

    monkeypatch.setattr("aquascope.archive.publish.publish_folder", fake_publish)
    assert feeds.publish(tmp_path) == "https://hf.co/commit/2"
    assert seen["files"] == ["feeds/index.json", "feeds/usgs/1.xml"] and seen["message"] == "feeds: 2026-10-08"
    # the Atom files themselves must pass the upload's allow-list
    assert "*.xml" in seen["patterns"]


def test_publish_folder_uploads_xml_when_asked(monkeypatch, tmp_path):
    from aquascope.archive import publish as pub

    calls = {}

    class FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, *a, **k):
            return None

        def upload_folder(self, **kw):
            calls.update(kw)
            return type("Info", (), {"commit_url": "https://hf.co/commit/3"})()

    monkeypatch.setattr(pub, "require", lambda *a, **k: type("Hub", (), {"HfApi": FakeApi}))
    pub.publish_folder(tmp_path, "x/y", token="t", allow_patterns=feeds.PUBLISH_PATTERNS)
    assert "*.xml" in calls["allow_patterns"]
    pub.publish_folder(tmp_path, "x/y", token="t")
    assert "*.xml" not in calls["allow_patterns"] and "*.parquet" in calls["allow_patterns"]


def test_the_workflow_builds_feeds_after_the_forecasts_and_publishes_like_them():
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = wf["jobs"]["forecast"]["steps"]
    runs = [s.get("run", "") for s in steps]
    build = next(i for i, r in enumerate(runs) if "aquascope.archive.feeds run" in r)
    assert build > next(i for i, r in enumerate(runs) if "aquascope.archive.forecasts run" in r)
    publish = next(s for s in steps if "aquascope.archive.feeds publish" in s.get("run", ""))
    assert "inputs.max_items == ''" in publish["if"] and "github.event_name == 'schedule'" in publish["if"]
