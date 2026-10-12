"""Retention of the daily jobs' dated files (#597): the rule, the plan on a staged folder, and each publish step."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from aquascope.archive import retention


def _days(start: str, n: int) -> list[str]:
    d0 = date.fromisoformat(start)
    return [(d0 + timedelta(days=i)).isoformat() for i in range(n)]


# ── the rule ────────────────────────────────────────────────────────────────


def test_a_year_of_daily_files_keeps_sixty_days_and_one_a_month():
    dates = _days("2026-01-01", 365)                       # 2026-01-01 .. 2026-12-31
    keep, prune = retention.retain(dates)
    assert keep[-60:] == dates[-60:] and dates[-60] == "2026-11-02"   # the last 60 days, back from the newest
    older = [d for d in keep if d < dates[-60]]
    assert older == [f"2026-{m:02d}-01" for m in range(1, 12)]   # then the first of each month (Jan to Nov)
    assert len(keep) == 71 and len(prune) == 294 and not set(keep) & set(prune)
    assert sorted(keep + prune) == dates


def test_the_window_counts_back_from_the_newest_file_not_the_clock():
    dates = _days("2025-03-10", 30)                          # an old, stopped job loses nothing
    assert retention.retain(dates) == (dates, [])


def test_a_gap_keeps_the_first_file_of_each_month_seen():
    dates = ["2026-01-05", "2026-01-20", "2026-03-15", "2026-03-16", "2026-08-01"]
    keep, prune = retention.retain(dates, days=60)
    assert keep == ["2026-01-05", "2026-03-15", "2026-08-01"] and prune == ["2026-01-20", "2026-03-16"]


def test_the_first_of_a_month_stays_kept_as_the_window_moves():
    dates = _days("2026-01-01", 200)
    kept_before = set(retention.retain(dates)[0])
    kept_after = set(retention.retain(dates + _days("2026-07-20", 30))[0])
    firsts = {d for d in kept_before if d.endswith("-01")}
    assert firsts <= kept_after                              # nothing kept once is pruned later for being first


def test_names_that_are_not_dates_are_left_alone():
    keep, prune = retention.retain(["latest", "2026-02-30", "manifest", "2026-10-11"])
    assert keep == ["2026-10-11"] and prune == []
    assert retention.retain([]) == ([], [])
    with pytest.raises(ValueError):
        retention.retain(["2026-10-11"], days=0)


def test_date_of_reads_only_dated_names_with_the_suffix():
    assert retention.date_of("2026-10-11.bin.gz", ".bin.gz") == "2026-10-11"
    assert retention.date_of("latest.bin.gz", ".bin.gz") is None
    assert retention.date_of("2026-10-11.parquet", ".bin.gz") is None
    assert retention.date_of("2026-13-01.json", ".json") is None


# ── the plan on a staged folder ─────────────────────────────────────────────


def _stage(tmp_path: Path, folder: str, suffix: str, local: list[str], history: list[str] | None) -> Path:
    stage = tmp_path / "stage" / folder
    stage.mkdir(parents=True)
    (stage / f"latest{suffix}").write_bytes(b"x")
    for d in local:
        (stage / f"{d}{suffix}").write_bytes(b"x")
    if history is not None:
        (stage / "manifest.json").write_text(json.dumps({"issue_date": local[-1], "history": [
            {"issue_date": d, "file": f"{folder}/{d}{suffix}"} for d in history]}))
    return stage


def test_the_plan_deletes_published_files_and_drops_their_history_in_the_folder_only():
    import tempfile

    published = _days("2026-01-01", 120)                     # up to 2026-04-30
    today = "2026-05-01"
    with tempfile.TemporaryDirectory() as tmp:
        stage = _stage(Path(tmp), "flowstatus", ".bin.gz", [today], published + [today])
        listing = ["latest.bin.gz", "latest.parquet", "manifest.json"] + [f"{d}.bin.gz" for d in published]
        res = retention.plan(stage, "flowstatus", ".bin.gz", repo_id="x/y", listing=listing)
        doc = json.loads((stage / "manifest.json").read_text())
        files = sorted(p.name for p in stage.iterdir())
    keep_from = (date.fromisoformat(today) - timedelta(days=59)).isoformat()
    assert res["listed"] and today in res["keep"] and "2026-01-01" in res["keep"] and "2026-02-01" in res["keep"]
    assert all(d >= keep_from or d.endswith("-01") for d in res["keep"])
    assert res["delete"] == [f"flowstatus/{d}.bin.gz" for d in res["prune"]]
    assert "2026-01-02" in res["prune"] and all(p.startswith("flowstatus/") for p in res["delete"])
    assert not any("latest" in p or "manifest" in p for p in res["delete"])
    assert [h["issue_date"] for h in doc["history"]] == res["keep"]            # the manifest says only what stays
    assert "60 days" in doc["retention"]
    assert f"{today}.bin.gz" in files and "latest.bin.gz" in files


def test_the_plan_deletes_nothing_it_has_not_seen_listed(tmp_path):
    stage = _stage(tmp_path, "forecasts/warnings", ".parquet", ["2026-10-11"], _days("2026-06-01", 133))
    res = retention.plan(stage, "forecasts/warnings", ".parquet", repo_id="x/y", listing=[])
    assert res["prune"] and res["delete"] == []                               # nothing published: nothing to delete
    assert res["history_dropped"] == res["prune"]


def test_a_backfilled_old_day_that_the_rule_drops_is_not_uploaded(tmp_path):
    stage = _stage(tmp_path, "scout", ".json", ["2026-01-15"], None)
    res = retention.plan(stage, "scout", ".json", repo_id="x/y", listing=["2026-01-01.json", "2026-10-11.json"])
    assert res["prune"] == ["2026-01-15"] and res["delete"] == [] and not (stage / "2026-01-15.json").exists()


def test_the_dry_run_changes_nothing(tmp_path, monkeypatch):
    src = _stage(tmp_path, "flowstatus", ".bin.gz", ["2026-10-11"], _days("2026-06-01", 133))
    monkeypatch.setattr(retention, "list_published", lambda folder, repo_id, token=None:
                        [f"{d}.bin.gz" for d in _days("2026-06-01", 132)])
    before = json.loads((src / "manifest.json").read_text())
    res = retention.dry_run(src, "flowstatus", ".bin.gz", repo_id="x/y")
    assert res["dry_run"] and res["delete"] and "flowstatus/2026-06-02.bin.gz" in res["delete"]
    assert json.loads((src / "manifest.json").read_text()) == before


def test_an_unlistable_folder_prunes_nothing(tmp_path, monkeypatch):
    class Boom:
        def __init__(self, token=None):
            pass

        def list_repo_tree(self, *a, **k):
            raise OSError("offline")

    monkeypatch.undo()
    monkeypatch.setattr("huggingface_hub.HfApi", Boom)
    assert retention.list_published("flowstatus", "x/y") is None


def test_only_the_three_derived_folders_are_pruned():
    assert set(retention.PRUNED) == {"flowstatus", "forecasts/warnings", "scout"}
    assert "forecasts/issued" in retention.KEPT_WHOLE and "forecasts/status" in retention.KEPT_WHOLE


# ── each publish step passes the deletions in its own commit ────────────────


def _fake_publish(seen: dict):
    def fake(folder, repo_id, **kw):
        seen["files"] = sorted(p.relative_to(folder).as_posix() for p in Path(folder).rglob("*") if p.is_file())
        manifest = Path(folder) / seen.get("folder", "") / "manifest.json"
        if manifest.exists():
            seen["history"] = [h["issue_date"] for h in json.loads(manifest.read_text()).get("history", [])]
        seen.update(kw)
        return "https://huggingface.co/commit/9"
    return fake


@pytest.mark.parametrize("module,folder,suffix", [
    ("aquascope.flowstatus", "flowstatus", ".bin.gz"),
    ("aquascope.archive.warnings", "forecasts/warnings", ".parquet"),
    ("aquascope.map_scout", "scout", ".json"),
])
def test_publish_prunes_in_the_same_commit(tmp_path, monkeypatch, module, folder, suffix):
    import importlib

    mod = importlib.import_module(module)
    published = _days("2026-06-01", 132)                      # 2026-06-01 .. 2026-10-10
    today = "2026-10-11"
    src = tmp_path / "out" / folder
    src.mkdir(parents=True)
    (src / f"latest{suffix}").write_bytes(b"x")
    (src / f"{today}{suffix}").write_bytes(b"x")
    if folder == "scout":
        (src / "latest.json").write_text(json.dumps({"date": today}))
    else:
        (src / "manifest.json").write_text(json.dumps({"issue_date": today, "smoke": False, "history": [
            {"issue_date": d} for d in published + [today]]}))
    monkeypatch.setattr(retention, "list_published", lambda f, repo_id, token=None:
                        ["manifest.json", f"latest{suffix}"] + [f"{d}{suffix}" for d in published])
    seen: dict = {"folder": folder}
    monkeypatch.setattr("aquascope.archive.publish.publish_folder", _fake_publish(seen))
    assert mod.publish(tmp_path / "out").endswith("/9")
    keep, prune = retention.retain(published + [today])
    assert seen["delete"] == [f"{folder}/{d}{suffix}" for d in prune]
    assert all(p.startswith(folder + "/") for p in seen["delete"]) and "pruned" in seen["commit_message"]
    assert f"{folder}/{today}{suffix}" in seen["files"]
    if folder != "scout":
        assert seen["history"] == keep
    dry = mod.prune_plan(tmp_path / "out")
    assert dry["dry_run"] and dry["delete"] == seen["delete"]
    if folder != "scout":                                     # the local folder is never changed, only the stage
        assert len(json.loads((src / "manifest.json").read_text())["history"]) == 133


def test_publish_folder_deletes_in_one_commit(tmp_path, monkeypatch):
    from aquascope.archive import publish as pub

    (tmp_path / "flowstatus").mkdir()
    (tmp_path / "flowstatus" / "latest.bin.gz").write_bytes(b"x")
    (tmp_path / "flowstatus" / "2026-10-11.bin.gz").write_bytes(b"x")
    commits = []

    class Api:
        def __init__(self, token=None):
            pass

        def create_repo(self, *a, **k):
            pass

        def create_commit(self, **kw):
            commits.append(kw)
            return type("Info", (), {"commit_url": "https://huggingface.co/commit/7"})()

        def upload_folder(self, **kw):  # pragma: no cover - must not be used when deleting
            raise AssertionError("upload_folder cannot delete exact files without listing the whole repo")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    url = pub.publish_folder(tmp_path, "x/y", token="t", allow_patterns=["flowstatus/*"],
                             delete=["flowstatus/2026-06-02.bin.gz", "flowstatus/2026-10-11.bin.gz"])
    assert url.endswith("/7")
    ops = commits[0]["operations"]
    adds = sorted(o.path_in_repo for o in ops if type(o).__name__ == "CommitOperationAdd")
    dels = [o.path_in_repo for o in ops if type(o).__name__ == "CommitOperationDelete"]
    assert adds == ["flowstatus/2026-10-11.bin.gz", "flowstatus/latest.bin.gz"]
    assert dels == ["flowstatus/2026-06-02.bin.gz"]                           # never delete what is uploaded


def test_a_failed_deleting_commit_still_uploads_the_day(tmp_path, monkeypatch):
    from aquascope.archive import publish as pub

    (tmp_path / "flowstatus").mkdir()
    (tmp_path / "flowstatus" / "2026-10-11.bin.gz").write_bytes(b"x")
    uploads = []

    class Api:
        def __init__(self, token=None):
            pass

        def create_repo(self, *a, **k):
            pass

        def create_commit(self, **kw):
            raise RuntimeError("404 flowstatus/2026-06-02.bin.gz does not exist")

        def upload_folder(self, **kw):
            uploads.append(kw)
            return type("Info", (), {"commit_url": "https://huggingface.co/commit/8"})()

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    url = pub.publish_folder(tmp_path, "x/y", token="t", allow_patterns=["flowstatus/*"],
                             delete=["flowstatus/2026-06-02.bin.gz"])
    assert url.endswith("/8") and uploads[0]["allow_patterns"] == ["flowstatus/*"]   # pruning never costs the day
