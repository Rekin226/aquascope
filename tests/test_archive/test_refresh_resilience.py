"""A stopped source must leave its completed work and other sources publishable."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from aquascope.archive import observations as obs
from aquascope.archive.refresh import refresh

CATALOG = [{"source": "usgs", "station_id": "A", "variables": ["discharge"], "period_start": "1903-01-01"}]


def _record(start="2000-01-01", count=10):
    return {"series": pd.Series(range(1, count + 1), index=pd.date_range(start, periods=count)),
            "variable": "discharge", "unit": "m3/s"}


def _run(out, result):
    with patch("aquascope.explore.fetch_series", return_value=result) as fetch:
        obs.harvest_observations(out, sources=["usgs"], variable="discharge", catalog=CATALOG, refresh_days=0)
    return fetch


def test_empty_refresh_retains_last_success_and_data(tmp_path):
    fetch = _run(tmp_path, _record())
    assert fetch.call_args.kwargs["period_start"] == "1903-01-01"
    first = obs.load_manifest(tmp_path)["sources"]["usgs/discharge"]["stations"]["A"]
    _run(tmp_path, {"series": None, "variable": "discharge"})
    later = obs.load_manifest(tmp_path)["sources"]["usgs/discharge"]["stations"]["A"]
    assert later["n"] == first["n"] == 10 and later["harvested_at"] == first["harvested_at"]
    assert later["sha256"] == first["sha256"] and later["last_attempt_status"] == "empty"
    assert obs.read_csv_gz(obs.obs_path(tmp_path, "discharge", "usgs", "A").read_bytes()).size == 10


def test_short_refresh_merges_new_values_without_losing_history(tmp_path):
    _run(tmp_path, _record())
    _run(tmp_path, _record("2000-01-10", 5))
    s = obs.read_csv_gz(obs.obs_path(tmp_path, "discharge", "usgs", "A").read_bytes())
    assert len(s) == 14 and s.loc["2000-01-10"] == 1
    assert obs.load_manifest(tmp_path)["sources"]["usgs/discharge"]["n_stations"] == 1


def test_station_checkpoint_survives_interruption_before_end_of_source(tmp_path):
    catalog = CATALOG + [{**CATALOG[0], "station_id": "B"}]
    with patch("aquascope.explore.fetch_series", side_effect=[_record(), KeyboardInterrupt]):
        with pytest.raises(KeyboardInterrupt):
            obs.harvest_observations(tmp_path, sources=["usgs"], variable="discharge", catalog=catalog)
    m = obs.load_manifest(tmp_path)
    assert m["sources"]["usgs/discharge"]["stations"]["A"]["n"] == 10
    assert m["sources"]["usgs/discharge"]["n_stations"] == 1


def test_timeout_of_one_source_does_not_prevent_the_next_source(tmp_path):
    with patch("aquascope.archive.refresh.subprocess.run", side_effect=[
        subprocess.TimeoutExpired("worker", 1), SimpleNamespace(returncode=0),
    ]) as run:
        result = refresh(tmp_path, pairs=[("usgs", "discharge"), ("uk_ea", "discharge")], timeout=1)
    assert run.call_count == 2 and result["status"] == "partial"
    assert [s["status"] for s in result["sources"]] == ["timeout", "ok"]
    assert result["completed_at"]
    assert json.loads((tmp_path / "obs/refresh_status.json").read_text()) == result


def test_sync_failure_cannot_silently_reset_the_published_manifest(tmp_path):
    hub = SimpleNamespace(snapshot_download=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    with patch("aquascope.utils.imports.require", return_value=hub), pytest.raises(RuntimeError, match="offline"):
        obs.sync_from_hub(tmp_path, "owner/archive")


def test_corrupt_manifest_is_not_silently_replaced(tmp_path):
    (tmp_path / "obs").mkdir()
    path = tmp_path / "obs/manifest.json"
    path.write_text('{"sources":', encoding="utf-8")
    with pytest.raises(ValueError, match="last good manifest"):
        obs.load_manifest(tmp_path)
    assert path.read_text(encoding="utf-8") == '{"sources":'
