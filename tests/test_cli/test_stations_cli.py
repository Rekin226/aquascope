"""`aquascope stations` CLI: registry-driven search, three output formats, honest exit codes."""

from __future__ import annotations

import csv
import json
import sys

import pytest

from aquascope import cli
from aquascope.registry import StationCatalog
from aquascope.schemas.station import Station


def _fake_catalogs(**kwargs):
    ok = StationCatalog(
        source="ireland_opw",
        stations=[
            Station(source="ireland_opw", station_id="1", name="A", latitude=53.3, longitude=-6.2,
                    variables=("water_level",), extra={"k": "v"}),
        ],
        seconds=0.1,
    )
    bad = StationCatalog(source="uk_ea", error="RuntimeError: 503", seconds=0.2)
    return {"ireland_opw": ok, "uk_ea": bad}


@pytest.mark.parametrize("fmt", ["geojson", "json", "csv"])
def test_stations_writes_requested_format(tmp_path, monkeypatch, fmt):
    monkeypatch.setattr("aquascope.registry.station_catalogs", _fake_catalogs)
    out = tmp_path / f"out.{fmt}"
    monkeypatch.setattr(sys, "argv", ["aquascope", "stations", "--source", "ireland_opw", "--source", "uk_ea",
                                      "--format", fmt, "-o", str(out)])
    cli.main()
    assert out.exists()
    if fmt == "geojson":
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["type"] == "FeatureCollection"
        feat = data["features"][0]
        assert feat["geometry"]["coordinates"] == [-6.2, 53.3]
        assert feat["properties"]["source"] == "ireland_opw"
        assert "latitude" not in feat["properties"]
    elif fmt == "json":
        rows = json.loads(out.read_text(encoding="utf-8"))
        assert rows[0]["station_id"] == "1" and rows[0]["variables"] == ["water_level"]
    else:
        with out.open() as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["variables"] == "water_level"
        assert json.loads(rows[0]["extra"]) == {"k": "v"}


def test_stations_forwards_filters(tmp_path, monkeypatch):
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return _fake_catalogs()

    monkeypatch.setattr("aquascope.registry.station_catalogs", fake)
    out = tmp_path / "o.json"
    # A bbox starting with a minus must be passed as --bbox=... (argparse would read it as an option otherwise)
    monkeypatch.setattr(sys, "argv", ["aquascope", "stations", "--bbox=-7,53,-6,54", "--variable", "water_level",
                                      "--max-items", "5", "--format", "json", "-o", str(out)])
    cli.main()
    assert seen["bbox"] == (-7.0, 53.0, -6.0, 54.0)
    assert seen["variable"] == "water_level"
    assert seen["max_items"] == 5
    assert seen["sources"] is None


def test_stations_exits_nonzero_when_nothing_found_and_a_source_failed(tmp_path, monkeypatch):
    def fake(**kwargs):
        return {"uk_ea": StationCatalog(source="uk_ea", error="RuntimeError: 503")}

    monkeypatch.setattr("aquascope.registry.station_catalogs", fake)
    monkeypatch.setattr(sys, "argv", ["aquascope", "stations", "--source", "uk_ea", "-o", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1


def test_stations_rejects_unknown_variable(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["aquascope", "stations", "--variable", "lava"])
    with pytest.raises(SystemExit):
        cli.main()


def test_harvest_cli_calls_harvest_and_publish(tmp_path, monkeypatch, capsys):
    from aquascope.archive.harvest import HarvestReport, SourceHealth

    report = HarvestReport(run_at="2026-08-16T00:00:00+00:00", aquascope_version="t", n_stations=3, sources=[
        SourceHealth("ireland_opw", True, 3, 1.0, None, "CC-BY-4.0", True, "OPW"),
        SourceHealth("uk_ea", False, 0, 2.0, "RuntimeError: 503", "OGL-UK-3.0", True, "EA"),
    ])
    calls = {}
    monkeypatch.setattr("aquascope.archive.harvest_stations", lambda out, **kw: calls.update(out=out, **kw) or report)
    monkeypatch.setattr("aquascope.archive.publish_folder", lambda out, repo, **kw: calls.update(repo=repo) or "url")
    monkeypatch.setattr(sys, "argv", ["aquascope", "harvest", "stations", "--out", str(tmp_path), "--source", "uk_ea",
                                      "--max-items", "9", "--publish", "me/ds"])
    cli.main()
    out = capsys.readouterr().out
    assert calls["out"] == str(tmp_path) and calls["sources"] == ["uk_ea"] and calls["max_items"] == 9
    assert calls["repo"] == "me/ds"
    assert "FAILED: RuntimeError: 503" in out and "published: url" in out


def test_harvest_cli_exits_nonzero_when_all_failed(tmp_path, monkeypatch):
    from aquascope.archive.harvest import HarvestReport, SourceHealth

    report = HarvestReport(run_at="x", aquascope_version="t", n_stations=0, sources=[
        SourceHealth("uk_ea", False, 0, 2.0, "RuntimeError: 503", "OGL-UK-3.0", True, "EA"),
    ])
    monkeypatch.setattr("aquascope.archive.harvest_stations", lambda out, **kw: report)
    monkeypatch.setattr(sys, "argv", ["aquascope", "harvest", "stations", "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
