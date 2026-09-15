"""Archive Phase 0 (#188): harvest station catalogs into GeoParquet + GeoJSON + health.json."""

from __future__ import annotations

import json
import struct
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from aquascope.registry import StationCatalog
from aquascope.schemas.station import Station

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from aquascope.archive import harvest_stations, publish_folder, write_dataset_card  # noqa: E402
from aquascope.archive.harvest import HarvestReport, stations_to_table  # noqa: E402


def _stations():
    return [
        Station(source="ireland_opw", station_id="0000001041", name="Sandy Mills", latitude=54.84, longitude=-7.58,
                variables=("water_level",), url="https://waterlevel.ie/0001/1041/", country="IRL"),
        Station(source="pegelonline", station_id="u1", site_id="celle-site", name="CELLE",
                latitude=52.62, longitude=10.06, variables=("discharge", "water_level"), river="ALLER", country="DEU",
                period_start=date(1990, 1, 1), extra={"number": "48300105"}),
    ]


def _fake_catalogs(**kwargs):
    ok = StationCatalog(source="ireland_opw", stations=[_stations()[0]], seconds=1.2)
    ok2 = StationCatalog(source="pegelonline", stations=[_stations()[1]], seconds=0.4)
    bad = StationCatalog(source="uk_ea", error="RuntimeError: 503", seconds=3.0)
    return {"ireland_opw": ok, "pegelonline": ok2, "uk_ea": bad}


def test_stations_table_is_geoparquet():
    table = stations_to_table(_stations())
    assert table.num_rows == 2
    geo = json.loads(table.schema.metadata[b"geo"])
    assert geo["primary_column"] == "geometry"
    col = geo["columns"]["geometry"]
    assert col["encoding"] == "WKB" and col["geometry_types"] == ["Point"]
    assert col["bbox"] == [-7.58, 52.62, 10.06, 54.84]
    # WKB: little-endian point with the station's lon/lat
    wkb = table.column("geometry")[0].as_py()
    order, gtype, x, y = struct.unpack("<BIdd", wkb)
    assert (order, gtype) == (1, 1) and (x, y) == (-7.58, 54.84)
    # registry-derived columns
    row = table.slice(1, 1).to_pylist()[0]
    assert row["agency"].startswith("Wasserstra")
    assert row["license"] == "DL-DE-BY-2.0" and row["redistributable"] is True
    assert row["variables"] == ["discharge", "water_level"]
    assert json.loads(row["extra"]) == {"number": "48300105"}
    assert row["period_start"] == date(1990, 1, 1)


def test_harvest_writes_files_and_health(tmp_path):
    with patch("aquascope.archive.harvest.station_catalogs", side_effect=_fake_catalogs):
        report = harvest_stations(tmp_path / "archive")
    out = tmp_path / "archive"
    assert (out / "stations.parquet").exists()
    assert (out / "stations.geojson").exists()
    assert (out / "health.json").exists()
    assert (out / "README.md").exists()

    assert report.n_stations == 2 and report.n_ok == 2 and report.n_failed == 1
    health = json.loads((out / "health.json").read_text())
    by = {s["source"]: s for s in health["sources"]}
    assert by["uk_ea"]["ok"] is False and "503" in by["uk_ea"]["error"]
    assert by["ireland_opw"]["n_stations"] == 1 and by["ireland_opw"]["license"] == "CC-BY-4.0"
    assert health["n_ok"] == 2 and health["aquascope_version"]

    table = pq.read_table(out / "stations.parquet")
    assert table.num_rows == 2
    assert b"geo" in table.schema.metadata
    # sorted by (source, station_id)
    assert table.column("source").to_pylist() == ["ireland_opw", "pegelonline"]
    assert table.column("site_id").to_pylist() == ["0000001041", "celle-site"]

    gj = json.loads((out / "stations.geojson").read_text())
    assert gj["type"] == "FeatureCollection" and len(gj["features"]) == 2
    f0 = gj["features"][0]
    assert f0["geometry"]["coordinates"] == [-7.58, 54.84]
    assert f0["properties"] == {"source": "ireland_opw", "station_id": "0000001041", "name": "Sandy Mills",
                                "site_id": "0000001041",
                                "variables": ["water_level"], "url": "https://waterlevel.ie/0001/1041/"}
    assert "extra" not in gj["features"][1]["properties"]  # extras live in the parquet only
    assert gj["features"][1]["properties"]["period_start"] == "1990-01-01"
    assert gj["features"][1]["properties"]["site_id"] == "celle-site"

    card = (out / "README.md").read_text()
    assert card.startswith("---\nlicense: other")
    assert "| `uk_ea` |" in card and "failed: RuntimeError: 503" in card
    assert "resolve/main/stations.parquet" in card
    assert "Group by `(source, site_id)`" in card


def test_catalog_site_ids_round_trip_and_legacy_fallback(tmp_path):
    from aquascope.archive.catalog import _rows_from_geojson, load_stations
    from aquascope.archive.harvest import write_stations_geojson

    path = tmp_path / "stations.parquet"
    table = stations_to_table(_stations())
    pq.write_table(table, path)
    assert [r["site_id"] for r in load_stations(path=path)] == ["0000001041", "celle-site"]

    pq.write_table(table.drop(["site_id"]), path)
    assert [r["site_id"] for r in load_stations(path=path)] == ["0000001041", "u1"]

    geojson_path = tmp_path / "stations.geojson"
    write_stations_geojson(_stations(), geojson_path)
    assert [r["site_id"] for r in _rows_from_geojson(geojson_path)] == ["0000001041", "celle-site"]
    legacy = json.loads(geojson_path.read_text())
    for feature in legacy["features"]:
        del feature["properties"]["site_id"]
    geojson_path.write_text(json.dumps(legacy))
    assert [r["site_id"] for r in _rows_from_geojson(geojson_path)] == ["0000001041", "u1"]


def test_harvest_restricts_sources_and_passes_options(tmp_path):
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {"pegelonline": StationCatalog(source="pegelonline", stations=[_stations()[1]])}

    with patch("aquascope.archive.harvest.station_catalogs", side_effect=fake):
        report = harvest_stations(tmp_path, sources=["pegelonline"], max_items=7, write_geojson=False, write_card=False)
    assert seen["sources"] == ["pegelonline"] and seen["max_items"] == 7
    assert report.n_stations == 1
    assert not (tmp_path / "stations.geojson").exists()
    assert not (tmp_path / "README.md").exists()
    assert (tmp_path / "stations.parquet").exists()


def test_empty_harvest_still_writes_valid_files(tmp_path):
    with patch("aquascope.archive.harvest.station_catalogs", return_value={
        "uk_ea": StationCatalog(source="uk_ea", error="RuntimeError: down")
    }):
        report = harvest_stations(tmp_path)
    assert report.n_stations == 0 and report.n_ok == 0
    table = pq.read_table(tmp_path / "stations.parquet")
    assert table.num_rows == 0
    assert json.loads(table.schema.metadata[b"geo"])["columns"]["geometry"]["bbox"] is None


def test_dataset_card_lists_every_source(tmp_path):
    report = HarvestReport(run_at="2026-08-16T00:00:00+00:00", aquascope_version="x", n_stations=0, sources=[])
    path = write_dataset_card(tmp_path / "README.md", report, repo_id="me/ds")
    text = path.read_text()
    assert "hf://datasets/me/ds/stations.parquet" in text
    assert "aquascope harvest stations" in text


def test_publish_folder_uses_hf_api(tmp_path, monkeypatch):
    (tmp_path / "stations.parquet").write_bytes(b"x")
    fake_hub = MagicMock()
    fake_api = fake_hub.HfApi.return_value
    fake_api.upload_folder.return_value = MagicMock(commit_url="https://hf.co/commit/1")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    with patch("aquascope.archive.publish.require", return_value=fake_hub):
        url = publish_folder(tmp_path, "me/ds", commit_message="msg")
    fake_hub.HfApi.assert_called_once_with(token="hf_test")
    fake_api.create_repo.assert_called_once_with("me/ds", repo_type="dataset", private=False, exist_ok=True)
    kwargs = fake_api.upload_folder.call_args.kwargs
    assert kwargs["repo_id"] == "me/ds" and kwargs["repo_type"] == "dataset" and kwargs["commit_message"] == "msg"
    assert url == "https://hf.co/commit/1"


def test_publish_folder_rejects_missing_folder(tmp_path):
    with patch("aquascope.archive.publish.require", return_value=MagicMock()):
        with pytest.raises(FileNotFoundError):
            publish_folder(tmp_path / "nope", "me/ds")
