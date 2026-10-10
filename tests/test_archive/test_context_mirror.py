"""The place-context mirror builder (#520), on small local inputs: nothing is downloaded or published."""

from __future__ import annotations

import gzip
import io
import json
import struct
import zipfile
from pathlib import Path

import pytest

from aquascope.archive import context_mirror as cm
from aquascope.context._common import cell_key, read_csv_gz

pd = pytest.importorskip("pandas")


def _dbf(fields: list[tuple[str, str, int]], records: list[list[str]]) -> bytes:
    """A dBASE III table, the attribute half of a shapefile."""
    reclen = 1 + sum(n for _, _, n in fields)
    hlen = 32 + 32 * len(fields) + 1
    out = bytearray(struct.pack("<BBBBIHH20x", 3, 124, 7, 25, len(records), hlen, reclen))
    for name, kind, n in fields:
        out += name.encode("ascii").ljust(11, b"\x00") + kind.encode("ascii") + b"\x00" * 4 + bytes([n, 0])
        out += b"\x00" * 14
    out += b"\x0d"
    for rec in records:
        out += b" " + b"".join(v.encode("utf-8").ljust(n)[:n] for v, (_, _, n) in zip(rec, fields))
    return bytes(out + b"\x1a")


GDW_DBF_FIELDS = [("GDW_ID", "N", 8), ("DAM_NAME", "C", 30), ("RES_NAME", "C", 30), ("RIVER", "C", 20),
                  ("COUNTRY", "C", 20), ("YEAR_DAM", "N", 6), ("DAM_HGT_M", "N", 8), ("CAP_MCM", "N", 10),
                  ("AREA_SKM", "N", 8), ("MAIN_USE", "C", 20), ("DOR_PC", "N", 8), ("CATCH_SKM", "N", 10),
                  ("LAT_RIV", "N", 12), ("LONG_RIV", "N", 12), ("LAT_DAM", "N", 12), ("LONG_DAM", "N", 12),
                  ("GRAND_ID", "N", 8)]


def test_read_dbf_and_dam_rows_clean_the_missing_codes():
    data = _dbf(GDW_DBF_FIELDS, [
        ["1", "Hoover", "Lake Mead", "Colorado", "United States", "1936", "221", "34852", "640", "Irrigation",
         "", "", "36.0161", "-114.7377", "0", "0", "597"],
        ["2", "Nameless", "", "", "Peru", "-99", "-99", "-99", "", "", "", "", "-12.5", "-75.1", "-12.49", "-75.11",
         "-99"],
        ["3", "Off the map", "", "", "", "", "", "", "", "", "", "", "123.0", "0.0", "", "", ""],
        ["4", "Nowhere", "", "", "", "", "", "", "", "", "", "", "0", "0", "", "", ""],
    ])
    recs = list(cm.read_dbf(data))
    assert recs[0]["DAM_NAME"] == "Hoover" and recs[0]["CAP_MCM"] == 34852.0
    rows = cm.dam_rows(recs)
    assert len(rows) == 2  # the latitude-123 row and the 0, 0 row are dropped
    hoover, nameless = rows
    assert hoover["gdw_id"] == 1 and hoover["year"] == 1936 and hoover["grand_id"] == 597
    # placed at the river point (LAT_RIV/LONG_RIV), which GDW fills even where the surveyed LAT_DAM is 0
    assert (hoover["lat"], hoover["lon"]) == (36.0161, -114.7377) and nameless["lat"] == -12.5
    assert hoover["cell"] == cell_key(36.0161, -114.7377)
    assert nameless["year"] is None and nameless["capacity_mcm"] is None and nameless["grand_id"] is None


def test_build_dams_writes_sorted_parquet_and_cells(tmp_path):
    pytest.importorskip("pyarrow")
    zpath = tmp_path / "GDW_v1_0_shp.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("GDW_v1_0_shp/GDW_barriers_v1_0.dbf", _dbf(GDW_DBF_FIELDS, [
            ["2", "B", "", "", "", "2000", "", "", "", "", "", "", "45.5", "5.5", "", "", ""],
            ["1", "A", "", "", "", "1950", "", "", "", "", "", "", "-33.0", "151.0", "", "", ""],
        ]))
    info = cm.build_dams(tmp_path, source=zpath)
    assert info["rows"] == 2 and sorted(info["cells"]) == sorted([cell_key(45.5, 5.5), cell_key(-33.0, 151.0)])
    assert info["licence"] == "CC-BY-4.0" and "Global Dam Watch" in info["attribution"]
    root = tmp_path / "context"
    table = pd.read_parquet(root / "dams" / "gdw_barriers.parquet")
    assert list(table["cell"]) == sorted(table["cell"])  # sorted by cell, so a bbox read skips row groups
    cell_rows = read_csv_gz((root / "dams" / "cells" / f"{cell_key(45.5, 5.5)}.csv.gz").read_bytes())
    assert cell_rows[0]["name"] == "B" and cell_rows[0]["year"] == "2000"


def test_groundsource_footprints_become_centres_and_boxes(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    shapely = pytest.importorskip("shapely")
    polys = [shapely.box(5.0, 45.0, 5.2, 45.2), shapely.box(-1.0, 51.0, -0.8, 51.4)]
    src = tmp_path / "gs.parquet"
    pq.write_table(pa.table({
        "uuid": ["a", "b"], "area_km2": [10.5, 4.25], "geometry": [shapely.to_wkb(p) for p in polys],
        "start_date": ["2021-05-01", "2018-02-10"], "end_date": ["2021-05-03", "2018-02-11"],
        "unused": [1, 2],
    }), src)
    info = cm.build_groundsource(tmp_path, source=src)
    assert info["rows"] == 2 and info["first"] == "2018-02-10" and info["last"] == "2021-05-01"
    rows = read_csv_gz((tmp_path / "context" / "floods" / "groundsource" / "cells" /
                        f"{cell_key(45.1, 5.1)}.csv.gz").read_bytes())
    assert rows[0]["uuid"] == "a" and float(rows[0]["lat"]) == pytest.approx(45.1)
    assert float(rows[0]["west"]) == 5.0 and float(rows[0]["north"]) == 45.2


def test_microsoft_filters_aggregates_and_merges(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    good = {"dem_metric_2": 1, "soil_moisture_sca": 2, "soil_moisture_zscore": 2, "soil_moisture": 30, "temp": 5,
            "land_cover": 40, "edge_false_positives": 0}
    rows = [
        {"year": 2020, "month": 7, "lat": 45.051, "lon": 5.051, **good},
        {"year": 2020, "month": 7, "lat": 45.052, "lon": 5.052, **good},
        {"year": 2020, "month": 7, "lat": 45.053, "lon": 5.053, **{**good, "land_cover": 60}},  # permanent water
        {"year": 2020, "month": 7, "lat": 45.054, "lon": 5.054, **{**good, "temp": -3}},  # frozen
    ]
    agg = cm.ms_aggregate(pd.DataFrame(rows))
    assert agg.to_dict("records") == [{"lat": 45.075, "lon": 5.075, "year": 2020, "month": 7, "n": 2}]
    parts = tmp_path / "parts"
    parts.mkdir()
    pq.write_table(cm._to_arrow(agg), parts / "part-000.parquet")
    pq.write_table(cm._to_arrow(agg), parts / "part-001.parquet")
    info = cm.merge_microsoft(tmp_path, parts)
    assert info["rows"] == 1 and info["shards"] == 2 and info["licence"] == "MIT"
    merged = pd.read_parquet(tmp_path / "context" / "floods" / "microsoft.parquet")
    assert int(merged["n"].iloc[0]) == 4  # the two shards' counts summed


def test_shards_balance_by_size_and_cover_every_file():
    files = [(f"f{i}.parquet", size) for i, size in enumerate([100, 90, 10, 10, 5, 1])]
    shards = [cm.shard_files(files, k, 3) for k in range(3)]
    assert sorted(p for s in shards for p in s) == sorted(p for p, _ in files)
    assert "f0.parquet" in shards[0] and "f1.parquet" in shards[1]


def test_ghcn_precipitation_stations_with_their_years(tmp_path):
    stations = "\n".join([
        f"{'USW00094728':<11} {40.7789:8.4f} {-73.9692:9.4f} {39.6:6.1f}    {'NY CITY CNTRL PARK':<30}",
        f"{'TMAXONLY001':<11} {10.0:8.4f} {10.0:9.4f} {1.0:6.1f}    {'NO RAIN':<30}",
    ])
    inventory = "\n".join([
        f"{'USW00094728':<11} {40.7789:8.4f} {-73.9692:9.4f} PRCP 1869 2026",
        f"{'USW00094728':<11} {40.7789:8.4f} {-73.9692:9.4f} TMAX 1869 2026",
        f"{'TMAXONLY001':<11} {10.0:8.4f} {10.0:9.4f} TMAX 1990 2000",
    ])
    rows = cm.ghcn_prcp_stations(stations, inventory)
    assert [(r["id"], r["first_year"], r["last_year"]) for r in rows] == [("USW00094728", 1869, 2026)]
    info = cm.build_ghcn(tmp_path, stations_txt=stations, inventory_txt=inventory)
    assert info["rows"] == 1 and info["licence"] == "CC0-1.0"
    text = gzip.decompress((tmp_path / "context" / "ghcn" / "prcp_stations.csv.gz").read_bytes()).decode()
    assert text.splitlines()[0] == "id,lat,lon,elev_m,name,first_year,last_year"


def test_manifest_collects_the_built_datasets_and_publish_uploads_only_the_context_folder(tmp_path, monkeypatch):
    cm.build_ghcn(tmp_path, stations_txt="", inventory_txt="")
    manifest = cm.write_manifest(tmp_path, merge=False)
    assert set(manifest["datasets"]) == {"ghcn"} and manifest["cell_deg"] == 2
    seen = {}

    (tmp_path / "ms_parts").mkdir()
    (tmp_path / "ms_parts" / "part-000.parquet").write_bytes(b"working file")

    def fake_publish(folder, repo_id, token=None, commit_message=None, **kw):
        from pathlib import Path

        top = {p.relative_to(folder).parts[0] for p in Path(folder).rglob("*")}
        seen.update(top=top, repo=repo_id, message=commit_message)
        return "https://huggingface.co/commit/x"

    monkeypatch.setattr("aquascope.archive.publish.publish_folder", fake_publish)
    assert cm.publish(tmp_path, repo_id="me/test") == "https://huggingface.co/commit/x"
    assert seen["repo"] == "me/test" and "ghcn" in seen["message"]
    assert seen["top"] == {"context"}  # the shard parts beside it stay local
    assert not list((tmp_path / "context").glob("_*.json"))  # the per-step notes are not uploaded
    on_disk = json.loads((tmp_path / "context" / "manifest.json").read_text())
    assert on_disk["datasets"]["ghcn"]["rows"] == 0


def test_the_cli_entry_runs_a_step(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cm, "build_ghcn", lambda out: {"rows": 3, "cells": []})
    assert cm.main(["ghcn", "--out", str(tmp_path)]) == 0
    assert json.load(io.StringIO(capsys.readouterr().out)) == {"rows": 3}


def test_a_smoke_run_of_the_workflow_never_publishes():
    yaml = pytest.importorskip("yaml")
    path = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "mirror-context.yml"
    wf = yaml.safe_load(path.read_text())
    steps = wf["jobs"]["publish"]["steps"]
    publish = next(s for s in steps if "context_mirror publish" in str(s.get("run", "")))
    cond = publish["if"]
    assert "HF_TOKEN" in cond and "inputs.publish" in cond
    assert "inputs.max_files == ''" in cond and "inputs.groundsource_max_rows == ''" in cond


def test_floods_monthly_rolls_both_flood_mirrors_into_the_grid(tmp_path):
    """Floods past (#547): the grid, the index and one gzipped JSON per month, all under context/floods/monthly/."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    src = tmp_path / "src"
    src.mkdir()
    pq.write_table(pa.table({"start_date": ["2021-07-14", "2021-07-15", "2024-01-02"], "lat": [50.61, 50.62, -6.2],
                             "lon": [5.61, 5.70, 106.8]}), src / "gs.parquet")
    pq.write_table(pa.table({"lat": [50.625, 10.025], "lon": [5.625, 10.025], "year": [2021, 2019],
                             "month": [7, 12], "n": [120, 5]}), src / "ms.parquet")
    stale = tmp_path / "context" / "floods" / "monthly" / "months" / "1999-01.json.gz"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    info = cm.build_floods_monthly(tmp_path, groundsource=src / "gs.parquet", microsoft=src / "ms.parquet")
    root = tmp_path / "context" / "floods" / "monthly"
    assert info["rows"] == 3 and info["months"] == 3 and info["first"] == "2019-12" and info["last"] == "2024-01"
    assert info["licence"] == "CC-BY-4.0 (news), MIT (radar)" and info["folder"] == "floods/monthly"
    assert not stale.exists()  # a rebuilt grid has no months left over from the last one
    assert sorted(p.name for p in (root / "months").iterdir()) == ["2019-12.json.gz", "2021-07.json.gz",
                                                                   "2024-01.json.gz"]
    july = json.loads(gzip.decompress((root / "months" / "2021-07.json.gz").read_bytes()))
    assert july["cells"] == [[281, 371, 2, 120]] and july["deg"] == 0.5
    index = json.loads((root / "index.json").read_text())
    assert index["news"] == {"first": "2021-07-14", "last": "2024-01-02"}
    assert pq.read_table(root / "grid.parquet").num_rows == 3
    assert info["bytes"]["months"] > 0 and (tmp_path / "context" / "_floods_monthly.json").exists()
    # every file it wrote is one the context publish uploads
    from fnmatch import fnmatch

    for p in root.rglob("*"):
        if p.is_file():
            assert any(fnmatch(p.name, pat) for pat in cm.PUBLISH_PATTERNS), p.name


def test_floods_monthly_reads_what_this_run_built_before_the_archive(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    floods = tmp_path / "context" / "floods"
    floods.mkdir(parents=True)
    pq.write_table(pa.table({"start_date": ["2021-07-14"], "lat": [1.0], "lon": [1.0]}),
                   floods / "groundsource.parquet")
    fetched = []

    def fake_download(url, dest, **kw):
        fetched.append(url)
        pq.write_table(pa.table({"lat": [1.0], "lon": [1.0], "year": [2021], "month": [7], "n": [4]}), dest)
        return dest

    monkeypatch.setattr(cm, "download", fake_download)
    info = cm.build_floods_monthly(tmp_path, repo_id="me/test")
    assert fetched == ["https://huggingface.co/datasets/me/test/resolve/main/context/floods/microsoft.parquet"]
    assert info["rows"] == 1


def test_floods_monthly_leaves_nothing_half_built_when_it_fails(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aquascope.context import floods_past as fp

    src = tmp_path / "src"
    src.mkdir()
    pq.write_table(pa.table({"start_date": ["2021-07-14"], "lat": [1.0], "lon": [1.0]}), src / "gs.parquet")
    pq.write_table(pa.table({"lat": [1.0], "lon": [1.0], "year": [2021], "month": [7], "n": [4]}), src / "ms.parquet")
    cm.build_floods_monthly(tmp_path, groundsource=src / "gs.parquet", microsoft=src / "ms.parquet")
    before = (tmp_path / "context" / "floods" / "monthly" / "index.json").read_text()

    def boom(payload):
        raise OSError("disk full")

    monkeypatch.setattr(fp, "encode_month", boom)
    with pytest.raises(OSError):
        cm.build_floods_monthly(tmp_path, groundsource=src / "gs.parquet", microsoft=src / "ms.parquet")
    floods = tmp_path / "context" / "floods"
    assert sorted(p.name for p in floods.iterdir()) == ["monthly"]  # no partial folder to publish
    assert (floods / "monthly" / "index.json").read_text() == before


def test_the_workflow_builds_the_monthly_grid_and_can_build_only_it():
    yaml = pytest.importorskip("yaml")
    path = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "mirror-context.yml"
    wf = yaml.safe_load(path.read_text())
    trigger = wf.get("on") or wf.get(True)
    assert trigger["workflow_dispatch"]["inputs"]["floods_monthly_only"]["default"] is False
    steps = wf["jobs"]["publish"]["steps"]
    runs = [str(s.get("run", "")) for s in steps]
    grid = next(i for i, r in enumerate(runs) if "context_mirror floods-monthly" in r)
    assert grid < next(i for i, r in enumerate(runs) if "context_mirror manifest" in r)
    assert grid > next(i for i, r in enumerate(runs) if "microsoft-merge" in r)
    assert "floods_monthly_only" in wf["jobs"]["small"]["if"] and "floods_monthly_only" in wf["jobs"]["microsoft"]["if"]
    assert "inputs.floods_monthly_only" in wf["jobs"]["publish"]["if"]
