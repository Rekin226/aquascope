"""What gets flooded (#622): buildings, roads and people under the review's synthetic flood, from synthetic
footprints, a synthetic road and a synthetic population grid; and the readers, on in-memory files.

The grid is 10 x 8 cells of 1 km on a flat bed; the west half (columns 0 to 4) is wet from day 2 to the end of the
eight-day run, deepest on day 4 (tests/test_mesh_review.py).
"""

from __future__ import annotations

import io
import json
import struct

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("pyproj")

from aquascope import flood_exposure as X  # noqa: E402, N812
from aquascope.io import mesh as M  # noqa: E402, N812
from tests.cog_builder import build_tiff  # noqa: E402
from tests.test_mesh_review import NX, STEPS, cell, make_run  # noqa: E402


@pytest.fixture
def m(tmp_path):
    mesh = M.read_mesh(make_run(tmp_path / "run.p01.hdf"))
    yield mesh
    mesh.close()


def square(lon, lat, half=0.001):
    ring = np.array([[lon - half, lat - half], [lon + half, lat - half], [lon + half, lat + half],
                     [lon - half, lat + half], [lon - half, lat - half]])
    return {"id": f"b{lon:.4f}", "class": "house", "parts": [[ring]]}


def ll(m, i):
    return float(m.lon[i]), float(m.lat[i])


def edge_between(m, a, b):
    return (m.lon[a] + m.lon[b]) / 2, (m.lat[a] + m.lat[b]) / 2


# ── geometry ────────────────────────────────────────────────────────────────


def wkb_polygon(rings, bo="<"):
    flag = 1 if bo == "<" else 0
    out = struct.pack(bo + "BI", flag, 3) if bo == "<" else bytes([0]) + struct.pack(">I", 3)
    out += struct.pack(bo + "I", len(rings))
    for r in rings:
        out += struct.pack(bo + "I", len(r)) + struct.pack(bo + "d" * (2 * len(r)), *np.asarray(r, "f8").ravel())
    _ = flag
    return out


def wkb_line(pts, *, z=False):
    code = 1002 if z else 2
    dims = 3 if z else 2
    arr = np.asarray(pts, "f8")
    if z:
        arr = np.column_stack([arr, np.full(len(arr), 7.0)])
    return struct.pack("<BII", 1, code, len(arr)) + struct.pack("<" + "d" * (dims * len(arr)), *arr.ravel())


def test_wkb_polygons_lines_and_multis():
    ring = [[0, 0], [1, 0], [1, 1], [0, 0]]
    kind, parts = X.wkb_parts(wkb_polygon([ring]))
    assert kind == "polygon" and np.allclose(parts[0][0], ring)
    kind, parts = X.wkb_parts(wkb_polygon([ring], bo=">"))
    assert kind == "polygon" and np.allclose(parts[0][0], ring)
    multi = struct.pack("<BII", 1, 6, 2) + wkb_polygon([ring]) + wkb_polygon([ring, ring])
    kind, parts = X.wkb_parts(multi)
    assert kind == "polygon" and len(parts) == 2 and len(parts[1]) == 2
    kind, parts = X.wkb_parts(wkb_line([[0, 0], [2, 3]], z=True))
    assert kind == "line" and np.allclose(parts[0][0], [[0, 0], [2, 3]])
    with pytest.raises(ValueError):
        X.wkb_parts(struct.pack("<BI", 1, 99))


# ── the counts ──────────────────────────────────────────────────────────────


def test_buildings_count_when_water_reaches_a_cell_under_them(m):
    wet = square(*ll(m, cell(3, 1)))                               # in the flooded half
    dry = square(*ll(m, cell(3, 8)))                               # in the dry half
    lon, lat = edge_between(m, cell(5, 4), cell(5, 5))
    straddle = square(lon, lat, half=0.003)                        # half in a wet cell, half in a dry one
    w, s, e, n = m.bbox()
    off = square(e + 0.05, n + 0.05)                               # not on the model
    res = X.exposure(m, buildings=[wet, dry, straddle, off])
    b = res["buildings"]
    assert b["on_mesh"] == 3 and b["read"] == 4
    # day 1 is dry everywhere; from day 2 the west half is wet to the end of the run
    assert b["per_step"] == [0] + [2] * (STEPS - 1)
    assert b["ever"] == 2
    assert b["peak"] == {"value": 2.0, "step": 1, "time": res["times"][1]}
    by_id = {f["id"]: f for f in b["features"]}
    assert by_id[wet["id"]]["wet"] == "0" + "1" * (STEPS - 1)
    assert by_id[dry["id"]]["wet"] == "0" * STEPS and by_id[dry["id"]]["max_depth"] is None
    assert by_id[straddle["id"]]["max_depth"] == pytest.approx(1.6 * (1 + 0.4 + 0.05), abs=1e-3)
    assert off["id"] not in by_id


def test_a_higher_threshold_counts_fewer_steps(m):
    wet = square(*ll(m, cell(3, 1)))
    res = X.exposure(m, buildings=[wet], threshold=1.0)
    # depth there: SHAPE x 1.13, over 1 m on days 3, 4 and 5 only
    assert res["buildings"]["per_step"] == [0, 0, 1, 1, 1, 0, 0, 0]
    assert res["threshold"] == 1.0


def test_road_length_counts_the_stretches_in_flooded_cells(m):
    a, b = ll(m, cell(3, 0)), ll(m, cell(3, NX - 1))
    road = {"id": "r1", "class": "primary", "parts": [[np.array([a, b])]]}
    res = X.exposure(m, roads=[road])
    r = res["roads"]
    assert r["on_mesh_km"] == pytest.approx(9.0, rel=0.01)
    # from column 0's centre to the edge between columns 4 and 5: 4.5 km
    assert r["per_step_km"][0] == 0
    assert r["per_step_km"][3] == pytest.approx(4.5, rel=0.02)
    assert r["ever_km"] == pytest.approx(4.5, rel=0.02)
    assert r["segments_flooded"] == 1
    assert all(p["wet"][3] == "1" for p in r["pieces"])
    assert sum(p["length_m"] for p in r["pieces"]) == pytest.approx(4500, rel=0.02)


def test_people_are_counted_where_their_grid_square_is_flooded(m):
    (lw, aw), (ld, ad) = ll(m, cell(4, 2)), ll(m, cell(4, 7))
    pop = {"lon": np.array([lw, ld]), "lat": np.array([aw, ad]), "people": np.array([90.0, 40.0]),
           "step": (0.002, -0.002)}
    res = X.exposure(m, population=pop)
    p = res["population"]
    assert p["per_step"][0] == 0
    assert p["per_step"][3] == pytest.approx(90.0)
    assert p["ever"] == pytest.approx(90.0) and p["on_mesh"] == pytest.approx(130.0)


def test_exposure_needs_depths(tmp_path):
    m = M.read_mesh(make_run(tmp_path / "ws.p01.hdf", bed=False))
    try:
        with pytest.raises(M.MeshError, match="no bed levels"):
            X.exposure(m, buildings=[])
    finally:
        m.close()


def test_the_payload_is_plain_json_with_sources_and_limits(m):
    read = {
        "buildings": [{"id": "b1", "class": None, "parts": [[square(*ll(m, cell(3, 1)))["parts"][0][0].tolist()]]}],
        "roads": [{"id": "r1", "class": "residential",
                   "parts": [[[list(ll(m, cell(3, 0))), list(ll(m, cell(3, 3)))]]]}],
        "population": {"lon": [ll(m, cell(4, 2))[0]], "lat": [ll(m, cell(4, 2))[1]], "people": [12.0],
                       "step": [0.002, -0.002]},
        "info": {"buildings": {"release": "2026-10-21.0"}}, "errors": {"roads": "nothing"},
    }
    res = X.exposure_payload(m, read)
    json.dumps(res)   # plain types only
    assert res["sources"]["buildings"]["licence"] == "ODbL 1.0"
    assert res["sources"]["buildings"]["release"] == "2026-10-21.0"
    assert "OpenStreetMap contributors" in res["sources"]["roads"]["credit"]
    assert res["sources"]["population"]["licence"] == "CC BY 4.0"
    assert set(res["limits"]) == {"buildings", "roads", "population"}
    assert res["errors"] == {"roads": "nothing"}
    assert isinstance(res["buildings"]["features"][0]["ring"][0], list)


# ── the readers ─────────────────────────────────────────────────────────────


class Files:
    """``fetch(url, start, end)`` over in-memory files, counting the bytes read, with ``size_of``."""

    def __init__(self, files):
        self.files, self.calls = files, []

    def __call__(self, url, start, end):
        self.calls.append((url, start, end))
        return self.files[url][start:end + 1]

    def size_of(self, url):
        return len(self.files[url])


def overture_parquet(rows, *, row_group_size=4):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    geoms, boxes = [], []
    for r in rows:
        pts = np.asarray(r["coords"], "f8")
        geoms.append(wkb_polygon([pts]) if r["kind"] == "polygon" else wkb_line(pts))
        boxes.append({"xmin": float(pts[:, 0].min()), "xmax": float(pts[:, 0].max()),
                      "ymin": float(pts[:, 1].min()), "ymax": float(pts[:, 1].max())})
    table = pa.table({
        "id": [r["id"] for r in rows], "geometry": pa.array(geoms, pa.binary()),
        "bbox": pa.array(boxes, pa.struct([("xmin", pa.float32()), ("xmax", pa.float32()), ("ymin", pa.float32()),
                                           ("ymax", pa.float32())])),
        "class": [r.get("class") for r in rows], "subtype": [r.get("subtype") for r in rows],
        "height": [None] * len(rows),
    })
    buf = io.BytesIO()
    pq.write_table(table, buf, row_group_size=row_group_size, compression="zstd")
    return buf.getvalue()


def box_at(x, y, h=0.1):
    return [[x - h, y - h], [x + h, y - h], [x + h, y + h], [x - h, y + h], [x - h, y - h]]


def test_read_overture_keeps_the_box_and_reads_only_its_row_groups():
    # three row groups of four: buildings near (1, 1), near (5, 5) and near (9, 9)
    rows = [{"id": f"b{i}", "kind": "polygon", "coords": box_at(1 + 4 * (i // 4), 1 + 4 * (i // 4) + 0.01 * i),
             "class": "house"} for i in range(12)]
    blob = overture_parquet(rows)
    files = Files({"https://x/b.parquet": blob})
    out = X.read_overture([4, 4, 6, 6], "buildings", files=["https://x/b.parquet"], release="test", fetch=files)
    assert sorted(f["id"] for f in out["features"]) == [f"b{i}" for i in range(4, 8)]
    assert out["features"][0]["kind"] == "polygon" and out["features"][0]["class"] == "house"
    assert out["truncated"] is False
    everything = X.read_overture([0, 0, 10, 10], "buildings", files=["https://x/b.parquet"], release="t",
                                 fetch=Files({"https://x/b.parquet": blob}))
    assert len(everything["features"]) == 12
    # only the middle row group's statistics meet the box, so only it is read
    pq = pytest.importorskip("pyarrow.parquet")
    md = pq.ParquetFile(io.BytesIO(blob)).metadata
    assert md.num_row_groups == 3 and X._row_groups(md, [4, 4, 6, 6]) == [1]
    capped = X.read_overture([0, 0, 10, 10], "buildings", files=["https://x/b.parquet"], release="t", fetch=files,
                             limit=5)
    assert len(capped["features"]) == 5 and capped["truncated"] is True


def test_read_overture_roads_are_the_road_segments():
    rows = [{"id": "r1", "kind": "line", "coords": [[1, 1], [1.2, 1.1]], "subtype": "road", "class": "primary"},
            {"id": "t1", "kind": "line", "coords": [[1, 1], [1.1, 1.3]], "subtype": "rail", "class": "standard_gauge"},
            {"id": "r2", "kind": "line", "coords": [[1.1, 1], [1.3, 1.2]], "subtype": "road", "class": "residential"},
            {"id": "f1", "kind": "line", "coords": [[1.1, 1], [1.2, 1.2]], "subtype": "road", "class": "footway"},
            {"id": "c1", "kind": "line", "coords": [[1.1, 1], [1.2, 1.3]], "subtype": "road", "class": "cycleway"},
            {"id": "u1", "kind": "line", "coords": [[1.2, 1], [1.2, 1.3]], "subtype": "road", "class": None}]
    files = Files({"https://x/s.parquet": overture_parquet(rows)})
    out = X.read_overture([0, 0, 2, 2], "roads", files=["https://x/s.parquet"], release="t", fetch=files)
    # roads for vehicles: footways and cycleways are left out, a road of no class stays
    assert [f["id"] for f in out["features"]] == ["r1", "r2", "u1"]
    assert out["features"][0]["kind"] == "line"


def test_overture_files_picks_the_files_whose_boxes_meet():
    base = "https://stac.overturemaps.org/2026-10-21.0/buildings/building"
    docs = {
        X.STAC_ROOT: {"latest": "2026-10-21.0", "links": []},
        f"{base}/collection.json": {
            "extent": {"spatial": {"bbox": [[-180, -90, 180, 90], [0, 0, 10, 10], [10, 0, 20, 10], [0, 10, 10, 20]]}},
            "links": [{"rel": "root"}] + [{"rel": "item", "href": f"{base}/{i:05d}/{i:05d}.json"} for i in range(3)],
        },
        **{f"{base}/{i:05d}/{i:05d}.json": {"assets": {"aws": {"href": f"https://s3/part-{i}.parquet"}}}
           for i in range(3)},
    }
    got = X.overture_files([8, 8, 12, 9], "buildings", "building", get_json=docs.__getitem__)
    assert got == {"release": "2026-10-21.0", "files": ["https://s3/part-0.parquet", "https://s3/part-1.parquet"]}


@pytest.mark.parametrize("lon, lat, tile", [(14.29, 48.30, "R5_C20"), (2.35, 48.85, "R5_C19"),
                                            (-58.4, -34.6, "R13_C13"),
                                            # the tiles start 0.0079 degrees west of the round degree
                                            (9.995, 48.3, "R5_C20"), (9.99, 48.3, "R5_C19")])
def test_ghsl_tiles_are_ten_degree_squares(lon, lat, tile):
    assert X.ghsl_tile_url(lon, lat).endswith(f"_{tile}.tif")


def test_read_population_reads_the_window_of_the_grid():
    # a 40 x 40 grid of 0.01 degree squares from 14.0 E, 48.4 N, people = row * 100 + col
    arr = (np.arange(40)[:, None] * 100 + np.arange(40)[None, :]).astype("f8")
    arr[0, 0] = -200.0   # nodata
    tif = build_tiff(arr, tile=(16, 16), geotransform=(14.0, 0.01, 48.4, -0.01), nodata=-200.0)
    url = X.ghsl_tile_url(14.1, 48.3)
    files = Files({url: tif})
    pop = X.read_population([14.105, 48.255, 14.135, 48.275], fetch=files)
    assert pop["tiles"] == [url]
    assert pop["step"] == pytest.approx((0.01, -0.01))
    # columns 10 to 13, rows 12 to 14: centres 14.105 ... 14.135 and 48.275 ... 48.255
    assert sorted(set(np.round(pop["lon"], 3))) == pytest.approx([14.105, 14.115, 14.125, 14.135])
    assert sorted(set(np.round(pop["lat"], 3))) == pytest.approx([48.255, 48.265, 48.275])
    k = int(np.argmin(np.hypot(pop["lon"] - 14.115, pop["lat"] - 48.265)))
    assert pop["people"][k] == 13 * 100 + 11


def test_a_window_across_tiles_comes_back_whole():
    from aquascope.utils.cog import open_cog

    arr = (np.arange(40)[:, None] * 100 + np.arange(40)[None, :]).astype("f8")
    arr[15, 15] = -200.0
    tif = build_tiff(arr, tile=(16, 16), geotransform=(0.0, 1.0, 40.0, -1.0), nodata=-200.0)
    files = Files({"https://x/p.tif": tif})
    win = open_cog("https://x/p.tif", fetch=files).read_window(14, 13, 17, 17)
    want = arr[13:18, 14:18].copy()
    want[want == -200.0] = np.nan
    assert win.shape == (5, 4)
    assert np.allclose(win, want, equal_nan=True)


def test_reading_errors_are_said_and_the_rest_counts(m, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no network here")

    monkeypatch.setattr(X, "read_overture", boom)
    monkeypatch.setattr(X, "read_population", lambda bbox: {"lon": np.zeros(0), "lat": np.zeros(0),
                                                            "people": np.zeros(0), "step": None, "tiles": []})
    res = X.exposure_for_mesh(m)
    assert set(res["errors"]) == {"buildings", "roads"} and "no network" in res["errors"]["roads"]
    assert "buildings" not in res and res["population"]["per_step"] == [0.0] * STEPS


def test_a_model_too_large_to_read_for_is_refused():
    with pytest.raises(M.MeshError, match="km²"):
        X.read_sources([0, 0, 1, 1])


def test_the_threshold_can_be_given_in_metres(m):
    wet = square(*ll(m, cell(3, 1)))
    a = X.exposure(m, buildings=[wet], threshold_m=1.0)
    assert a["buildings"]["per_step"] == [0, 0, 1, 1, 1, 0, 0, 0]
    assert a["threshold"] == 1.0 and a["threshold_m"] == 1.0
    assert set(X.THRESHOLDS_M) == {M.DRY_DEPTH, 0.15, 0.30}
